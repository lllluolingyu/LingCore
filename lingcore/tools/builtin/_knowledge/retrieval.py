"""Lexical, semantic, and hybrid retrieval and status reporting."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import replace
from typing import Any, Literal, Mapping, Sequence

from lingcore.errors import ToolError
from lingcore.knowledge import (
    embedding_enabled,
    embedding_identity,
    reranker_enabled,
    reranker_options,
)
from lingcore.tools import ToolContext
from lingcore.tools.builtin._knowledge.common import (
    _DEFAULT_CANDIDATE_MULTIPLIER,
    _DEFAULT_CONTEXT_CHARS,
    _DEFAULT_EXCERPT_CHARS,
    _DEFAULT_RRF_K,
    _MAX_HITS,
    _MAX_INDEX_SOURCE_BYTES,
    EMBEDDING_PROVIDER_KEY,
    _embedder,
    _float_option,
    _int_option,
    _normalize_vector,
    _provider_identity,
    _reranker,
    _SearchHit,
    _unpack_vector,
)
from lingcore.tools.builtin._knowledge.sources import (
    _chunk_config,
    _iter_source_files,
    _stale_state,
)
from lingcore.tools.builtin._knowledge.storage import (
    _index_path,
    _open_index,
)


def _row_hit(
    row: sqlite3.Row, score: float, *, semantic: float | None = None
) -> _SearchHit:
    return _SearchHit(
        chunk_id=int(row["id"]),
        path=str(row["path"]),
        ordinal=int(row["ordinal"]),
        page=int(row["page"]) if row["page"] is not None else None,
        line_start=int(row["line_start"]),
        line_end=int(row["line_end"]),
        text=str(row["text"]),
        score=score,
        semantic_score=semantic,
    )


def _fts_expression(query: str) -> str | None:
    terms = list(dict.fromkeys(re.findall(r"\w+", query, flags=re.UNICODE)))
    if not terms:
        return None
    escaped = [term.replace('"', '""') for term in terms[:32]]
    return " OR ".join(f'"{term}"' for term in escaped)


def _lexical_hits(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    excluded: frozenset[str],
) -> list[_SearchHit]:
    expression = _fts_expression(query)
    if expression is None:
        return []
    params: list[Any] = [expression]
    exclusion = ""
    if excluded:
        marks = ",".join("?" for _ in excluded)
        exclusion = f" AND c.path NOT IN ({marks})"
        params.extend(sorted(excluded))
    params.append(limit)
    try:
        rows = connection.execute(
            f"""
            SELECT c.id, c.path, c.ordinal, c.page, c.line_start, c.line_end,
                   c.text, bm25(chunks_fts) AS lexical_rank
            FROM chunks_fts
            JOIN chunks AS c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?{exclusion}
            ORDER BY lexical_rank, c.path, c.ordinal
            LIMIT ?
            """,
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [_row_hit(row, 1.0 / rank) for rank, row in enumerate(rows, start=1)]


def _semantic_hits(
    connection: sqlite3.Connection,
    query_vector: Sequence[float],
    identity: str,
    *,
    limit: int,
    excluded: frozenset[str],
) -> list[_SearchHit]:
    rows = connection.execute(
        """
        SELECT id, path, ordinal, page, line_start, line_end, text,
               embedding_dim, embedding
        FROM chunks
        WHERE embedding_identity = ? AND embedding IS NOT NULL
        """,
        (identity,),
    ).fetchall()
    hits: list[_SearchHit] = []
    for row in rows:
        if row["path"] in excluded:
            continue
        stored = _unpack_vector(bytes(row["embedding"]), int(row["embedding_dim"]))
        if stored is None or len(stored) != len(query_vector):
            continue
        score = sum(left * right for left, right in zip(query_vector, stored))
        hits.append(_row_hit(row, score, semantic=score))
    hits.sort(key=lambda hit: (-hit.score, hit.path, hit.ordinal))
    return hits[:limit]


def _hybrid_hits(
    lexical: Sequence[_SearchHit],
    semantic: Sequence[_SearchHit],
    *,
    lexical_weight: float,
    semantic_weight: float,
    rrf_k: int,
    limit: int,
) -> list[_SearchHit]:
    by_id: dict[int, _SearchHit] = {}
    scores: dict[int, float] = {}
    semantic_scores: dict[int, float] = {}
    for rank, hit in enumerate(lexical, start=1):
        by_id[hit.chunk_id] = hit
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (
            lexical_weight / (rrf_k + rank)
        )
    for rank, hit in enumerate(semantic, start=1):
        by_id.setdefault(hit.chunk_id, hit)
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (
            semantic_weight / (rrf_k + rank)
        )
        semantic_scores[hit.chunk_id] = hit.score
    merged = [
        replace(
            hit,
            score=scores[chunk_id],
            semantic_score=semantic_scores.get(chunk_id),
        )
        for chunk_id, hit in by_id.items()
    ]
    merged.sort(key=lambda hit: (-hit.score, hit.path, hit.ordinal))
    return merged[:limit]


async def _query_index(
    ctx: ToolContext,
    options: Mapping[str, Any],
    sources: Sequence[str],
    query: str,
    backend: Literal["index", "hybrid"],
) -> str:
    base = ctx.workspace.resolve()
    path = _index_path(base, options)
    provider = _embedder(ctx, options)
    identity = _provider_identity(provider)
    connection = _open_index(base, path, create=False)
    if connection is None:
        raise ToolError("knowledge index is empty; run knowledge action=index first")
    _, _, config_fingerprint = _chunk_config(options)
    max_hits = _int_option(options, "max_hits", _MAX_HITS, minimum=1, maximum=100)
    candidate_limit = max(
        max_hits,
        max_hits
        * _int_option(
            options,
            "candidate_multiplier",
            _DEFAULT_CANDIDATE_MULTIPLIER,
            minimum=1,
            maximum=20,
        ),
    )
    try:
        stale = _stale_state(
            connection,
            base,
            sources,
            config_fingerprint,
            _int_option(
                options,
                "max_source_bytes",
                _MAX_INDEX_SOURCE_BYTES,
                minimum=1,
                maximum=100 * 1024 * 1024,
            ),
            path,
        )
        vectors = await provider.embed([query])
        if len(vectors) != 1:
            raise ToolError("embedding provider returned the wrong vector count")
        query_vector = _normalize_vector(vectors[0])
        semantic = _semantic_hits(
            connection,
            query_vector,
            identity,
            limit=candidate_limit,
            excluded=stale.excluded,
        )
        if not semantic:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE embedding_identity = ? "
                    "AND embedding IS NOT NULL",
                    (identity,),
                ).fetchone()[0]
            )
            total = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            if total and not count:
                raise ToolError(
                    "knowledge index has no usable vectors for the configured "
                    "embedding model; run knowledge action=index"
                )
        if backend == "index":
            hits = semantic[:candidate_limit]
        else:
            lexical = _lexical_hits(
                connection,
                query,
                limit=candidate_limit,
                excluded=stale.excluded,
            )
            hits = _hybrid_hits(
                lexical,
                semantic,
                lexical_weight=_float_option(
                    options, "lexical_weight", 1.0, minimum=0.0
                ),
                semantic_weight=_float_option(
                    options, "semantic_weight", 1.0, minimum=0.0
                ),
                rrf_k=_int_option(
                    options, "rrf_k", _DEFAULT_RRF_K, minimum=1, maximum=1_000
                ),
                limit=candidate_limit,
            )

        if reranker_enabled(options) and hits:
            cfg = reranker_options(options)
            candidate_count = _int_option(
                cfg,
                "candidate_count",
                min(20, candidate_limit),
                minimum=1,
                maximum=100,
                option_path="tool_options.knowledge.reranker",
            )
            candidates = hits[:candidate_count]
            ranked = await _reranker(ctx, options).rerank(
                query,
                [hit.text for hit in candidates],
                top_n=min(max_hits, len(candidates)),
            )
            hits = [
                replace(candidates[result.index], score=result.score)
                for result in ranked
            ]

        result = _render_hits(hits[:max_hits], options)
        if stale.count:
            warning = (
                f"[index stale: changed={len(stale.changed)}, "
                f"deleted={len(stale.deleted)}, new={len(stale.new)}; "
                "stale sources were excluded — run knowledge action=index]"
            )
            return f"{warning}\n{result}"
        return result
    finally:
        connection.close()


def _render_hits(hits: Sequence[_SearchHit], options: Mapping[str, Any]) -> str:
    if not hits:
        return "(no matches)"
    excerpt_chars = _int_option(
        options,
        "max_excerpt_chars",
        _DEFAULT_EXCERPT_CHARS,
        minimum=100,
        maximum=8_000,
    )
    context_chars = _int_option(
        options,
        "max_context_chars",
        _DEFAULT_CONTEXT_CHARS,
        minimum=500,
        maximum=100_000,
    )
    blocks: list[str] = []
    used = 0
    for rank, hit in enumerate(hits, start=1):
        if hit.page is None:
            location = (
                f"{hit.path}:{hit.line_start}"
                if hit.line_start == hit.line_end
                else f"{hit.path}:{hit.line_start}-{hit.line_end}"
            )
        else:
            lines = (
                str(hit.line_start)
                if hit.line_start == hit.line_end
                else f"{hit.line_start}-{hit.line_end}"
            )
            location = f"{hit.path}#page={hit.page}:lines={lines}"
        excerpt = hit.text[:excerpt_chars]
        if len(hit.text) > excerpt_chars:
            excerpt += "…"
        block = f"[{rank}] {location} (score={hit.score:.4f})\n{excerpt}"
        separator = 2 if blocks else 0
        remaining = context_chars - used - separator
        if remaining <= 0:
            break
        if len(block) > remaining:
            if not blocks:
                block = block[: max(0, remaining - 1)] + "…"
            else:
                break
        blocks.append(block)
        used += separator + len(block)
    return "\n\n".join(blocks) if blocks else "(no matches)"


def _status(
    ctx: ToolContext,
    options: Mapping[str, Any],
    sources: Sequence[str],
    backend: Literal["grep", "index", "hybrid"],
) -> str:
    base = ctx.workspace.resolve()
    path = _index_path(base, options) if backend != "grep" else None
    source_count = sum(1 for _ in _iter_source_files(base, sources, index_path=path))
    enabled = embedding_enabled(options)
    prefix = (
        f"backend={backend}, embedding={'true' if enabled else 'false'}, "
        f"source files={source_count}"
    )
    if backend == "grep":
        return prefix
    assert path is not None
    connection = _open_index(base, path, create=False)
    if connection is None:
        return f"{prefix}, indexed files=0, chunks=0, embedded chunks=0, stale files=0"
    try:
        _, _, config_fingerprint = _chunk_config(options)
        stale = _stale_state(
            connection,
            base,
            sources,
            config_fingerprint,
            _int_option(
                options,
                "max_source_bytes",
                _MAX_INDEX_SOURCE_BYTES,
                minimum=1,
                maximum=100 * 1024 * 1024,
            ),
            path,
        )
        indexed = int(
            connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        identity: str | None = None
        injected = ctx.options.get(EMBEDDING_PROVIDER_KEY)
        if injected is not None:
            identity = _provider_identity(injected)
        elif enabled:
            identity = embedding_identity(options)
        embedded = 0
        if identity is not None:
            embedded = int(
                connection.execute(
                    """
                SELECT COUNT(*) FROM chunks
                WHERE embedding_identity = ? AND embedding IS NOT NULL
                """,
                    (identity,),
                ).fetchone()[0]
            )
        return (
            f"{prefix}, indexed files={indexed}, chunks={chunks}, "
            f"embedded chunks={embedded}, stale files={stale.count}"
        )
    finally:
        connection.close()
