"""knowledge — grep, semantic, and hybrid retrieval over workspace files.

``grep`` remains the zero-cost, offline default.  The ``index`` and ``hybrid``
backends use a local incremental SQLite index; semantic vectors are created
only after the profile explicitly sets ``embedding.enabled: true``.  The model
provider is injected through ``ToolContext`` (or lazily built from config), so
the index and tool never depend on one vendor's SDK.

The index stores deterministic text chunks with source path, page, and line
metadata.  Content hashes avoid re-embedding unchanged chunks, full refreshes
remove deleted sources, and queries exclude changed/deleted documents until the
operator indexes again rather than returning stale citations.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import os
import re
import secrets
import sqlite3
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

from pydantic import BaseModel, Field

from lingcore.errors import ConfigError, ToolError
from lingcore.knowledge import (
    EmbeddingProvider,
    RerankingProvider,
    build_embedding_provider,
    build_reranking_provider,
    embedding_enabled,
    embedding_identity,
    reranker_enabled,
    reranker_options,
)
from lingcore.paths import PathEscapeError, confined_directory
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin._offload import RUNTIME_DIRNAME

_MAX_READ_BYTES = 256 * 1024
_MAX_INDEX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_HITS = 50
_GREP_LINE_CHARS = 200
_DEFAULT_SOURCES = ["**/*"]
_DEFAULT_CHUNK_CHARS = 1_600
_DEFAULT_CHUNK_OVERLAP_LINES = 2
_DEFAULT_EXCERPT_CHARS = 1_000
_DEFAULT_CONTEXT_CHARS = 8_000
_DEFAULT_CANDIDATE_MULTIPLIER = 4
_DEFAULT_RRF_K = 60
_DEFAULT_INDEX_PATH = f"{RUNTIME_DIRNAME}/knowledge.sqlite3"
_MAX_INDEX_BYTES = 512 * 1024 * 1024
_SCHEMA_VERSION = 1

# Injection keys are public so alternate composition roots and tests can supply
# provider implementations without placing live Python objects inside YAML.
EMBEDDING_PROVIDER_KEY = "_knowledge_embedding_provider"
RERANKING_PROVIDER_KEY = "_knowledge_reranking_provider"


@dataclass(frozen=True, slots=True)
class _ChunkDraft:
    chunk_key: str
    ordinal: int
    page: int | None
    line_start: int
    line_end: int
    text_hash: str
    text: str


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    path: str
    content_hash: str
    size: int
    mtime_ns: int
    chunks: tuple[_ChunkDraft, ...]


@dataclass(frozen=True, slots=True)
class _SearchHit:
    chunk_id: int
    path: str
    ordinal: int
    page: int | None
    line_start: int
    line_end: int
    text: str
    score: float
    semantic_score: float | None = None


@dataclass(frozen=True, slots=True)
class _StaleState:
    changed: frozenset[str]
    deleted: frozenset[str]
    new: frozenset[str]

    @property
    def count(self) -> int:
        return len(self.changed | self.deleted | self.new)

    @property
    def excluded(self) -> frozenset[str]:
        return self.changed | self.deleted


def _confined(base: Path, path: Path) -> bool:
    return path == base or path.is_relative_to(base)


def _knowledge_options(ctx: ToolContext) -> dict[str, Any]:
    raw = ctx.options.get("knowledge", {})
    if not isinstance(raw, Mapping):
        raise ConfigError("tool_options.knowledge must be a mapping")
    return dict(raw)


def _backend(options: Mapping[str, Any]) -> Literal["grep", "index", "hybrid"]:
    value = options.get("backend", "grep")
    if value not in {"grep", "index", "hybrid"}:
        raise ConfigError(
            "tool_options.knowledge.backend must be grep, index, or hybrid"
        )
    return value


def _sources(options: Mapping[str, Any]) -> list[str]:
    raw = options.get("sources", _DEFAULT_SOURCES)
    if not isinstance(raw, list) or not raw or any(
        not isinstance(value, str) or not value.strip() for value in raw
    ):
        raise ConfigError(
            "tool_options.knowledge.sources must be a non-empty list of globs"
        )
    return [value.strip() for value in raw]


def _int_option(
    options: Mapping[str, Any],
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    option_path: str = "tool_options.knowledge",
) -> int:
    raw = options.get(name, default)
    if isinstance(raw, bool):
        raise ConfigError(f"{option_path}.{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"{option_path}.{name} must be an integer") from None
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ConfigError(
            f"{option_path}.{name} must be >= {minimum}{upper}"
        )
    return value


def _float_option(
    options: Mapping[str, Any], name: str, default: float, *, minimum: float = 0.0
) -> float:
    raw = options.get(name, default)
    if isinstance(raw, bool):
        raise ConfigError(f"tool_options.knowledge.{name} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"tool_options.knowledge.{name} must be a number"
        ) from None
    if not math.isfinite(value) or value < minimum:
        raise ConfigError(
            f"tool_options.knowledge.{name} must be >= {minimum}"
        )
    return value


def _iter_source_files(base: Path, sources: Sequence[str]) -> Iterator[Path]:
    """Yield each confined source once, deterministically."""
    seen: set[Path] = set()
    for pattern in sources:
        pat = Path(pattern)
        if pat.is_absolute() or any(part == ".." for part in pat.parts):
            raise ToolError(f"knowledge source escapes workspace: {pattern!r}")
        try:
            candidates = sorted(base.glob(pattern))
        except (OSError, ValueError):
            raise ToolError(f"invalid knowledge source glob: {pattern!r}") from None
        for candidate in candidates:
            try:
                full = candidate.resolve()
            except OSError:
                continue
            if full in seen or not _confined(base, full) or not full.is_file():
                continue
            rel = full.relative_to(base)
            if RUNTIME_DIRNAME in rel.parts:
                continue
            seen.add(full)
            yield full


def _read_source_bytes(
    base: Path, full: Path, *, max_bytes: int
) -> tuple[bytes, os.stat_result]:
    """Bounded no-follow source read anchored beneath ``base``."""
    rel = full.relative_to(base)
    with confined_directory(base, rel.parent) as directory:
        return directory.read_regular_with_stat(rel.name, max_bytes=max_bytes)


def _grep(
    base: Path,
    sources: Sequence[str],
    query: str,
    *,
    max_hits: int = _MAX_HITS,
    max_line_chars: int = _GREP_LINE_CHARS,
) -> list[str]:
    hits: list[str] = []
    try:
        rx = re.compile(query)
    except re.error:
        rx = None  # fall back to substring matching

    for full in _iter_source_files(base, sources):
        try:
            payload, _ = _read_source_bytes(
                base, full, max_bytes=_MAX_READ_BYTES
            )
            text = payload.decode("utf-8")
        except (UnicodeDecodeError, OSError, PathEscapeError):
            continue
        rel = full.relative_to(base)
        for lineno, line in enumerate(text.splitlines(), start=1):
            matched = (rx.search(line) if rx else False) or (query in line)
            if matched:
                hits.append(f"{rel}:{lineno}: {line.strip()[:max_line_chars]}")
                if len(hits) >= max_hits:
                    hits.append(f"... (truncated at {max_hits} hits)")
                    return hits
    return hits


def _selector_list(paths: list[str] | None) -> tuple[str, ...] | None:
    if paths is None:
        return None
    selectors: list[str] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError("knowledge index paths must be non-empty strings")
        value = raw.strip().replace("\\", "/")
        path = Path(value)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            raise ToolError(f"knowledge index path escapes workspace: {raw!r}")
        selectors.append(value.removeprefix("./").rstrip("/"))
    return tuple(dict.fromkeys(selectors))


def _selected(path: str, selectors: tuple[str, ...] | None) -> bool:
    if selectors is None:
        return True
    for selector in selectors:
        if not selector:
            return True
        if fnmatch.fnmatchcase(path, selector):
            return True
        if path == selector or path.startswith(selector + "/"):
            return True
    return False


def _chunk_config(options: Mapping[str, Any]) -> tuple[int, int, str]:
    chunk_chars = _int_option(
        options,
        "chunk_chars",
        _DEFAULT_CHUNK_CHARS,
        minimum=32,
        maximum=32_000,
    )
    overlap = _int_option(
        options,
        "chunk_overlap_lines",
        _DEFAULT_CHUNK_OVERLAP_LINES,
        minimum=0,
        maximum=100,
    )
    max_source_bytes = _int_option(
        options,
        "max_source_bytes",
        _MAX_INDEX_SOURCE_BYTES,
        minimum=1,
        maximum=100 * 1024 * 1024,
    )
    fingerprint = (
        f"chars={chunk_chars};overlap_lines={overlap};"
        f"max_source_bytes={max_source_bytes}"
    )
    return chunk_chars, overlap, fingerprint


def _chunk_text(
    path: str,
    text: str,
    *,
    chunk_chars: int,
    overlap_lines: int,
    page: int | None = None,
    ordinal_start: int = 0,
) -> list[_ChunkDraft]:
    """Split text deterministically while retaining page/line citations.

    Very long physical lines are first divided into bounded pieces.  Repeated
    chunks receive a content-occurrence suffix, so their keys remain stable
    across re-indexes without collapsing legitimate duplicates.
    """
    lines = text.splitlines()
    if not lines and text:
        lines = [text]
    pieces: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, start=1):
        if not line:
            pieces.append((lineno, ""))
            continue
        for start in range(0, len(line), chunk_chars):
            pieces.append((lineno, line[start : start + chunk_chars]))

    drafts: list[_ChunkDraft] = []
    occurrences: dict[str, int] = {}
    cursor = 0
    while cursor < len(pieces):
        end = cursor
        total = 0
        while end < len(pieces):
            addition = len(pieces[end][1]) + (1 if end > cursor else 0)
            if end > cursor and total + addition > chunk_chars:
                break
            total += addition
            end += 1
        if end == cursor:  # defensive; a single piece is always <= chunk_chars
            end += 1
        group = pieces[cursor:end]
        chunk_text = "\n".join(piece for _, piece in group).strip()
        if chunk_text:
            text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            occurrence = occurrences.get(text_hash, 0)
            occurrences[text_hash] = occurrence + 1
            identity = f"{path}\0{page or 0}\0{text_hash}\0{occurrence}"
            drafts.append(_ChunkDraft(
                chunk_key=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                ordinal=ordinal_start + len(drafts),
                page=page,
                line_start=group[0][0],
                line_end=group[-1][0],
                text_hash=text_hash,
                text=chunk_text,
            ))
        if end >= len(pieces):
            break
        # Overlap by bounded pieces (normally one physical line each).  Always
        # advance at least one piece, including for a split long line.
        cursor = max(cursor + 1, end - overlap_lines)
    return drafts


def _load_snapshot(
    full: Path,
    base: Path,
    *,
    max_source_bytes: int,
    chunk_chars: int,
    overlap_lines: int,
) -> tuple[_SourceSnapshot | None, str | None]:
    rel = full.relative_to(base).as_posix()
    try:
        payload, info = _read_source_bytes(
            base, full, max_bytes=max_source_bytes
        )
    except PathEscapeError as exc:
        if "exceeds" in str(exc):
            return None, f"{rel} (over max_source_bytes)"
        return None, f"{rel} (unreadable)"
    except OSError:
        return None, f"{rel} (unreadable)"

    chunks: list[_ChunkDraft] = []
    if full.suffix.lower() == ".pdf":
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError:
            return None, f"{rel} (PDF extraction needs the pdf extra)"
        try:
            with fitz.open(stream=payload, filetype="pdf") as document:
                for page_index, pdf_page in enumerate(document, start=1):
                    page_chunks = _chunk_text(
                        rel,
                        pdf_page.get_text("text"),
                        chunk_chars=chunk_chars,
                        overlap_lines=overlap_lines,
                        page=page_index,
                        ordinal_start=len(chunks),
                    )
                    chunks.extend(page_chunks)
        except Exception:
            return None, f"{rel} (invalid PDF)"
    else:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return None, f"{rel} (not UTF-8 text)"
        chunks = _chunk_text(
            rel,
            text,
            chunk_chars=chunk_chars,
            overlap_lines=overlap_lines,
        )

    return _SourceSnapshot(
        path=rel,
        content_hash=hashlib.sha256(payload).hexdigest(),
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        chunks=tuple(chunks),
    ), None


def _index_path(base: Path, options: Mapping[str, Any]) -> Path:
    raw_value = options.get("index_path", _DEFAULT_INDEX_PATH)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ConfigError("tool_options.knowledge.index_path must be non-empty")
    raw = Path(raw_value)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise ToolError(f"knowledge index path escapes workspace: {raw_value!r}")
    candidate = base / raw
    if candidate.is_symlink():
        raise ToolError("knowledge index path must not be a symlink")
    resolved = candidate.resolve()
    if not _confined(base, resolved):
        raise ToolError(f"knowledge index path escapes workspace: {raw_value!r}")
    return resolved


def _open_index(
    base: Path, path: Path, *, create: bool
) -> sqlite3.Connection | None:
    """Load a confined SQLite file into an in-memory connection.

    SQLite's pathname API cannot accept an already-open no-follow descriptor.
    Deserializing a bounded, descriptor-read database avoids reopening an
    attacker-swappable workspace path, and indexing later publishes the whole
    database with the same atomic confined-write primitive as other runtime
    artifacts.
    """
    rel = path.relative_to(base)
    payload: bytes | None = None
    try:
        with confined_directory(base, rel.parent, create=create) as directory:
            if not directory.entry_exists(rel.name):
                if not create:
                    return None
            else:
                payload = directory.read_regular(
                    rel.name, max_bytes=_MAX_INDEX_BYTES
                )
    except FileNotFoundError:
        if not create:
            return None
        raise ToolError(f"cannot create knowledge index directory: {path.parent}") from None
    except (OSError, PathEscapeError) as exc:
        raise ToolError(f"cannot safely open knowledge index {path}: {exc}") from None

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(":memory:")
        if payload:
            connection.deserialize(payload)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, _SCHEMA_VERSION}:
            raise ConfigError(
                f"knowledge index schema {version} is unsupported; remove {path} "
                "and run knowledge action=index"
            )
        if version == 0:
            if not create and payload is not None:
                raise ConfigError(
                    f"knowledge index has no recognized schema; remove {path} "
                    "and rebuild it"
                )
            _create_schema(connection)
        return connection
    except ConfigError:
        if connection is not None:
            connection.close()
        raise
    except sqlite3.DatabaseError:
        if connection is not None:
            connection.close()
        raise ToolError(
            f"knowledge index is unreadable or corrupt: {path}; remove it and rebuild"
        ) from None


def _save_index(base: Path, path: Path, connection: sqlite3.Connection) -> None:
    """Atomically publish an in-memory index through a confined directory."""
    try:
        payload = connection.serialize()
        if len(payload) > _MAX_INDEX_BYTES:
            raise ToolError(
                f"knowledge index exceeds the {_MAX_INDEX_BYTES}-byte size limit"
            )
        rel = path.relative_to(base)
        with confined_directory(base, rel.parent, create=True) as directory:
            part = f".{rel.name}.{secrets.token_hex(8)}.part"
            try:
                with directory.open_exclusive(part) as handle:
                    handle.write(payload)
                directory.replace(part, rel.name)
            finally:
                directory.unlink(part, missing_ok=True)
    except ToolError:
        raise
    except (OSError, PathEscapeError, sqlite3.DatabaseError) as exc:
        raise ToolError(f"cannot safely write knowledge index {path}: {exc}") from None


def _create_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.executescript(f"""
            CREATE TABLE IF NOT EXISTS documents (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                chunk_config TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                chunk_key TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL REFERENCES documents(path) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                page INTEGER,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding_identity TEXT,
                embedding_dim INTEGER,
                embedding BLOB,
                UNIQUE(path, ordinal)
            );
            CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
            CREATE INDEX IF NOT EXISTS chunks_vector_idx
                ON chunks(embedding_identity) WHERE embedding IS NOT NULL;
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            PRAGMA user_version = {_SCHEMA_VERSION};
        """)
        connection.commit()
    except sqlite3.DatabaseError:
        connection.rollback()
        raise ConfigError(
            "SQLite FTS5 support is required for the knowledge index backend"
        ) from None


def _normalize_vector(values: Iterable[Any]) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ToolError("embedding provider returned a non-numeric vector") from None
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ToolError("embedding provider returned an empty or non-finite vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ToolError("embedding provider returned a zero-length vector")
    return [value / norm for value in vector]


def _pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(payload: bytes, dimensions: int) -> tuple[float, ...] | None:
    if dimensions <= 0 or len(payload) != dimensions * 4:
        return None
    try:
        return struct.unpack(f"<{dimensions}f", payload)
    except struct.error:
        return None


def _provider_identity(provider: Any) -> str:
    identity = getattr(provider, "identity", None)
    if not isinstance(identity, str) or not identity.strip():
        raise ConfigError("knowledge provider identity must be a non-empty string")
    return identity.strip()


def _embedder(ctx: ToolContext, options: Mapping[str, Any]) -> EmbeddingProvider:
    if not embedding_enabled(options):
        raise ConfigError(
            "embedding retrieval is disabled by default; set "
            "tool_options.knowledge.embedding.enabled: true to opt in"
        )
    provider = ctx.options.get(EMBEDDING_PROVIDER_KEY)
    if provider is None:
        provider = build_embedding_provider(options, environment=ctx.environment)
        ctx.options[EMBEDDING_PROVIDER_KEY] = provider
    if not isinstance(provider, EmbeddingProvider):
        raise ConfigError("injected knowledge embedder does not implement embed()")
    _provider_identity(provider)
    return provider


def _reranker(ctx: ToolContext, options: Mapping[str, Any]) -> RerankingProvider:
    provider = ctx.options.get(RERANKING_PROVIDER_KEY)
    if provider is None:
        provider = build_reranking_provider(options, environment=ctx.environment)
        ctx.options[RERANKING_PROVIDER_KEY] = provider
    if not isinstance(provider, RerankingProvider):
        raise ConfigError("injected knowledge reranker does not implement rerank()")
    _provider_identity(provider)
    return provider


async def _index_sources(
    ctx: ToolContext,
    options: Mapping[str, Any],
    sources: Sequence[str],
    selectors: tuple[str, ...] | None,
) -> str:
    base = ctx.workspace.resolve()
    path = _index_path(base, options)
    provider = _embedder(ctx, options)
    vector_identity = _provider_identity(provider)
    connection = _open_index(base, path, create=True)
    assert connection is not None
    chunk_chars, overlap_lines, config_fingerprint = _chunk_config(options)
    max_source_bytes = _int_option(
        options,
        "max_source_bytes",
        _MAX_INDEX_SOURCE_BYTES,
        minimum=1,
        maximum=100 * 1024 * 1024,
    )

    snapshots: list[_SourceSnapshot] = []
    skipped: list[str] = []
    try:
        all_files = list(_iter_source_files(base, sources))
        for full in all_files:
            rel = full.relative_to(base).as_posix()
            if not _selected(rel, selectors):
                continue
            snapshot, reason = _load_snapshot(
                full,
                base,
                max_source_bytes=max_source_bytes,
                chunk_chars=chunk_chars,
                overlap_lines=overlap_lines,
            )
            if snapshot is not None:
                snapshots.append(snapshot)
            elif reason is not None:
                skipped.append(reason)

        existing_rows = connection.execute(
            "SELECT path, content_hash, size, mtime_ns, chunk_config FROM documents"
        ).fetchall()
        existing = {str(row["path"]): row for row in existing_rows}
        desired = {snapshot.path for snapshot in snapshots}
        if selectors is None:
            remove_paths = set(existing) - desired
        else:
            remove_paths = {
                indexed_path
                for indexed_path in existing
                if _selected(indexed_path, selectors) and indexed_path not in desired
            }

        new_count = 0
        updated_count = 0
        unchanged_count = 0
        reused_vectors = 0
        targets: list[tuple[str, str | int, str, str]] = []
        # target tuple: ("draft"/"existing", chunk key/id, text hash, text)
        changed_snapshots: list[
            tuple[_SourceSnapshot, dict[str, tuple[str, int, bytes]]]
        ] = []

        reusable_by_hash: dict[str, tuple[str, int, bytes]] = {}
        for row in connection.execute(
            """
            SELECT text_hash, embedding_identity, embedding_dim, embedding
            FROM chunks
            WHERE embedding_identity = ? AND embedding IS NOT NULL
            """,
            (vector_identity,),
        ):
            if row["embedding_dim"] and row["embedding"] is not None:
                reusable_by_hash.setdefault(
                    str(row["text_hash"]),
                    (
                        str(row["embedding_identity"]),
                        int(row["embedding_dim"]),
                        bytes(row["embedding"]),
                    ),
                )

        for snapshot in snapshots:
            old = existing.get(snapshot.path)
            content_changed = (
                old is None
                or old["content_hash"] != snapshot.content_hash
                or old["chunk_config"] != config_fingerprint
            )
            if content_changed:
                if old is None:
                    new_count += 1
                else:
                    updated_count += 1
                reusable = reusable_by_hash
                changed_snapshots.append((snapshot, reusable))
                for draft in snapshot.chunks:
                    if draft.text_hash in reusable:
                        reused_vectors += 1
                    else:
                        targets.append(
                            ("draft", draft.chunk_key, draft.text_hash, draft.text)
                        )
            else:
                unchanged_count += 1
                for row in connection.execute(
                    """
                    SELECT id, text_hash, text FROM chunks
                    WHERE path = ? AND (
                        embedding IS NULL OR embedding_identity IS NULL
                        OR embedding_identity != ?
                    )
                    ORDER BY ordinal
                    """,
                    (snapshot.path, vector_identity),
                ):
                    targets.append((
                        "existing",
                        int(row["id"]),
                        str(row["text_hash"]),
                        str(row["text"]),
                    ))
                already = connection.execute(
                    """
                    SELECT COUNT(*) FROM chunks
                    WHERE path = ? AND embedding IS NOT NULL
                        AND embedding_identity = ?
                    """,
                    (snapshot.path, vector_identity),
                ).fetchone()[0]
                reused_vectors += int(already)

        unique_inputs = {target[2]: target[3] for target in targets}
        raw_vectors = (
            await provider.embed(list(unique_inputs.values()))
            if unique_inputs
            else []
        )
        if len(raw_vectors) != len(unique_inputs):
            raise ToolError("embedding provider returned the wrong vector count")
        encoded: dict[str, tuple[int, bytes]] = {}
        expected_dim: int | None = None
        for text_hash, raw_vector in zip(unique_inputs, raw_vectors):
            vector = _normalize_vector(raw_vector)
            if expected_dim is None:
                expected_dim = len(vector)
            elif len(vector) != expected_dim:
                raise ToolError(
                    "embedding provider returned inconsistent vector dimensions"
                )
            encoded[text_hash] = (len(vector), _pack_vector(vector))
        reused_vectors += len(targets) - len(unique_inputs)

        with connection:
            for remove_path in sorted(remove_paths):
                connection.execute("DELETE FROM documents WHERE path = ?", (remove_path,))

            for snapshot, reusable in changed_snapshots:
                connection.execute("DELETE FROM documents WHERE path = ?", (snapshot.path,))
                connection.execute(
                    """
                    INSERT INTO documents(path, content_hash, size, mtime_ns, chunk_config)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.path,
                        snapshot.content_hash,
                        snapshot.size,
                        snapshot.mtime_ns,
                        config_fingerprint,
                    ),
                )
                for draft in snapshot.chunks:
                    stored = reusable.get(draft.text_hash)
                    if stored is None:
                        dim, payload = encoded[draft.text_hash]
                        stored = (vector_identity, dim, payload)
                    connection.execute(
                        """
                        INSERT INTO chunks(
                            chunk_key, path, ordinal, page, line_start, line_end,
                            text_hash, text, embedding_identity, embedding_dim, embedding
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            draft.chunk_key,
                            snapshot.path,
                            draft.ordinal,
                            draft.page,
                            draft.line_start,
                            draft.line_end,
                            draft.text_hash,
                            draft.text,
                            stored[0],
                            stored[1],
                            stored[2],
                        ),
                    )

            for target_type, target_id, text_hash, _ in targets:
                if target_type != "existing":
                    continue
                dim, payload = encoded[text_hash]
                connection.execute(
                    """
                    UPDATE chunks SET embedding_identity = ?, embedding_dim = ?,
                        embedding = ? WHERE id = ?
                    """,
                    (vector_identity, dim, payload, target_id),
                )

        total_files = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        total_chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        _save_index(base, path, connection)
        report = (
            f"indexed files={total_files} "
            f"(new={new_count}, updated={updated_count}, "
            f"unchanged={unchanged_count}, deleted={len(remove_paths)}), "
            f"chunks={total_chunks}, embedded={len(unique_inputs)}, "
            f"reused_vectors={reused_vectors}"
        )
        if skipped:
            preview = "; ".join(skipped[:3])
            if len(skipped) > 3:
                preview += f"; +{len(skipped) - 3} more"
            report += f"; skipped={len(skipped)} [{preview}]"
        return report
    finally:
        connection.close()


def _stale_state(
    connection: sqlite3.Connection,
    base: Path,
    sources: Sequence[str],
    config_fingerprint: str,
    max_source_bytes: int,
) -> _StaleState:
    current: dict[str, tuple[Path, int, int]] = {}
    for full in _iter_source_files(base, sources):
        try:
            info = full.stat()
        except OSError:
            continue
        current[full.relative_to(base).as_posix()] = (
            full,
            info.st_size,
            info.st_mtime_ns,
        )
    indexed = {
        str(row["path"]): (
            str(row["content_hash"]),
            int(row["size"]),
            int(row["mtime_ns"]),
            str(row["chunk_config"]),
        )
        for row in connection.execute(
            "SELECT path, content_hash, size, mtime_ns, chunk_config FROM documents"
        )
    }
    changed: set[str] = set()
    for path in current.keys() & indexed.keys():
        full, size, mtime_ns = current[path]
        content_hash, old_size, old_mtime_ns, old_config = indexed[path]
        if (
            size != old_size
            or mtime_ns != old_mtime_ns
            or old_config != config_fingerprint
        ):
            changed.add(path)
            continue
        # Metadata can be preserved deliberately or by a coarse filesystem
        # clock. Verify the content hash so stale text is never cited merely
        # because size/mtime happen to match.
        try:
            payload, _ = _read_source_bytes(
                base, full, max_bytes=max_source_bytes
            )
            current_hash = hashlib.sha256(payload).hexdigest()
        except (OSError, PathEscapeError):
            changed.add(path)
        else:
            if current_hash != content_hash:
                changed.add(path)

    new: set[str] = set()
    for path in current.keys() - indexed.keys():
        full, size, _ = current[path]
        if size > max_source_bytes:
            continue
        try:
            payload, _ = _read_source_bytes(base, full, max_bytes=max_source_bytes)
        except (OSError, PathEscapeError):
            continue
        if full.suffix.lower() == ".pdf":
            try:
                import fitz  # type: ignore[import-not-found]

                with fitz.open(stream=payload, filetype="pdf"):
                    pass
            except Exception:
                # Mirror index-time PDF validation: a source that can never be
                # opened must not remain a permanently "new" stale entry.
                continue
        else:
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError:
                continue
        new.add(path)
    return _StaleState(
        changed=frozenset(changed),
        deleted=frozenset(indexed.keys() - current.keys()),
        new=frozenset(new),
    )


def _row_hit(row: sqlite3.Row, score: float, *, semantic: float | None = None) -> _SearchHit:
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
    return [
        _row_hit(row, 1.0 / rank)
        for rank, row in enumerate(rows, start=1)
    ]


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
            count = int(connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding_identity = ? "
                "AND embedding IS NOT NULL",
                (identity,),
            ).fetchone()[0])
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
    source_count = sum(1 for _ in _iter_source_files(base, sources))
    enabled = embedding_enabled(options)
    prefix = (
        f"backend={backend}, embedding={'true' if enabled else 'false'}, "
        f"source files={source_count}"
    )
    if backend == "grep":
        return prefix
    path = _index_path(base, options)
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
        )
        indexed = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        identity: str | None = None
        injected = ctx.options.get(EMBEDDING_PROVIDER_KEY)
        if injected is not None:
            identity = _provider_identity(injected)
        elif enabled:
            identity = embedding_identity(options)
        embedded = 0
        if identity is not None:
            embedded = int(connection.execute(
                """
                SELECT COUNT(*) FROM chunks
                WHERE embedding_identity = ? AND embedding IS NOT NULL
                """,
                (identity,),
            ).fetchone()[0])
        return (
            f"{prefix}, indexed files={indexed}, chunks={chunks}, "
            f"embedded chunks={embedded}, stale files={stale.count}"
        )
    finally:
        connection.close()


class KnowledgeArgs(BaseModel):
    action: Literal["query", "index", "status"] = Field(
        description="Operation: query (search), index (build/update), or status."
    )
    query: str | None = Field(
        default=None, description="Search query (required for query action)."
    )
    paths: list[str] | None = Field(
        default=None,
        description=(
            "Optional workspace-relative files, directories, or globs to update "
            "during an index action."
        ),
    )


@tool(description=(
    "Search the knowledge corpus of workspace files. Use `query` to find "
    "relevant, source-cited excerpts; `index` to incrementally build/update an "
    "opt-in semantic index; and `status` to inspect index freshness."
))
async def knowledge(args: KnowledgeArgs, ctx: ToolContext) -> str:
    options = _knowledge_options(ctx)
    backend = _backend(options)
    sources = _sources(options)
    base = ctx.workspace.resolve()

    if args.action == "status":
        return _status(ctx, options, sources, backend)

    if args.action == "index":
        if backend == "grep":
            return "grep backend needs no index; query directly"
        return await _index_sources(
            ctx, options, sources, _selector_list(args.paths)
        )

    if args.action == "query":
        if not args.query or not args.query.strip():
            raise ToolError("query action requires a query string")
        query = args.query.strip()
        if backend == "grep":
            hits = _grep(
                base,
                sources,
                query,
                max_hits=_int_option(
                    options, "max_hits", _MAX_HITS, minimum=1, maximum=100
                ),
                max_line_chars=_int_option(
                    options,
                    "max_line_chars",
                    _GREP_LINE_CHARS,
                    minimum=20,
                    maximum=10_000,
                ),
            )
            return "\n".join(hits) if hits else "(no matches)"
        return await _query_index(ctx, options, sources, query, backend)

    raise ToolError(f"unknown knowledge action: {args.action!r}")
