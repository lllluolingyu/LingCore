"""Incremental indexing with cross-process locking and vector reuse."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping, Sequence

from lingcore.errors import ToolError
from lingcore.paths import (
    ConfinedDirectory,
    ConfinedLockFile,
    PathEscapeError,
    confined_directory,
)
from lingcore.tools import ToolContext
from lingcore.tools.builtin._knowledge.common import (
    _MAX_INDEX_SOURCE_BYTES,
    _embedder,
    _int_option,
    _normalize_vector,
    _pack_vector,
    _provider_identity,
    _SourceSnapshot,
)
from lingcore.tools.builtin._knowledge.sources import (
    _chunk_config,
    _iter_source_files,
    _load_snapshot,
    _selected,
)
from lingcore.tools.builtin._knowledge.storage import (
    _index_path,
    _open_index,
    _save_index,
)


async def _index_sources(
    ctx: ToolContext,
    options: Mapping[str, Any],
    sources: Sequence[str],
    selectors: tuple[str, ...] | None,
) -> str:
    base = ctx.workspace.resolve()
    path = _index_path(base, options)
    try:
        with confined_directory(
            base, path.relative_to(base).parent, create=True
        ) as directory:
            with directory.open_lock_file(path.name + ".lock") as lock:
                while not lock.try_acquire():
                    await asyncio.sleep(0.05)
                return await _update_index_sources(
                    ctx, options, sources, selectors, base, path, directory, lock
                )
    except (OSError, PathEscapeError) as exc:
        raise ToolError(f"cannot safely lock knowledge index {path}: {exc}") from None


async def _update_index_sources(
    ctx: ToolContext,
    options: Mapping[str, Any],
    sources: Sequence[str],
    selectors: tuple[str, ...] | None,
    base: Path,
    path: Path,
    directory: ConfinedDirectory,
    lock: ConfinedLockFile,
) -> str:
    provider = _embedder(ctx, options)
    vector_identity = _provider_identity(provider)
    chunk_chars, overlap_lines, config_fingerprint = _chunk_config(options)
    max_source_bytes = _int_option(
        options,
        "max_source_bytes",
        _MAX_INDEX_SOURCE_BYTES,
        minimum=1,
        maximum=100 * 1024 * 1024,
    )
    connection = _open_index(base, path, create=True, directory=directory)
    assert connection is not None

    snapshots: list[_SourceSnapshot] = []
    skipped: list[str] = []
    try:
        all_files = list(_iter_source_files(base, sources, index_path=path))
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
        unchanged_snapshots: list[_SourceSnapshot] = []
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
                # Same content/chunk config, but the source may have been
                # touched (or restored with a preserved hash). Refresh the
                # stored metadata below so query-time staleness checks pass
                # after a successful reindex.
                unchanged_snapshots.append(snapshot)
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
                    targets.append(
                        (
                            "existing",
                            int(row["id"]),
                            str(row["text_hash"]),
                            str(row["text"]),
                        )
                    )
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
            await provider.embed(list(unique_inputs.values())) if unique_inputs else []
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
                connection.execute(
                    "DELETE FROM documents WHERE path = ?", (remove_path,)
                )

            for snapshot in unchanged_snapshots:
                connection.execute(
                    """
                    UPDATE documents
                    SET content_hash = ?, size = ?, mtime_ns = ?, chunk_config = ?
                    WHERE path = ?
                    """,
                    (
                        snapshot.content_hash,
                        snapshot.size,
                        snapshot.mtime_ns,
                        config_fingerprint,
                        snapshot.path,
                    ),
                )

            for snapshot, reusable in changed_snapshots:
                connection.execute(
                    "DELETE FROM documents WHERE path = ?", (snapshot.path,)
                )
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

        total_files = int(
            connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        total_chunks = int(
            connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
        lock.ensure_anchored()
        _save_index(base, path, connection, directory=directory)
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
