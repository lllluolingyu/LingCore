"""Confined knowledge sources, chunk construction, and freshness checks."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from lingcore.errors import ToolError
from lingcore.paths import (
    PathEscapeError,
    confined_directory,
)
from lingcore.tools.builtin._knowledge.common import (
    _DEFAULT_CHUNK_CHARS,
    _DEFAULT_CHUNK_OVERLAP_LINES,
    _MAX_INDEX_SOURCE_BYTES,
    _ChunkDraft,
    _confined,
    _int_option,
    _SourceSnapshot,
    _StaleState,
)
from lingcore.tools.builtin._offload import RUNTIME_DIRNAME


def _iter_source_files(
    base: Path, sources: Sequence[str], *, index_path: Path | None = None
) -> Iterator[Path]:
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
            if RUNTIME_DIRNAME in rel.parts or (
                index_path is not None
                and full
                in (index_path, index_path.with_name(index_path.name + ".lock"))
            ):
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
            drafts.append(
                _ChunkDraft(
                    chunk_key=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    ordinal=ordinal_start + len(drafts),
                    page=page,
                    line_start=group[0][0],
                    line_end=group[-1][0],
                    text_hash=text_hash,
                    text=chunk_text,
                )
            )
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
        payload, info = _read_source_bytes(base, full, max_bytes=max_source_bytes)
    except PathEscapeError as exc:
        if "exceeds" in str(exc):
            return None, f"{rel} (over max_source_bytes)"
        return None, f"{rel} (unreadable)"
    except OSError:
        return None, f"{rel} (unreadable)"

    chunks: list[_ChunkDraft] = []
    if full.suffix.lower() == ".pdf":
        try:
            import fitz
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


def _stale_state(
    connection: sqlite3.Connection,
    base: Path,
    sources: Sequence[str],
    config_fingerprint: str,
    max_source_bytes: int,
    index_path: Path,
) -> _StaleState:
    current: dict[str, tuple[Path, int, int]] = {}
    for full in _iter_source_files(base, sources, index_path=index_path):
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
            payload, _ = _read_source_bytes(base, full, max_bytes=max_source_bytes)
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
                import fitz

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
