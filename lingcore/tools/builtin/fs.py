"""Built-in filesystem tools for the coding agent.

Every path is validated with the shared workspace guards: ordinary operations
resolve and reject escapes, while recursive search walks and reads through
no-follow directory descriptors. Together they confine filesystem access to
the configured workspace directory.
"""

from __future__ import annotations

import asyncio
import codecs
import fnmatch
import heapq
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

import regex
from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.media import attachment_from_path, detect_media, is_probably_binary
from lingcore.paths import (
    ConfinedDirectory,
    PathEscapeError,
    confined_directory,
    resolve_confined,
)
from lingcore.tool_options import SearchOptions, parse_search_options
from lingcore.tools import ToolContext, ToolOutput, tool
from lingcore.tools.builtin._offload import (
    RUNTIME_DIRNAME,
    offload_text,
)

_MAX_READ_BYTES = 256 * 1024
_READ_CHUNK_BYTES = 64 * 1024
# Default read window: keep results targetable and light so re-reads stay cheap
# and the conversation prefix grows slowly (better prompt-cache behavior).
_READ_MAX_LINES = 2_000
_READ_MAX_LINE_CHARS = 2_000
_LIST_MAX_ENTRIES = 200
_SEARCH_CHECK_BATCH = 128


def _resolve(ctx: ToolContext, path: str) -> Path:
    """Resolve ``path`` relative to the workspace, rejecting escapes.

    Delegates to the shared ``resolve_confined`` guard: ``Path.resolve()``
    collapses ``..`` and follows symlinks before the containment check, so
    neither traversal nor a symlink pointing outside the workspace can slip
    through.
    """
    try:
        return resolve_confined(ctx.workspace, path)
    except PathEscapeError as e:
        raise ToolError(str(e)) from None


def _validate_search_glob(pattern: str) -> None:
    """Reject glob patterns that can enumerate outside the workspace."""
    if not pattern.strip():
        raise ToolError("search glob must not be empty")
    p = Path(pattern)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ToolError(f"glob escapes workspace: {pattern!r}")


class ReadArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    offset: int = Field(
        default=1, ge=1, description="1-based line number to start reading from."
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of lines to return (capped by the tool default).",
    )


def _format_lines(
    text: str, *, offset: int, limit: int | None, max_lines: int, max_line_chars: int
) -> str:
    """Render file text as ``<lineno>\\t<line>`` over a bounded window.

    Line numbers are absolute (so a slice references correctly), over-long lines
    are clipped, and a stable marker announces any remainder — keeping a single
    read light and re-reads cheap.
    """
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return "(empty file)"
    start = offset - 1
    if start >= total:
        return f"(file has {total} lines; offset {offset} is past the end)"
    count = max_lines if limit is None else min(limit, max_lines)
    end = min(start + count, total)
    width = len(str(end))
    out: list[str] = []
    for i in range(start, end):
        line = lines[i]
        if len(line) > max_line_chars:
            line = line[:max_line_chars] + f"… (+{len(line) - max_line_chars} chars)"
        out.append(f"{i + 1:>{width}}\t{line}")
    body = "\n".join(out)
    if end < total:
        body += (
            f"\n… (showed lines {start + 1}–{end} of {total}; "
            "pass offset/limit for more)"
        )
    return body


class _BinaryFileError(Exception):
    """Internal signal: NUL bytes make the streamed payload non-text."""


# Exactly the boundaries ``str.splitlines()`` recognizes, so a streamed large
# file numbers its lines the same way the in-memory path does.
_LINE_BREAK = re.compile(r"\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")


class _StreamLines:
    """Split a binary stream into lines like ``str.splitlines()``, lazily.

    Memory stays bounded: lines longer than ``max_line_chars`` are clipped as
    they stream and their discarded suffix is only counted. NUL bytes in any
    consumed chunk raise :class:`_BinaryFileError` so binary files are still
    refused rather than rendered as mojibake.
    """

    def __init__(self, stream: IO[bytes], *, max_line_chars: int) -> None:
        self._stream = stream
        self._max_line_chars = max_line_chars
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._text = ""
        self._pos = 0
        self._eof = False

    def _fill(self) -> bool:
        """Append the next decoded chunk to the buffer; False once at EOF."""
        if self._eof:
            return False
        chunk = self._stream.read(_READ_CHUNK_BYTES)
        if b"\x00" in chunk:
            raise _BinaryFileError
        self._text = self._text[self._pos :] + self._decoder.decode(
            chunk, final=not chunk
        )
        self._pos = 0
        if not chunk:
            self._eof = True
            return False
        return True

    def next_line(self) -> str | None:
        """Return the next line (clipped), or ``None`` at end of file."""
        pending = ""
        dropped = 0
        started = False

        def absorb(piece: str) -> None:
            nonlocal pending, dropped
            if dropped:
                dropped += len(piece)
                return
            room = self._max_line_chars - len(pending)
            if len(piece) <= room:
                pending += piece
            else:
                pending += piece[:room]
                dropped = len(piece) - room

        def render() -> str:
            return f"{pending}… (+{dropped} chars)" if dropped else pending

        while True:
            match = _LINE_BREAK.search(self._text, self._pos)
            # A trailing "\r" may be the first half of a "\r\n" split across
            # chunks; decide only once the next chunk (or EOF) is known.
            split_crlf = (
                match is not None
                and match.group() == "\r"
                and match.end() == len(self._text)
                and not self._eof
            )
            if match is not None and not split_crlf:
                absorb(self._text[self._pos : match.start()])
                self._pos = match.end()
                return render()
            stop = len(self._text) - 1 if split_crlf else len(self._text)
            if stop > self._pos:
                started = True
                absorb(self._text[self._pos : stop])
                self._pos = stop
            if not self._fill() and self._pos >= len(self._text):
                return render() if started else None

    def has_more(self) -> bool:
        """Whether any content remains, reading at most one more chunk."""
        while self._pos >= len(self._text):
            if not self._fill() and self._pos >= len(self._text):
                return False
        return True


def _format_stream_lines(
    stream: IO[bytes],
    *,
    offset: int,
    limit: int | None,
    max_lines: int,
    max_line_chars: int,
) -> str:
    """Render a bounded line window without loading the whole file.

    Only the requested window is retained. Whether anything follows it is
    decided by peeking for a single character, never by reading the next
    (possibly huge) line, and the *total* line count is intentionally omitted
    for the same reason. The remainder marker still tells the model to page
    with ``offset``/``limit``.
    """
    start = offset - 1
    count = max_lines if limit is None else min(limit, max_lines)
    end = start + count
    lines = _StreamLines(stream, max_line_chars=max_line_chars)
    shown: list[tuple[int, str]] = []
    total = 0
    while total < end:
        line = lines.next_line()
        if line is None:
            break
        total += 1
        if total > start:
            shown.append((total, line))

    if not shown:
        if total == 0:
            return "(empty file)"
        return f"(file has {total} lines; offset {offset} is past the end)"
    width = len(str(shown[-1][0]))
    body = "\n".join(f"{lineno:>{width}}\t{line}" for lineno, line in shown)
    if total == end and lines.has_more():
        body += (
            f"\n… (showed lines {start + 1}–{shown[-1][0]}; more lines; "
            "pass offset/limit for more)"
        )
    return body


def _read_large_window(
    base: Path,
    rel: Path,
    *,
    offset: int,
    limit: int | None,
    max_lines: int,
    max_line_chars: int,
) -> str:
    with confined_directory(base, rel.parent) as directory:
        with directory.open_regular(rel.name, "rb") as stream:
            return _format_stream_lines(
                stream,
                offset=offset,
                limit=limit,
                max_lines=max_lines,
                max_line_chars=max_line_chars,
            )


@tool(
    description=(
        "Read a file from the workspace as line-numbered text "
        "(`<lineno>\\t<line>`), starting at `offset` (1-based) for up to `limit` "
        "lines — read large files in slices instead of all at once. For an image "
        "or PDF it attaches the file so the model can view it natively (degrading "
        "to extracted/described text when the model cannot). Use pdf2md instead "
        "to read a PDF as cheap markdown text."
    )
)
async def read_file(args: ReadArgs, ctx: ToolContext) -> str | ToolOutput:
    full = _resolve(ctx, args.path)
    base = ctx.workspace.resolve()
    if not full.is_file():
        raise ToolError(f"not a file: {args.path!r}")
    # An image/PDF (extension + magic bytes agree) is attached, not decoded —
    # attachment_from_path applies the larger media size caps (5/10 MB).
    with full.open("rb") as fh:
        head = fh.read(16)
    if detect_media(head, full):
        attachment = attachment_from_path(full)
        size = full.stat().st_size
        return ToolOutput(
            text=f"attached {attachment.name} ({attachment.media_type}, {size} bytes)",
            attachments=[attachment],
        )
    opts = ctx.options.get("read_file", {}) if ctx.options else {}
    max_lines = int(opts.get("max_lines", _READ_MAX_LINES))
    max_line_chars = int(opts.get("max_line_chars", _READ_MAX_LINE_CHARS))
    try:
        size = full.stat().st_size
    except OSError as exc:
        raise ToolError(f"cannot read file {args.path!r}: {exc}") from None
    if size <= _MAX_READ_BYTES:
        data = full.read_bytes()
        if len(data) > _MAX_READ_BYTES:
            raise ToolError(
                f"file too large ({len(data)} bytes; limit {_MAX_READ_BYTES})"
            )
        if is_probably_binary(data):
            raise ToolError(
                "binary file; not readable as text — inspect it with shell "
                "tools if available"
            )
        return _format_lines(
            data.decode("utf-8", errors="replace"),
            offset=args.offset,
            limit=args.limit,
            max_lines=max_lines,
            max_line_chars=max_line_chars,
        )

    # Large file: stream only the requested line window through a descriptor -
    # anchored no-follow open, so offloaded tool output can still be paged with
    # `offset`/`limit` instead of being rejected by the whole-file size guard.
    # Skipping to a deep offset is still a synchronous scan, so it runs in a
    # worker thread rather than blocking the event loop.
    try:
        return await asyncio.to_thread(
            _read_large_window,
            base,
            full.relative_to(base),
            offset=args.offset,
            limit=args.limit,
            max_lines=max_lines,
            max_line_chars=max_line_chars,
        )
    except _BinaryFileError:
        raise ToolError(
            "binary file; not readable as text — inspect it with shell tools "
            "if available"
        ) from None
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from None
    except OSError as exc:
        raise ToolError(f"cannot read file {args.path!r}: {exc}") from None


class WriteArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    content: str = Field(description="Full UTF-8 content to write.")


@tool(description="Create or overwrite a UTF-8 text file relative to the workspace.")
async def write_file(args: WriteArgs, ctx: ToolContext) -> str:
    base = ctx.workspace.resolve()
    full = _resolve(ctx, args.path)
    rel = full.relative_to(base)
    try:
        # Keep the validated parent directory open through the write. A
        # concurrent swap of any ancestor for a symlink is refused by the
        # descriptor anchor instead of redirecting the write outside the
        # workspace. The final component is opened O_NOFOLLOW.
        with confined_directory(base, rel.parent, create=True) as directory:
            with directory.open_regular(rel.name, "wb") as stream:
                stream.write(args.content.encode("utf-8"))
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from None
    except OSError as exc:
        raise ToolError(f"cannot write {args.path!r}: {exc}") from None
    return f"wrote {len(args.content)} chars to {args.path}"


class EditArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    old: str = Field(description="Exact text to replace; must occur exactly once.")
    new: str = Field(description="Replacement text.")


def _match_newlines(text: str, old: str, new: str) -> tuple[str, str]:
    """Adapt an LF-written edit to a CRLF/CR file, preserving its endings.

    ``read_file`` shows normalized lines, so the model writes ``old``/``new``
    with ``\\n``. When that misses verbatim but the file uses ``\\r\\n`` (or bare
    ``\\r``), the newlines are translated to the file's own convention.
    """
    if "\n" not in old or "\r" in old or old in text:
        return old, new
    newline = "\r\n" if "\r\n" in text else "\r" if "\r" in text else None
    if newline is None:
        return old, new
    if "\r" not in new:
        new = new.replace("\n", newline)
    return old.replace("\n", newline), new


@tool(
    description=(
        "Replace an exact, unique snippet in a file. `old` must occur exactly "
        "once; otherwise the edit is rejected so the model can disambiguate."
    )
)
async def edit_file(args: EditArgs, ctx: ToolContext) -> str:
    base = ctx.workspace.resolve()
    full = _resolve(ctx, args.path)
    rel = full.relative_to(base)
    try:
        # One descriptor-anchored read/write keeps the validated parent in
        # place for the entire edit; only then is the replacement serialized.
        with confined_directory(base, rel.parent) as directory:
            with directory.open_regular(rel.name, "r+b") as stream:
                text = stream.read().decode("utf-8")
                old, new = _match_newlines(text, args.old, args.new)
                count = text.count(old)
                if count == 0:
                    raise ToolError(f"`old` text not found in {args.path!r}")
                if count > 1:
                    raise ToolError(
                        f"`old` text occurs {count} times in {args.path!r}; "
                        "make it unique to target a single location"
                    )
                stream.seek(0)
                stream.write(text.replace(old, new).encode("utf-8"))
                stream.truncate()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        raise ToolError(f"not a file: {args.path!r}") from None
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from None
    except OSError as exc:
        raise ToolError(f"cannot edit {args.path!r}: {exc}") from None
    return f"edited {args.path}"


class ListArgs(BaseModel):
    path: str = Field(default=".", description="Directory relative to workspace.")


@tool(description="List entries of a directory relative to the workspace root.")
async def list_dir(args: ListArgs, ctx: ToolContext) -> str:
    full = _resolve(ctx, args.path)
    if not full.is_dir():
        raise ToolError(f"not a directory: {args.path!r}")
    opts = ctx.options.get("list_dir", {}) if ctx.options else {}
    max_entries = int(opts.get("max_entries", _LIST_MAX_ENTRIES))
    entries = sorted(f"{p.name}/" if p.is_dir() else p.name for p in full.iterdir())
    if not entries:
        return "(empty)"
    shown = entries[:max_entries]
    out = "\n".join(shown)
    if len(entries) > len(shown):
        out += f"\n… ({len(entries) - len(shown)} more entries)"
    return out


class SearchArgs(BaseModel):
    query: str | None = Field(
        default=None,
        description="Content pattern; required when mode is `content`.",
    )
    mode: Literal["content", "files"] = Field(
        default="content",
        description=(
            "`content` returns path:line matches; `files` returns matching paths. "
            "In files mode, omit query to find files by name only."
        ),
    )
    path: str = Field(
        default=".", description="Workspace-relative directory that scopes the walk."
    )
    glob: str = Field(
        default="**/*",
        description=(
            "Filename filter. Without `/`, matches the basename at any depth "
            "(`*.py`); with `/`, matches paths relative to `path` using fnmatch "
            "semantics. A leading `**/` also matches files at the scope root."
        ),
    )
    regex: bool = Field(
        default=False,
        description="Use Python regex syntax with timed regex VERSION0 matching.",
    )
    ignore_case: bool = Field(
        default=False, description="Match content without case sensitivity."
    )
    context: int = Field(
        default=0,
        ge=0,
        description="Context lines around content matches (configuration-capped).",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Per-call hit cap, bounded by the configured max_hits.",
    )


@dataclass(slots=True)
class _SearchState:
    deadline: float
    matches: int = 0
    matched_files: int = 0
    scanned_files: int = 0
    scanned_dirs: int = 0
    skipped_too_large: int = 0
    skipped_binary: int = 0
    skipped_non_utf8: int = 0
    skipped_non_regular: int = 0
    skipped_symlink: int = 0
    skipped_unreadable: int = 0
    pruned: set[str] = field(default_factory=set)
    hit_match_cap: bool = False
    hit_file_cap: bool = False
    hit_dir_entry_cap: bool = False
    hit_depth_cap: bool = False
    stopped_time: bool = False
    content_results: list[tuple[str, str]] = field(default_factory=list)
    file_results: list[str] = field(default_factory=list)


def _search_options(ctx: ToolContext) -> SearchOptions:
    raw = ctx.options.get("search", {}) if ctx.options else {}
    return parse_search_options(raw)


def _search_matcher(
    args: SearchArgs, state: _SearchState
) -> Callable[[str], bool] | None:
    query = args.query
    if query is None:
        if args.mode == "content":
            raise ToolError("content search requires a query")
        return None
    if not query:
        raise ToolError("search query must not be empty")
    if args.regex:
        try:
            # Keep Python regex syntax validation. The timed engine uses its
            # VERSION0 semantics (Unicode case folding can differ from re).
            re.compile(query, re.IGNORECASE if args.ignore_case else 0)
            pattern = regex.compile(
                query, regex.VERSION0 | (regex.IGNORECASE if args.ignore_case else 0)
            )
        except (re.error, regex.error) as exc:
            raise ToolError(f"invalid search regex: {exc}") from None

        def match(line: str) -> bool:
            remaining = state.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            # A deadline between lines cannot interrupt catastrophic
            # backtracking within a line. Bound each match by the remaining
            # scan budget and release the GIL so other turns can keep running.
            return pattern.search(line, timeout=remaining, concurrent=True) is not None

        return match
    if args.ignore_case:
        needle = query.casefold()
        return lambda line: needle in line.casefold()
    return lambda line: query in line


def _glob_matches(pattern: str, relative_path: str) -> bool:
    if "/" not in pattern:
        return fnmatch.fnmatchcase(relative_path.rsplit("/", 1)[-1], pattern)
    if fnmatch.fnmatchcase(relative_path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatchcase(
        relative_path, pattern.removeprefix("**/")
    )


def _deadline_reached(state: _SearchState) -> bool:
    if state.stopped_time:
        return True
    if time.monotonic() >= state.deadline:
        state.stopped_time = True
        return True
    return False


def _bounded_entries(
    directory: ConfinedDirectory,
    options: SearchOptions,
    state: _SearchState,
) -> list[tuple[str, bool, bool, bool]]:
    count = 0

    def entries() -> Iterator[tuple[str, bool, bool, bool]]:
        nonlocal count
        for entry in directory.iter_entries():
            count += 1
            if count % _SEARCH_CHECK_BATCH == 0 and _deadline_reached(state):
                break
            yield entry

    try:
        selected = heapq.nsmallest(
            options.max_dir_entries, entries(), key=lambda entry: entry[0]
        )
    except PathEscapeError:
        state.skipped_unreadable += 1
        return []
    if count > options.max_dir_entries:
        state.hit_dir_entry_cap = True
    if state.stopped_time:
        return []
    return selected


def _render_content_file(
    path: str,
    lines: list[str],
    line_numbers: list[int],
    *,
    context: int,
    max_line_chars: int,
    omitted: int,
) -> str:
    def match_text(line: str) -> str:
        return line.strip()[:max_line_chars]

    def context_text(line: str) -> str:
        return line[:max_line_chars]

    if context == 0:
        rendered = [
            f"{path}:{lineno}: {match_text(lines[lineno - 1])}"
            for lineno in line_numbers
        ]
    else:
        intervals: list[tuple[int, int]] = []
        for lineno in line_numbers:
            start = max(1, lineno - context)
            end = min(len(lines), lineno + context)
            if intervals and start <= intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
            else:
                intervals.append((start, end))
        matches = set(line_numbers)
        groups: list[str] = []
        for start, end in intervals:
            group: list[str] = []
            for lineno in range(start, end + 1):
                separator = ":" if lineno in matches else "-"
                text = (
                    match_text(lines[lineno - 1])
                    if lineno in matches
                    else context_text(lines[lineno - 1])
                )
                group.append(f"{path}{separator}{lineno}{separator} {text}")
            groups.append("\n".join(group))
        rendered = ["\n--\n".join(groups)]
    if omitted:
        rendered.append(f"{path}: (+{omitted} more matches in this file)")
    return "\n".join(rendered)


def _record_content_matches(
    path: str,
    lines: list[str],
    matcher: Callable[[str], bool],
    *,
    context: int,
    hit_cap: int,
    options: SearchOptions,
    state: _SearchState,
) -> bool:
    line_numbers: list[int] = []
    matches_seen = 0
    stop = False
    completed = True
    for lineno, line in enumerate(lines, start=1):
        if (lineno - 1) % _SEARCH_CHECK_BATCH == 0 and _deadline_reached(state):
            stop = True
            completed = False
            break
        try:
            matched = matcher(line)
        except TimeoutError:
            state.stopped_time = True
            stop = True
            completed = False
            break
        if not matched:
            continue
        matches_seen += 1
        if len(line_numbers) >= options.max_hits_per_file:
            continue
        if state.matches >= hit_cap:
            state.hit_match_cap = True
            stop = True
            completed = False
            break
        line_numbers.append(lineno)
        state.matches += 1
    if line_numbers:
        state.matched_files += 1
        state.content_results.append(
            (
                path,
                _render_content_file(
                    path,
                    lines,
                    line_numbers,
                    context=context,
                    max_line_chars=options.max_line_chars,
                    omitted=matches_seen - len(line_numbers) if completed else 0,
                ),
            )
        )
    return stop


def _record_file_content_match(
    path: str,
    lines: list[str],
    matcher: Callable[[str], bool],
    *,
    hit_cap: int,
    options: SearchOptions,
    state: _SearchState,
) -> bool:
    for lineno, line in enumerate(lines, start=1):
        if (lineno - 1) % _SEARCH_CHECK_BATCH == 0 and _deadline_reached(state):
            return True
        try:
            matched = matcher(line)
        except TimeoutError:
            state.stopped_time = True
            return True
        if not matched:
            continue
        if state.matches >= hit_cap:
            state.hit_match_cap = True
            return True
        state.matches += 1
        state.matched_files += 1
        state.file_results.append(path)
        return False
    return False


def _count_read_skip(exc: PathEscapeError, state: _SearchState) -> None:
    message = str(exc)
    if "exceeds" in message or "changed beyond" in message:
        state.skipped_too_large += 1
    elif "not a regular file" in message:
        state.skipped_non_regular += 1
    else:
        state.skipped_unreadable += 1


def _scan_file(
    directory: ConfinedDirectory,
    name: str,
    path: str,
    args: SearchArgs,
    matcher: Callable[[str], bool] | None,
    *,
    context: int,
    hit_cap: int,
    options: SearchOptions,
    state: _SearchState,
) -> bool:
    if state.scanned_files >= options.max_files_scanned:
        state.hit_file_cap = True
        return True
    state.scanned_files += 1
    if args.mode == "files" and matcher is None:
        if state.matches >= hit_cap:
            state.hit_match_cap = True
            return True
        state.matches += 1
        state.matched_files += 1
        state.file_results.append(path)
        return False
    try:
        payload, _ = directory.read_regular_with_stat(
            name, max_bytes=options.max_file_bytes
        )
    except PathEscapeError as exc:
        _count_read_skip(exc, state)
        return False
    if is_probably_binary(payload):
        state.skipped_binary += 1
        return False
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        state.skipped_non_utf8 += 1
        return False
    lines = text.splitlines()
    if matcher is None:  # only files mode without a query reaches this branch
        return False
    if args.mode == "files":
        return _record_file_content_match(
            path,
            lines,
            matcher,
            hit_cap=hit_cap,
            options=options,
            state=state,
        )
    return _record_content_matches(
        path,
        lines,
        matcher,
        context=context,
        hit_cap=hit_cap,
        options=options,
        state=state,
    )


def _walk_search(
    directory: ConfinedDirectory,
    scope_parts: tuple[str, ...],
    workspace_parts: tuple[str, ...],
    depth: int,
    args: SearchArgs,
    matcher: Callable[[str], bool] | None,
    *,
    context: int,
    hit_cap: int,
    options: SearchOptions,
    state: _SearchState,
) -> bool:
    if _deadline_reached(state):
        return True
    state.scanned_dirs += 1
    for name, is_dir, is_file, is_symlink in _bounded_entries(
        directory, options, state
    ):
        if _deadline_reached(state):
            return True
        relative_scope = "/".join(scope_parts + (name,))
        relative_workspace = "/".join(workspace_parts + (name,))
        if is_symlink:
            if _glob_matches(args.glob, relative_scope):
                state.skipped_symlink += 1
            continue
        if is_dir:
            if name == RUNTIME_DIRNAME or name in options.exclude_dirs:
                # Offloading may create this internal directory after the walk.
                # Keep that side effect out of coverage so repeats stay stable.
                if name != RUNTIME_DIRNAME:
                    state.pruned.add(name)
                continue
            if depth >= options.max_depth:
                state.hit_depth_cap = True
                continue
            try:
                with directory.subdirectory(name) as child:
                    if _walk_search(
                        child,
                        scope_parts + (name,),
                        workspace_parts + (name,),
                        depth + 1,
                        args,
                        matcher,
                        context=context,
                        hit_cap=hit_cap,
                        options=options,
                        state=state,
                    ):
                        return True
            except PathEscapeError:
                state.skipped_unreadable += 1
            continue
        if not _glob_matches(args.glob, relative_scope):
            continue
        if not is_file:
            state.skipped_non_regular += 1
            continue
        if _scan_file(
            directory,
            name,
            relative_workspace,
            args,
            matcher,
            context=context,
            hit_cap=hit_cap,
            options=options,
            state=state,
        ):
            return True
    return state.stopped_time


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or singular + "s")


def _coverage_footer(state: _SearchState, options: SearchOptions, hit_cap: int) -> str:
    parts = [
        f"{state.matches} {_plural(state.matches, 'match', 'matches')} in "
        f"{state.matched_files} {_plural(state.matched_files, 'file')}",
        f"scanned {state.scanned_files} {_plural(state.scanned_files, 'file')}, "
        f"{state.scanned_dirs} {_plural(state.scanned_dirs, 'dir')}",
    ]
    skipped = [
        (state.skipped_too_large, "too large", "too large"),
        (state.skipped_binary, "binary", "binaries"),
        (state.skipped_non_utf8, "non-UTF-8", "non-UTF-8"),
        (state.skipped_non_regular, "non-regular", "non-regular"),
        (state.skipped_symlink, "symlink", "symlinks"),
        (state.skipped_unreadable, "unreadable", "unreadable"),
    ]
    skip_notes = [
        f"{count} {_plural(count, singular, plural)}"
        for count, singular, plural in skipped
        if count
    ]
    if skip_notes:
        parts.append("skipped " + ", ".join(skip_notes))
    if state.pruned:
        parts.append("pruned " + ", ".join(sorted(state.pruned)))
    if state.hit_match_cap:
        parts.append(f"hit the {hit_cap}-match cap")
    if state.stopped_time:
        parts.append(f"stopped after the {options.time_budget_ms} ms budget")
    if state.hit_file_cap:
        parts.append(f"hit the {options.max_files_scanned}-file scan cap")
    if state.hit_dir_entry_cap:
        parts.append(f"hit a {options.max_dir_entries}-entry directory cap")
    if state.hit_depth_cap:
        parts.append(f"hit the {options.max_depth}-level depth cap")
    return "(" + "; ".join(parts) + ")"


def _search_sync(args: SearchArgs, ctx: ToolContext, options: SearchOptions) -> str:
    state = _SearchState(deadline=time.monotonic() + options.time_budget_ms / 1_000)
    matcher = _search_matcher(args, state)
    base = ctx.workspace.resolve()
    try:
        scope_path = resolve_confined(base, args.path)
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from None
    if not scope_path.is_dir():
        raise ToolError(f"not a directory: {args.path!r}")
    hit_cap = min(args.limit or options.max_hits, options.max_hits)
    context = min(args.context, options.max_context_lines)
    try:
        with confined_directory(base, args.path) as scope:
            workspace_parts = tuple(scope.path.relative_to(base).parts)
            _walk_search(
                scope,
                (),
                workspace_parts,
                0,
                args,
                matcher,
                context=context,
                hit_cap=hit_cap,
                options=options,
                state=state,
            )
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from None
    except OSError:
        raise ToolError(f"not a directory: {args.path!r}") from None

    if args.mode == "content":
        separator = "\n--\n" if context else "\n"
        body = separator.join(rendered for _, rendered in sorted(state.content_results))
    else:
        body = "\n".join(sorted(state.file_results))
    if not body:
        body = "(no matches)"
    footer = _coverage_footer(state, options, hit_cap)
    result = body + "\n" + footer
    if options.offload_over_chars > 0 and len(result) <= options.offload_over_chars:
        return result
    # Stage/truncate only the body: coverage is the most important information
    # when output is large or partial, so it must remain visible at the end.
    body_threshold = (
        0
        if options.offload_over_chars <= 0
        else min(options.offload_over_chars, max(1, len(body) - 1))
    )
    rendered_body = offload_text(
        ctx,
        source="search",
        text=body,
        threshold=body_threshold,
    )
    return rendered_body + "\n" + footer


@tool(
    description=(
        "Search the workspace with a bounded, recursive, pruned scan. Use `path` "
        "to scope it; basename or relative-path `glob` filters filenames; literal "
        "content matching is the default, with optional timed `regex`, "
        "`ignore_case`, and context lines. Set mode=`files` to return paths whose "
        "content matches, or omit query in that mode to find files by name. Every "
        "result ends with a coverage footer reporting scanned/skipped/pruned work "
        "and any cap or time-budget stop. Symlinks are reported and never followed."
    )
)
async def search(args: SearchArgs, ctx: ToolContext) -> str:
    _validate_search_glob(args.glob)
    options = _search_options(ctx)
    return await asyncio.to_thread(_search_sync, args, ctx, options)
