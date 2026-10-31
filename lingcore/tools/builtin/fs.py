"""Built-in filesystem tools for the coding agent.

Every path is validated with the shared workspace guards: ordinary operations
resolve and reject escapes, while recursive search walks and reads through
no-follow directory descriptors. Together they confine filesystem access to
the configured workspace directory.
"""

from __future__ import annotations

import asyncio
import codecs
import re
from pathlib import Path
from typing import IO

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.media import attachment_from_path, detect_media, is_probably_binary
from lingcore.paths import PathEscapeError, confined_directory, resolve_confined
from lingcore.tools import ToolContext, ToolOutput, tool
from lingcore.tools.builtin._fs_search import (
    SearchArgs,
    _search_options,
    _search_sync,
    _validate_search_glob,
)

__all__ = ["SearchArgs"]

_MAX_READ_BYTES = 256 * 1024
_READ_CHUNK_BYTES = 64 * 1024
# Default read window: keep results targetable and light so re-reads stay cheap
# and the conversation prefix grows slowly (better prompt-cache behavior).
_READ_MAX_LINES = 2_000
_READ_MAX_LINE_CHARS = 2_000
_LIST_MAX_ENTRIES = 200


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
    ),
    workspace_path_args=("path",),
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


@tool(
    description="Create or overwrite a UTF-8 text file relative to the workspace.",
    workspace_path_args=("path",),
)
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


@tool(
    description="List entries of a directory relative to the workspace root.",
    workspace_path_args=("path",),
)
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
