"""Built-in filesystem tools for the coding agent.

Every path is resolved through ``_resolve``, which rejects any path that
escapes the workspace (``..`` traversal, absolute paths, symlink hops). This
is the single most important safety guard in the framework: it confines all
file reads and writes to the configured workspace directory.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.media import attachment_from_path, detect_media, is_probably_binary
from lingcore.tools import ToolContext, ToolOutput, tool
from lingcore.tools.builtin._offload import RUNTIME_DIRNAME

_MAX_READ_BYTES = 256 * 1024
_MAX_SEARCH_HITS = 100
# Default read window: keep results targetable and light so re-reads stay cheap
# and the conversation prefix grows slowly (better prompt-cache behavior).
_READ_MAX_LINES = 2_000
_READ_MAX_LINE_CHARS = 2_000
_LIST_MAX_ENTRIES = 200
_SEARCH_LINE_CHARS = 200


def _resolve(ctx: ToolContext, path: str) -> Path:
    """Resolve ``path`` relative to the workspace, rejecting escapes.

    ``Path.resolve()`` collapses ``..`` and follows symlinks before the
    containment check, so neither traversal nor a symlink pointing outside
    the workspace can slip through.
    """
    base = ctx.workspace.resolve()
    full = (base / path).resolve()
    if full != base and not full.is_relative_to(base):
        raise ToolError(f"path escapes workspace: {path!r}")
    return full


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
    data = full.read_bytes()
    if len(data) > _MAX_READ_BYTES:
        raise ToolError(
            f"file too large ({len(data)} bytes; limit {_MAX_READ_BYTES})"
        )
    if is_probably_binary(data):
        raise ToolError(
            "binary file; not readable as text — inspect it with shell tools if available"
        )
    opts = ctx.options.get("read_file", {}) if ctx.options else {}
    return _format_lines(
        data.decode("utf-8", errors="replace"),
        offset=args.offset,
        limit=args.limit,
        max_lines=int(opts.get("max_lines", _READ_MAX_LINES)),
        max_line_chars=int(opts.get("max_line_chars", _READ_MAX_LINE_CHARS)),
    )


class WriteArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    content: str = Field(description="Full UTF-8 content to write.")


@tool(description="Create or overwrite a UTF-8 text file relative to the workspace.")
async def write_file(args: WriteArgs, ctx: ToolContext) -> str:
    full = _resolve(ctx, args.path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(args.content, encoding="utf-8")
    return f"wrote {len(args.content)} chars to {args.path}"


class EditArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    old: str = Field(description="Exact text to replace; must occur exactly once.")
    new: str = Field(description="Replacement text.")


@tool(
    description=(
        "Replace an exact, unique snippet in a file. `old` must occur exactly "
        "once; otherwise the edit is rejected so the model can disambiguate."
    )
)
async def edit_file(args: EditArgs, ctx: ToolContext) -> str:
    full = _resolve(ctx, args.path)
    if not full.is_file():
        raise ToolError(f"not a file: {args.path!r}")
    text = full.read_text("utf-8")
    count = text.count(args.old)
    if count == 0:
        raise ToolError(f"`old` text not found in {args.path!r}")
    if count > 1:
        raise ToolError(
            f"`old` text occurs {count} times in {args.path!r}; "
            "make it unique to target a single location"
        )
    full.write_text(text.replace(args.old, args.new), encoding="utf-8")
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
    entries = sorted(
        f"{p.name}/" if p.is_dir() else p.name for p in full.iterdir()
    )
    if not entries:
        return "(empty)"
    shown = entries[:max_entries]
    out = "\n".join(shown)
    if len(entries) > len(shown):
        out += f"\n… ({len(entries) - len(shown)} more entries)"
    return out


class SearchArgs(BaseModel):
    query: str = Field(description="Substring to search for.")
    glob: str = Field(default="**/*", description="Glob of files to scan.")


@tool(description="Search workspace files for a substring; returns path:line matches.")
async def search(args: SearchArgs, ctx: ToolContext) -> str:
    base = ctx.workspace.resolve()
    _validate_search_glob(args.glob)
    opts = ctx.options.get("search", {}) if ctx.options else {}
    max_hits = int(opts.get("max_hits", _MAX_SEARCH_HITS))
    max_line_chars = int(opts.get("max_line_chars", _SEARCH_LINE_CHARS))
    hits: list[str] = []
    try:
        # Sorted iteration ⇒ byte-stable results across calls (Path.glob order
        # is filesystem-dependent otherwise).
        for p in sorted(base.glob(args.glob)):
            try:
                full = p.resolve()
            except OSError:
                continue
            if full != base and not full.is_relative_to(base):
                continue
            if not full.is_file():
                continue
            rel = p.relative_to(base)
            if RUNTIME_DIRNAME in rel.parts:
                continue  # skip LingCore's own runtime artifacts (offloaded output)
            try:
                if full.stat().st_size > _MAX_READ_BYTES:
                    continue
                text = full.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if args.query in line:
                    hits.append(f"{rel}:{lineno}: {line.strip()[:max_line_chars]}")
                    if len(hits) >= max_hits:
                        hits.append(f"... (truncated at {max_hits} hits)")
                        return "\n".join(hits)
    except (NotImplementedError, ValueError) as e:
        raise ToolError(f"invalid search glob {args.glob!r}: {e}") from None
    return "\n".join(hits) if hits else "(no matches)"
