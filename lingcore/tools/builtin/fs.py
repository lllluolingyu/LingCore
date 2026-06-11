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

_MAX_READ_BYTES = 256 * 1024
_MAX_SEARCH_HITS = 100


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


@tool(description="Read a UTF-8 text file relative to the workspace root.")
async def read_file(args: ReadArgs, ctx: ToolContext) -> str:
    full = _resolve(ctx, args.path)
    if not full.is_file():
        raise ToolError(f"not a file: {args.path!r}")
    data = full.read_bytes()
    if len(data) > _MAX_READ_BYTES:
        raise ToolError(
            f"file too large ({len(data)} bytes; limit {_MAX_READ_BYTES})"
        )
    if detect_media(data, full):
        raise ToolError(
            "media file is not readable as UTF-8 text; use read_media if available"
        )
    if is_probably_binary(data):
        raise ToolError("binary file; not readable as text")
    return data.decode("utf-8", errors="replace")


@tool(
    description="Attach an image or PDF file from the workspace for multimodal model input."
)
async def read_media(args: ReadArgs, ctx: ToolContext) -> ToolOutput:
    full = _resolve(ctx, args.path)
    if not full.is_file():
        raise ToolError(f"not a file: {args.path!r}")
    attachment = attachment_from_path(full)
    size = full.stat().st_size
    return ToolOutput(
        text=f"attached {attachment.name} ({attachment.media_type}, {size} bytes)",
        attachments=[attachment],
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
    entries = sorted(
        f"{p.name}/" if p.is_dir() else p.name for p in full.iterdir()
    )
    return "\n".join(entries) if entries else "(empty)"


class SearchArgs(BaseModel):
    query: str = Field(description="Substring to search for.")
    glob: str = Field(default="**/*", description="Glob of files to scan.")


@tool(description="Search workspace files for a substring; returns path:line matches.")
async def search(args: SearchArgs, ctx: ToolContext) -> str:
    base = ctx.workspace.resolve()
    _validate_search_glob(args.glob)
    hits: list[str] = []
    try:
        for p in base.glob(args.glob):
            try:
                full = p.resolve()
            except OSError:
                continue
            if full != base and not full.is_relative_to(base):
                continue
            if not full.is_file():
                continue
            try:
                if full.stat().st_size > _MAX_READ_BYTES:
                    continue
                text = full.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if args.query in line:
                    rel = p.relative_to(base)
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= _MAX_SEARCH_HITS:
                        hits.append(f"... (truncated at {_MAX_SEARCH_HITS} hits)")
                        return "\n".join(hits)
    except (NotImplementedError, ValueError) as e:
        raise ToolError(f"invalid search glob {args.glob!r}: {e}") from None
    return "\n".join(hits) if hits else "(no matches)"
