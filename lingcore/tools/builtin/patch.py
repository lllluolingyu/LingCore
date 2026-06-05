"""patch_file — apply a unified diff to a workspace file.

Accepts a standard unified diff (``diff -u`` / ``git diff`` output) for a
single file and applies it to the target path. Useful for multi-hunk edits
where ``edit_file`` would require many sequential calls.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool


class PatchArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    diff: str = Field(
        description=(
            "Unified diff for the file (output of `diff -u` or `git diff`). "
            "Must contain exactly one file's hunks; --- / +++ headers are optional."
        )
    )


def _resolve(ctx: ToolContext, path: str) -> Path:
    base = ctx.workspace.resolve()
    full = (base / path).resolve()
    if full != base and not full.is_relative_to(base):
        raise ToolError(f"path escapes workspace: {path!r}")
    return full


def _apply(text: str, diff: str) -> str:
    """Apply unified diff hunks to *text*. Returns the patched text."""
    lines = text.splitlines(keepends=True)
    # Ensure final newline is present so line indices stay stable.
    hunks = _parse_hunks(diff)
    if not hunks:
        raise ToolError("no hunks found in diff")

    # Apply hunks back-to-front so earlier line numbers stay valid.
    for orig_start, orig_len, new_lines in reversed(hunks):
        # orig_start is 1-based
        idx = orig_start - 1
        end = idx + orig_len
        if end > len(lines):
            raise ToolError(
                f"hunk @@ -{orig_start},{orig_len} … extends past end of file "
                f"({len(lines)} lines)"
            )
        lines[idx:end] = new_lines

    return "".join(lines)


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def _parse_hunks(
    diff: str,
) -> list[tuple[int, int, list[str]]]:
    """Return list of (orig_start, orig_len, new_lines) for each hunk."""
    hunks: list[tuple[int, int, list[str]]] = []
    orig_start = orig_len = 0
    new_lines: list[str] = []
    in_hunk = False

    for raw in diff.splitlines(keepends=True):
        line = raw.rstrip("\n")
        m = _HUNK_HEADER.match(line)
        if m:
            if in_hunk:
                hunks.append((orig_start, orig_len, new_lines))
            orig_start = int(m.group(1))
            orig_len = int(m.group(2)) if m.group(2) is not None else 1
            new_lines = []
            in_hunk = True
            continue
        if not in_hunk:
            continue  # skip --- / +++ headers and anything before first hunk
        if line.startswith("-"):
            pass  # removed line: consume from orig, don't add to new
        elif line.startswith("+"):
            new_lines.append(line[1:] + "\n")
        else:
            # context line (space or "\ No newline …")
            if not line.startswith("\\"):
                new_lines.append(line[1:] + "\n")

    if in_hunk:
        hunks.append((orig_start, orig_len, new_lines))
    return hunks


@tool(
    description=(
        "Apply a unified diff to a workspace file. Use for multi-hunk edits "
        "where edit_file would require many sequential calls."
    )
)
async def patch_file(args: PatchArgs, ctx: ToolContext) -> str:
    full = _resolve(ctx, args.path)
    if not full.is_file():
        raise ToolError(f"not a file: {args.path!r}")
    original = full.read_text("utf-8")
    patched = _apply(original, args.diff)
    full.write_text(patched, encoding="utf-8")
    orig_lines = original.count("\n")
    new_lines = patched.count("\n")
    return f"patched {args.path} ({orig_lines} → {new_lines} lines)"
