"""patch_file — apply a unified diff to a workspace file.

Accepts a standard unified diff (``diff -u`` / ``git diff`` output) for a
single file and applies it to the target path. Useful for multi-hunk edits
where ``edit_file`` would require many sequential calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path, PurePosixPath

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


@dataclass(slots=True)
class _Hunk:
    orig_start: int
    orig_len: int
    new_start: int
    new_len: int
    old_lines: list[str]
    new_lines: list[str]


def _lines_match(actual: list[str], expected: list[str], *, at_eof: bool) -> bool:
    """Compare a file slice against a hunk's old-lines.

    The final line of a file may legitimately lack a trailing newline (or carry
    a CRLF where the diff uses a bare LF). Tolerate a difference that is *only*
    the trailing line terminator, and only on the last line at end-of-file —
    interior lines must still match exactly so stale diffs are still rejected.
    """
    if len(actual) != len(expected):
        return False
    last = len(actual) - 1
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a == e:
            continue
        if at_eof and i == last and a.rstrip("\r\n") == e.rstrip("\r\n"):
            continue
        return False
    return True


def _apply(text: str, diff: str, *, target: str | None = None) -> str:
    """Apply unified diff hunks to *text*. Returns the patched text.

    When *target* is given and the diff carries ``---``/``+++`` headers, the
    headers must name the same file, so a diff for one file can't be applied to
    another by mistake.
    """
    lines = text.splitlines(keepends=True)
    ended_with_newline = text.endswith("\n")
    hunks = _parse_hunks(diff)
    if not hunks:
        raise ToolError("no hunks found in diff")
    if target is not None:
        _check_header_path(diff, target)
    _validate_hunk_ranges(hunks)

    # Apply hunks back-to-front so earlier line numbers stay valid.
    for hunk in reversed(hunks):
        # For pure insertion hunks, unified diff line numbers point after the
        # previous line. Otherwise they point at the first consumed line.
        idx = hunk.orig_start if hunk.orig_len == 0 else hunk.orig_start - 1
        end = idx + hunk.orig_len
        if end > len(lines):
            raise ToolError(
                f"hunk @@ -{hunk.orig_start},{hunk.orig_len} extends past end of file "
                f"({len(lines)} lines)"
            )
        actual = lines[idx:end]
        if not _lines_match(actual, hunk.old_lines, at_eof=end == len(lines)):
            raise ToolError(
                f"hunk @@ -{hunk.orig_start},{hunk.orig_len} does not match file "
                f"content; expected {''.join(hunk.old_lines)!r}, "
                f"got {''.join(actual)!r}"
            )
        lines[idx:end] = hunk.new_lines

    result = "".join(lines)
    # Diff lines are always LF-terminated; restore the file's original
    # trailing-newline state so patching the last line doesn't silently add or
    # drop a final newline.
    if ended_with_newline:
        if result and not result.endswith("\n"):
            result += "\n"
    elif result.endswith("\r\n"):
        result = result[:-2]
    elif result.endswith("\n"):
        result = result[:-1]
    return result


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_hunks(diff: str) -> list[_Hunk]:
    """Parse unified diff hunks and validate their declared line counts."""
    hunks: list[_Hunk] = []
    orig_start = orig_len = new_start = new_len = 0
    old_lines: list[str] = []
    new_lines: list[str] = []
    in_hunk = False

    def finish_hunk() -> None:
        if not in_hunk:
            return
        if len(old_lines) != orig_len:
            raise ToolError(
                f"hunk @@ -{orig_start},{orig_len} declares {orig_len} old lines "
                f"but contains {len(old_lines)}"
            )
        if len(new_lines) != new_len:
            raise ToolError(
                f"hunk @@ +{new_start},{new_len} declares {new_len} new lines "
                f"but contains {len(new_lines)}"
            )
        hunks.append(
            _Hunk(
                orig_start=orig_start,
                orig_len=orig_len,
                new_start=new_start,
                new_len=new_len,
                old_lines=list(old_lines),
                new_lines=list(new_lines),
            )
        )

    for raw in diff.splitlines(keepends=True):
        line = raw.rstrip("\n")
        m = _HUNK_HEADER.match(line)
        if m:
            if in_hunk:
                finish_hunk()
            orig_start = int(m.group(1))
            orig_len = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            old_lines = []
            new_lines = []
            in_hunk = True
            continue
        if not in_hunk:
            continue  # skip --- / +++ headers and anything before first hunk
        if raw.startswith(" "):
            content = raw[1:]
            old_lines.append(content)
            new_lines.append(content)
        elif raw.startswith("-"):
            old_lines.append(raw[1:])
        elif raw.startswith("+"):
            new_lines.append(raw[1:])
        elif raw.startswith("\\"):
            continue
        else:
            raise ToolError(f"invalid diff line in hunk: {line!r}")

    if in_hunk:
        finish_hunk()
    return hunks


def _validate_hunk_ranges(hunks: list[_Hunk]) -> None:
    """Reject hunks that are out of order or whose consumed ranges overlap.

    Hunks are applied back-to-front, which assumes they are sorted by position
    and don't overlap; a malformed (model-generated) diff that violates this
    would otherwise corrupt the file. Ranges are the same 0-based slices _apply
    consumes, so adjacent hunks (one ending where the next begins) are allowed.
    """
    prev_end = 0
    for h in hunks:
        idx = h.orig_start if h.orig_len == 0 else h.orig_start - 1
        if idx < 0:
            raise ToolError(f"invalid hunk start in @@ -{h.orig_start},{h.orig_len}")
        if idx < prev_end:
            raise ToolError(
                f"diff hunks overlap or are out of order at @@ -{h.orig_start},{h.orig_len}"
            )
        prev_end = idx + h.orig_len


def _header_path(diff: str) -> str | None:
    """Return the target path from a unified-diff ``+++`` header, if present."""
    for raw in diff.splitlines():
        if raw.startswith("@@"):
            break  # headers, if any, precede the first hunk
        if raw.startswith("+++ "):
            field = raw[4:].split("\t", 1)[0].strip()
            if not field or field == "/dev/null":
                return field or None
            if field.startswith(("a/", "b/")):
                field = field[2:]
            return field
    return None


def _check_header_path(diff: str, target: str) -> None:
    header = _header_path(diff)
    if header is None or header == "/dev/null":
        return
    if PurePosixPath(header) != PurePosixPath(target):
        raise ToolError(
            f"diff header targets {header!r} but patch_file was asked to patch {target!r}"
        )


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
    patched = _apply(original, args.diff, target=args.path)
    full.write_text(patched, encoding="utf-8")
    orig_lines = original.count("\n")
    new_lines = patched.count("\n")
    return f"patched {args.path} ({orig_lines} → {new_lines} lines)"
