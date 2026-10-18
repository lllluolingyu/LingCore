"""patch_file — apply a unified diff to a workspace file.

Accepts a standard unified diff (``diff -u`` / ``git diff`` output) for a
single file and applies it to the target path. Useful for multi-hunk edits
where ``edit_file`` would require many sequential calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.paths import PathEscapeError, confined_directory, resolve_confined
from lingcore.tools import ToolContext, tool


class PatchArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    diff: str = Field(
        description=(
            "Unified diff for the file (output of `diff -u` or `git diff`). "
            "Must contain exactly one file's hunks; --- / +++ headers are optional."
        )
    )


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


def _file_newline(text: str) -> str | None:
    """The file's non-LF line terminator, if it uses one (``\\r\\n`` or ``\\r``)."""
    return "\r\n" if "\r\n" in text else "\r" if "\r" in text else None


def _with_newline(lines: list[str], newline: str) -> list[str]:
    """Translate LF-terminated diff lines to the file's own terminator."""
    return [
        line[:-1] + newline
        if line.endswith("\n") and not line.endswith("\r\n")
        else line
        for line in lines
    ]


def _apply(text: str, diff: str, *, target: str | None = None) -> str:
    """Apply unified diff hunks to *text*. Returns the patched text.

    When *target* is given and the diff carries ``---``/``+++`` headers, the
    headers must name the same file, so a diff for one file can't be applied to
    another by mistake.
    """
    lines = text.splitlines(keepends=True)
    ended_with_newline = text.endswith(("\n", "\r"))
    newline = _file_newline(text)
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
        at_eof = end == len(lines)
        new_lines = hunk.new_lines
        if not _lines_match(actual, hunk.old_lines, at_eof=at_eof):
            # read_file shows normalized lines, so a model writes LF hunks even
            # for a CRLF/CR file. Retry with the file's terminator (like
            # edit_file's _match_newlines) and keep its endings in the result.
            if newline is None or not _lines_match(
                actual, _with_newline(hunk.old_lines, newline), at_eof=at_eof
            ):
                raise ToolError(
                    f"hunk @@ -{hunk.orig_start},{hunk.orig_len} does not match file "
                    f"content; expected {''.join(hunk.old_lines)!r}, "
                    f"got {''.join(actual)!r}"
                )
            new_lines = _with_newline(new_lines, newline)
        lines[idx:end] = new_lines

    result = "".join(lines)
    # Diff lines are always LF-terminated; restore the file's original
    # trailing-newline state so patching the last line doesn't silently add or
    # drop a final newline.
    if ended_with_newline:
        if result and not result.endswith(("\n", "\r")):
            result += newline or "\n"
    elif result.endswith("\r\n"):
        result = result[:-2]
    elif result.endswith(("\n", "\r")):
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
    Two zero-length insertions at the same anchor are rejected — they consume
    nothing, so the range check can't catch them, and their reverse-order
    application would silently swap the inserted blocks.
    """
    prev_end = 0
    last_insert_at: int | None = None
    for h in hunks:
        idx = h.orig_start if h.orig_len == 0 else h.orig_start - 1
        if idx < 0:
            raise ToolError(f"invalid hunk start in @@ -{h.orig_start},{h.orig_len}")
        if idx < prev_end:
            raise ToolError(
                f"diff hunks overlap or are out of order at @@ -{h.orig_start},{h.orig_len}"
            )
        if h.orig_len == 0:
            if idx == last_insert_at:
                raise ToolError(
                    f"diff has duplicate insertion hunks at the same position "
                    f"@@ -{h.orig_start},0"
                )
            last_insert_at = idx
            prev_end = max(prev_end, idx)
        else:
            last_insert_at = None
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
    base = ctx.workspace.resolve()
    try:
        rel = resolve_confined(base, args.path).relative_to(base)
        # Like edit_file: one descriptor-anchored read/write keeps the
        # validated parent in place for the whole patch, and the file is read
        # as bytes so its line endings survive (text mode would turn CRLF
        # into LF on every patch).
        with confined_directory(base, rel.parent) as directory:
            with directory.open_regular(rel.name, "r+b") as stream:
                try:
                    original = stream.read().decode("utf-8")
                except UnicodeDecodeError:
                    raise ToolError(f"not a UTF-8 text file: {args.path!r}") from None
                patched = _apply(original, args.diff, target=args.path)
                stream.seek(0)
                stream.write(patched.encode("utf-8"))
                stream.truncate()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        raise ToolError(f"not a file: {args.path!r}") from None
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from None
    except OSError as exc:
        raise ToolError(f"cannot patch {args.path!r}: {exc}") from None
    orig_lines = original.count("\n")
    new_lines = patched.count("\n")
    return f"patched {args.path} ({orig_lines} → {new_lines} lines)"
