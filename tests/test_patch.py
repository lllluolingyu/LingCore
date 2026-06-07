"""Tests for patch_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.errors import ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.patch import PatchArgs, _apply, patch_file


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "f.py").write_text("a\nb\nc\n", encoding="utf-8")
    return ToolContext(workspace=tmp_path)


async def test_basic_replace(ctx):
    diff = "@@ -2,1 +2,1 @@\n-b\n+B\n"
    out = await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nB\nc\n"
    assert "patched" in out


async def test_add_line(ctx):
    diff = "@@ -3,1 +3,2 @@\n c\n+d\n"
    await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nc\nd\n"


async def test_delete_line(ctx):
    diff = "@@ -2,1 +2,0 @@\n-b\n"
    await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nc\n"


async def test_multi_hunk(ctx):
    (ctx.workspace / "f.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    diff = "@@ -1,1 +1,1 @@\n-a\n+A\n@@ -5,1 +5,1 @@\n-e\n+E\n"
    await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "A\nb\nc\nd\nE\n"


async def test_rejects_stale_removed_line(ctx):
    diff = "@@ -2,1 +2,1 @@\n-x\n+B\n"
    with pytest.raises(ToolError, match="does not match"):
        await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nc\n"


async def test_rejects_stale_context_line(ctx):
    diff = "@@ -2,2 +2,2 @@\n nope\n-c\n+C\n"
    with pytest.raises(ToolError, match="does not match"):
        await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nc\n"


async def test_patches_last_line_without_trailing_newline(ctx):
    # Very common: the file's last line has no trailing newline. The diff still
    # carries one; the edit must apply and the no-newline state be preserved.
    (ctx.workspace / "f.py").write_text("a\nb\nc", encoding="utf-8")
    diff = "@@ -3,1 +3,1 @@\n-c\n+C\n"
    await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nC"


async def test_middle_edit_preserves_missing_final_newline(ctx):
    (ctx.workspace / "f.py").write_text("a\nb\nc", encoding="utf-8")
    diff = "@@ -1,1 +1,1 @@\n-a\n+A\n"
    await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "A\nb\nc"


def test_apply_tolerates_crlf_last_line():
    # A CRLF terminator on the final line differs from the diff's bare LF only
    # in the line ending; _apply should tolerate it rather than reject.
    assert _apply("a\nb\nc\r\n", "@@ -3,1 +3,1 @@\n-c\n+C\n") == "a\nb\nC\n"


def test_apply_error_message_shows_newline_difference():
    # The mismatch report must reveal the real bytes, not rstrip both sides to
    # identical-looking strings.
    with pytest.raises(ToolError, match=r"expected 'x\\n'"):
        _apply("a\nb\nc\n", "@@ -2,1 +2,1 @@\n-x\n+B\n")


async def test_rejects_overlapping_hunks(ctx):
    # First hunk consumes lines 1-2; the second starts inside that range.
    diff = "@@ -1,2 +1,1 @@\n-a\n-b\n+A\n@@ -2,1 +2,1 @@\n-b\n+X\n"
    with pytest.raises(ToolError, match="overlap or are out of order"):
        await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nc\n"


async def test_rejects_out_of_order_hunks(ctx):
    diff = "@@ -3,1 +3,1 @@\n-c\n+C\n@@ -1,1 +1,1 @@\n-a\n+A\n"
    with pytest.raises(ToolError, match="overlap or are out of order"):
        await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nc\n"


async def test_zero_length_insertion_applies(ctx):
    diff = "@@ -3,0 +4,1 @@\n+d\n"
    await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nc\nd\n"


async def test_rejects_duplicate_insertion_hunks(ctx):
    diff = "@@ -3,0 +4,1 @@\n+d\n@@ -3,0 +4,1 @@\n+e\n"
    with pytest.raises(ToolError, match="duplicate insertion"):
        await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nc\n"


async def test_rejects_mismatched_header_path(ctx):
    diff = "--- a/other.py\n+++ b/other.py\n@@ -2,1 +2,1 @@\n-b\n+B\n"
    with pytest.raises(ToolError, match="diff header targets"):
        await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nb\nc\n"


async def test_accepts_matching_header_path(ctx):
    diff = "--- a/f.py\n+++ b/f.py\n@@ -2,1 +2,1 @@\n-b\n+B\n"
    await patch_file(PatchArgs(path="f.py", diff=diff), ctx)
    assert (ctx.workspace / "f.py").read_text() == "a\nB\nc\n"


async def test_no_hunks_error(ctx):
    with pytest.raises(ToolError, match="no hunks"):
        await patch_file(PatchArgs(path="f.py", diff="--- a\n+++ b\n"), ctx)


async def test_missing_file(ctx):
    with pytest.raises(ToolError, match="not a file"):
        await patch_file(PatchArgs(path="nope.py", diff="@@ -1,1 +1,1 @@\n-x\n+y\n"), ctx)


async def test_escape_rejected(ctx):
    with pytest.raises(ToolError, match="escapes workspace"):
        await patch_file(PatchArgs(path="../evil.py", diff="@@ -1,1 +1,1 @@\n-a\n+b\n"), ctx)
