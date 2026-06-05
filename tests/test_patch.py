"""Tests for patch_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.errors import ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.patch import PatchArgs, patch_file


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


async def test_no_hunks_error(ctx):
    with pytest.raises(ToolError, match="no hunks"):
        await patch_file(PatchArgs(path="f.py", diff="--- a\n+++ b\n"), ctx)


async def test_missing_file(ctx):
    with pytest.raises(ToolError, match="not a file"):
        await patch_file(PatchArgs(path="nope.py", diff="@@ -1,1 +1,1 @@\n-x\n+y\n"), ctx)


async def test_escape_rejected(ctx):
    with pytest.raises(ToolError, match="escapes workspace"):
        await patch_file(PatchArgs(path="../evil.py", diff="@@ -1,1 +1,1 @@\n-a\n+b\n"), ctx)
