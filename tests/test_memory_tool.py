"""Tests for the memory builtin tool — covers all §8 security invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.errors import ConfigError, ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.memory import _PACKAGE_DIR, memory


def _ctx(tmp_path: Path, opts: dict | None = None) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        profile_dir=tmp_path,
        options={"memory": opts or {}},
    )


# --------------------------------------------------------------------------- #
# Basic operations                                                             #
# --------------------------------------------------------------------------- #

async def test_remember_and_read(tmp_path):
    ctx = _ctx(tmp_path)
    await memory(memory.args_model(action="remember", key="k", content="v"), ctx)
    result = await memory(memory.args_model(action="read"), ctx)
    assert "## k" in result and "v" in result


async def test_modify(tmp_path):
    ctx = _ctx(tmp_path)
    await memory(memory.args_model(action="remember", key="k", content="old"), ctx)
    await memory(memory.args_model(action="modify", key="k", content="new"), ctx)
    result = await memory(memory.args_model(action="read"), ctx)
    assert "new" in result and "old" not in result


async def test_forget(tmp_path):
    ctx = _ctx(tmp_path)
    await memory(memory.args_model(action="remember", key="k", content="v"), ctx)
    await memory(memory.args_model(action="forget", key="k"), ctx)
    result = await memory(memory.args_model(action="read"), ctx)
    assert "k" not in result


async def test_read_empty(tmp_path):
    ctx = _ctx(tmp_path)
    result = await memory(memory.args_model(action="read"), ctx)
    assert "empty" in result.lower()


# --------------------------------------------------------------------------- #
# Strict key semantics (§8 invariants)                                        #
# --------------------------------------------------------------------------- #

async def test_remember_fails_if_key_exists(tmp_path):
    ctx = _ctx(tmp_path)
    await memory(memory.args_model(action="remember", key="k", content="v"), ctx)
    with pytest.raises(ToolError, match="already exists"):
        await memory(memory.args_model(action="remember", key="k", content="v2"), ctx)


async def test_modify_fails_if_key_missing(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(ToolError, match="not found"):
        await memory(memory.args_model(action="modify", key="missing", content="x"), ctx)


async def test_forget_fails_if_key_missing(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(ToolError, match="not found"):
        await memory(memory.args_model(action="forget", key="missing"), ctx)


# --------------------------------------------------------------------------- #
# max_bytes on final content (§8)                                             #
# --------------------------------------------------------------------------- #

async def test_max_bytes_on_final_content(tmp_path):
    ctx = _ctx(tmp_path, {"max_bytes": 50})
    with pytest.raises(ToolError, match="max_bytes"):
        await memory(memory.args_model(action="remember", key="k", content="x" * 100), ctx)


async def test_max_bytes_gradual_growth(tmp_path):
    """Two small writes that each fit but together exceed max_bytes."""
    ctx = _ctx(tmp_path, {"max_bytes": 60})
    await memory(memory.args_model(action="remember", key="a", content="x" * 10), ctx)
    with pytest.raises(ToolError, match="max_bytes"):
        await memory(memory.args_model(action="remember", key="b", content="x" * 40), ctx)


# --------------------------------------------------------------------------- #
# Path confinement (§8)                                                       #
# --------------------------------------------------------------------------- #

async def test_relative_path_escape_rejected(tmp_path):
    ctx = _ctx(tmp_path, {"path": "../../evil.md"})
    with pytest.raises(ConfigError, match="escapes"):
        await memory(memory.args_model(action="remember", key="k", content="v"), ctx)


async def test_absolute_path_rejected_by_default(tmp_path):
    ctx = _ctx(tmp_path, {"path": "/tmp/evil.md"})
    with pytest.raises(ConfigError, match="allow_absolute_path"):
        await memory(memory.args_model(action="remember", key="k", content="v"), ctx)


async def test_absolute_path_allowed_with_flag(tmp_path):
    target = tmp_path / "sub" / "mem.md"
    ctx = _ctx(tmp_path, {"path": str(target), "allow_absolute_path": True})
    await memory(memory.args_model(action="remember", key="k", content="v"), ctx)
    assert target.is_file()


# --------------------------------------------------------------------------- #
# Built-in package profile guard (§8)                                         #
# --------------------------------------------------------------------------- #

async def test_package_dir_write_blocked(tmp_path):
    """profile_dir inside the installed package must not be writable."""
    package_profile = _PACKAGE_DIR / "profiles" / "coding"
    ctx = ToolContext(
        workspace=tmp_path,
        profile_dir=package_profile,
        options={"memory": {}},
    )
    with pytest.raises(ToolError, match="installed package"):
        await memory(memory.args_model(action="remember", key="k", content="v"), ctx)


# --------------------------------------------------------------------------- #
# No profile_dir                                                               #
# --------------------------------------------------------------------------- #

async def test_no_profile_dir_raises(tmp_path):
    ctx = ToolContext(workspace=tmp_path, options={"memory": {}})
    with pytest.raises(ToolError, match="profile_dir"):
        await memory(memory.args_model(action="remember", key="k", content="v"), ctx)
