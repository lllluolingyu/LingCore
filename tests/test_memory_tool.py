"""Tests for the memory builtin tool — covers all §8 security invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.errors import ConfigError, ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.memory import (
    MEMORY_SUMMARIZER_KEY,
    _PACKAGE_DIR,
    _compact_memory,
    memory,
)
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure


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
    package_profile = _PACKAGE_DIR / "some_profile"
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


# --------------------------------------------------------------------------- #
# Auto-compaction (opt-in, summarizer injected via MEMORY_SUMMARIZER_KEY)      #
# --------------------------------------------------------------------------- #


def _ctx_summ(tmp_path, summarizer, mem_opts=None):
    return ToolContext(
        workspace=tmp_path,
        profile_dir=tmp_path,
        options={"memory": mem_opts or {}, MEMORY_SUMMARIZER_KEY: summarizer},
    )


async def test_compact_memory_condenses():
    summ = FakeLLMClient([ScriptedTurn(text="## a\nshort\n\n## b\nalso short")])
    out = await _compact_memory(summ, "## a\n" + "x" * 1000, max_bytes=10_000)
    assert out is not None and "## a" in out and "## b" in out


async def test_compact_memory_rejects_unparseable():
    summ = FakeLLMClient([ScriptedTurn(text="plain prose, no headings")])
    assert await _compact_memory(summ, "## a\nx", max_bytes=10_000) is None


async def test_compact_memory_rejects_oversized():
    summ = FakeLLMClient([ScriptedTurn(text="## a\n" + "y" * 500)])
    assert await _compact_memory(summ, "## a\nx", max_bytes=50) is None


async def test_compact_memory_survives_summarizer_failure():
    summ = FakeLLMClient([StreamFailure(text="", reason="boom")])
    assert await _compact_memory(summ, "## a\nx", max_bytes=10_000) is None


async def test_auto_compact_shrinks_oversized_write(tmp_path):
    summ = FakeLLMClient([ScriptedTurn(text="## kept\ncondensed note")])
    ctx = _ctx_summ(
        tmp_path, summ, {"max_bytes": 200, "compact_at_ratio": 0.5, "auto_compact": True}
    )
    await memory(memory.args_model(action="remember", key="a", content="x" * 40), ctx)
    # This second write crosses compact_at (0.5 × 200 = 100) → compaction fires
    # and rescues a write that would otherwise have approached the cap.
    out = await memory(memory.args_model(action="remember", key="b", content="y" * 120), ctx)
    assert "auto-compacted" in out
    result = await memory(memory.args_model(action="read"), ctx)
    assert "condensed note" in result
    assert len(result.encode()) <= 200


async def test_summarizer_present_but_auto_compact_off_hard_fails(tmp_path):
    # Defense in depth: even with a summarizer injected, a context that didn't
    # opt into auto_compact must NOT compact — it hits the hard cap like before.
    summ = FakeLLMClient([ScriptedTurn(text="## kept\ncondensed note")])
    ctx = _ctx_summ(tmp_path, summ, {"max_bytes": 50, "compact_at_ratio": 0.5})
    with pytest.raises(ToolError, match="max_bytes"):
        await memory(memory.args_model(action="remember", key="k", content="x" * 100), ctx)


async def test_compaction_unparseable_falls_back_to_hard_cap(tmp_path):
    # Summarizer present but its reply has no ## sections → _compact_memory
    # returns None, so the original (oversized) content hits the hard cap.
    summ = FakeLLMClient([ScriptedTurn(text="not markdown, no headings")])
    ctx = _ctx_summ(
        tmp_path, summ, {"max_bytes": 50, "compact_at_ratio": 0.5, "auto_compact": True}
    )
    with pytest.raises(ToolError, match="max_bytes"):
        await memory(memory.args_model(action="remember", key="k", content="x" * 100), ctx)


def test_from_profile_injects_summarizer_when_enabled(tmp_path):
    from lingcore.agent import Agent
    from lingcore.config import AgentProfile

    root = tmp_path / "p"
    root.mkdir()
    (root / "config.yaml").write_text(
        "name: t\nllm:\n  model: m\ntools: [memory]\n"
        "tool_options:\n  memory:\n    auto_compact: true\n",
        encoding="utf-8",
    )
    agent = Agent.from_profile(AgentProfile.load(root), llm=FakeLLMClient([]))
    assert MEMORY_SUMMARIZER_KEY in agent.tool_ctx.options


def test_from_profile_no_summarizer_by_default(tmp_path):
    from lingcore.agent import Agent
    from lingcore.config import AgentProfile

    root = tmp_path / "p"
    root.mkdir()
    (root / "config.yaml").write_text(
        "name: t\nllm:\n  model: m\ntools: [memory]\n", encoding="utf-8"
    )
    agent = Agent.from_profile(AgentProfile.load(root), llm=FakeLLMClient([]))
    assert MEMORY_SUMMARIZER_KEY not in agent.tool_ctx.options
