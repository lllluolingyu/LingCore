"""Tests for PromptComposer, ComposeContext, and LayeredComposer."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.composer import ComposeContext, LayeredComposer, StaticComposer


def _ctx(**kw) -> ComposeContext:
    return ComposeContext(user_message="hi", turn_index=0, **kw)


async def test_static_composer_returns_text():
    c = StaticComposer("hello world")
    assert await c.compose(_ctx()) == "hello world"


async def test_static_composer_ignores_ctx():
    c = StaticComposer("x")
    ctx = ComposeContext(user_message="hi", turn_index=99, session_id="s")
    assert await c.compose(ctx) == "x"


async def test_layered_composer_joins_layers():
    c = LayeredComposer(layers=["world", "role"], memory_path=None)
    result = await c.compose(_ctx())
    assert result == "world\n\nrole"


async def test_layered_composer_skips_blank_layers():
    c = LayeredComposer(layers=["a", "", "  ", "b"], memory_path=None)
    assert await c.compose(_ctx()) == "a\n\nb"


async def test_layered_composer_reads_memory(tmp_path: Path):
    mem = tmp_path / "memory.md"
    mem.write_text("## key\nvalue", encoding="utf-8")
    c = LayeredComposer(layers=["base"], memory_path=mem)
    result = await c.compose(_ctx())
    assert "base" in result and "## key" in result


async def test_layered_composer_absent_memory_ignored(tmp_path: Path):
    c = LayeredComposer(layers=["base"], memory_path=tmp_path / "no.md")
    assert await c.compose(_ctx()) == "base"


async def test_layered_composer_injects_active_skill():
    c = LayeredComposer(
        layers=["base"],
        memory_path=None,
        skill_instructions={"review": "do a review"},
    )
    result = await c.compose(_ctx(active_skills=("review",)))
    assert "do a review" in result


async def test_layered_composer_ignores_inactive_skill():
    c = LayeredComposer(
        layers=["base"],
        memory_path=None,
        skill_instructions={"review": "do a review"},
    )
    assert "do a review" not in await c.compose(_ctx())


async def test_compose_context_is_frozen():
    ctx = _ctx(active_skills=("s",))
    with pytest.raises((AttributeError, TypeError)):
        ctx.turn_index = 99  # type: ignore[misc]


async def test_compose_called_per_iteration(tmp_path: Path):
    """Agent re-composes every loop iteration, so memory writes appear next turn."""
    import lingcore.tools.builtin  # noqa: F401
    from lingcore.agent import Agent
    from lingcore.composer import LayeredComposer
    from lingcore.memory import WindowMemory
    from lingcore.message import ToolCall
    from lingcore.tools import ToolContext, ToolRegistry
    from tests.fakes import FakeLLMClient, ScriptedTurn

    mem = tmp_path / "memory.md"
    composer = LayeredComposer(layers=["base"], memory_path=mem)

    reg = ToolRegistry()
    call = ToolCall(id="c1", name="read_file", arguments={"path": "x.txt"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    (tmp_path / "x.txt").write_text("hello", encoding="utf-8")
    from lingcore.tools import REGISTRY
    reg.register(REGISTRY.get("read_file"))

    agent = Agent(
        llm=llm,
        tools=reg,
        tool_ctx=ToolContext(workspace=tmp_path),
        composer=composer,
        memory=WindowMemory(model="gpt-4o"),
    )

    # Write memory AFTER first compose call to verify the second call picks it up.
    mem.write_text("## note\nremembered", encoding="utf-8")
    events = [ev async for ev in agent.run("go")]

    # Second LLM call must have seen the memory content in the system prompt.
    second_system = llm.calls[1][0]  # first message is system
    assert "remembered" in second_system.content
