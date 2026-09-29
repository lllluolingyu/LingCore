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
    llm = FakeLLMClient(
        [
            ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
            ScriptedTurn(text="done"),
        ]
    )
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
    [ev async for ev in agent.run("go")]

    # Second LLM call must have seen the memory content in the system prompt.
    second_system = llm.calls[1][0]  # first message is system
    assert "remembered" in second_system.content


# --- project instructions (AGENTS.md / CLAUDE.md) ---------------------------


async def test_project_instructions_first_found_wins(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text("claude rules", encoding="utf-8")
    c = LayeredComposer(
        layers=["base"],
        memory_path=None,
        workspace=tmp_path,
        project_instructions=("AGENTS.md", "CLAUDE.md"),
    )
    out = await c.compose(_ctx())
    assert out.startswith("base\n\n# Project instructions (CLAUDE.md)")
    assert "claude rules" in out

    (tmp_path / "AGENTS.md").write_text("agents rules", encoding="utf-8")
    out = await c.compose(_ctx())  # re-read on every compose
    assert "agents rules" in out
    assert "claude rules" not in out


async def test_project_instructions_absent_adds_nothing(tmp_path: Path):
    c = LayeredComposer(
        layers=["base"],
        memory_path=None,
        workspace=tmp_path,
        project_instructions=("AGENTS.md",),
    )
    assert await c.compose(_ctx()) == "base"


async def test_project_instructions_never_follow_symlinks(tmp_path: Path):
    outside = tmp_path / "outside.md"
    outside.write_text("host secret", encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "AGENTS.md").symlink_to(outside)
    (ws / "docs").symlink_to(tmp_path, target_is_directory=True)
    from lingcore.composer import read_project_instructions

    assert read_project_instructions(ws, ["AGENTS.md", "docs/outside.md"]) is None


def test_project_instructions_are_bounded(tmp_path: Path):
    from lingcore import composer

    (tmp_path / "AGENTS.md").write_text(
        "x" * (composer.PROJECT_INSTRUCTIONS_MAX_CHARS + 50), encoding="utf-8"
    )
    out = composer.read_project_instructions(tmp_path, ["AGENTS.md"])
    assert out is not None and "truncated" in out
    assert len(out) < composer.PROJECT_INSTRUCTIONS_MAX_CHARS + 1000

    (tmp_path / "AGENTS.md").write_bytes(
        b"y" * (composer.PROJECT_INSTRUCTIONS_MAX_BYTES + 1)
    )
    out = composer.read_project_instructions(tmp_path, ["AGENTS.md"])
    assert out is not None and "was not loaded" in out
    assert "yyyy" not in out


async def test_from_profile_injects_workspace_instructions(tmp_path: Path):
    from lingcore.agent import Agent
    from lingcore.config import AgentProfile
    from tests.fakes import FakeLLMClient

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "AGENTS.md").write_text("run make check", encoding="utf-8")
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        f"""
name: p
workspace: {ws}
llm: {{model: m}}
persona:
  system_prompt: inline persona
  project_instructions: [AGENTS.md]
tools: [read_file]
""",
        encoding="utf-8",
    )
    agent = Agent.from_profile(AgentProfile.load(cfg), llm=FakeLLMClient([]))
    out = await agent.composer.compose(_ctx())
    assert "inline persona" in out
    assert "run make check" in out


@pytest.mark.parametrize("bad", ["../AGENTS.md", "/etc/passwd", "a//b", "", "./x"])
def test_project_instructions_config_rejects_unsafe_paths(tmp_path: Path, bad):
    from lingcore.config import AgentProfile
    from lingcore.errors import ConfigError

    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "name: p\nllm: {model: m}\ntools: []\n"
        f"persona: {{project_instructions: [{bad!r}]}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="project_instructions"):
        AgentProfile.load(cfg)
