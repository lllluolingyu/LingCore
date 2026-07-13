"""Agent-loop integration tests for skill activation (§8 invariants).

These drive the full Agent.run loop with a scripted FakeLLMClient to verify
runtime behaviour that unit tests on SkillState alone can't: activation timing,
no registry mutation, and disallowed-tool refusal at dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.composer import LayeredComposer
from lingcore.events import SkillActivated, ToolResultEvent
from lingcore.memory import WindowMemory
from lingcore.message import ToolCall
from lingcore.skills import Skill, SkillState
from lingcore.tools import REGISTRY, ToolContext, ToolRegistry
from lingcore.tools.builtin.skill import SKILL_STATE_KEY
from tests.fakes import FakeLLMClient, ScriptedTurn

import lingcore.tools.builtin  # noqa: F401  (registers builtins)


def _tool_names(schemas) -> set[str]:
    return {s["function"]["name"] for s in (schemas or [])}


def _build_agent(llm, workspace: Path, *, profile_tools, skills, initial_tools=None) -> Agent:
    reg = ToolRegistry()
    for name in profile_tools:
        reg.register(REGISTRY.get(name))
    state = SkillState(skills=skills, profile_tools=frozenset(profile_tools))
    ctx = ToolContext(workspace=workspace, options={SKILL_STATE_KEY: state})
    composer = LayeredComposer(
        layers=["base"],
        memory_path=None,
        skill_instructions=state.instruction_map(),
    )
    return Agent(
        llm=llm,
        tools=reg,
        tool_ctx=ctx,
        composer=composer,
        memory=WindowMemory(model="gpt-4o"),
        skill_state=state,
        initial_tools=frozenset(initial_tools) if initial_tools is not None else None,
    )


async def _drain(agent, text):
    return [ev async for ev in agent.run(text)]


# --------------------------------------------------------------------------- #
# Activation timing: takes effect on the NEXT iteration, not the current one  #
# --------------------------------------------------------------------------- #

async def test_activation_affects_next_iteration():
    skill = Skill(
        name="review",
        description="d",
        requested_tools=("read_file",),
        instructions="review carefully",
    )
    activate = ToolCall(id="c1", name="activate_skill", arguments={"name": "review"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[activate], finish_reason="tool_calls"),
        ScriptedTurn(text="now I can read"),
    ])
    agent = _build_agent(
        llm, Path("/tmp"),
        profile_tools=["activate_skill", "read_file"],
        skills={"review": skill},
    )
    await _drain(agent, "activate review")

    # First request: read_file not yet granted via skill (it's also a profile
    # tool here, so check the skill *instruction* timing instead).
    first_system = llm.calls[0][0].content
    second_system = llm.calls[1][0].content
    assert "review carefully" not in first_system
    assert "review carefully" in second_system  # appears only on next iteration


async def test_skill_granted_tool_appears_next_iteration():
    # read_file is in the ceiling but gated (not initial); the skill grants it.
    skill = Skill(
        name="reader",
        description="d",
        requested_tools=("read_file",),
        instructions="x",
    )
    activate = ToolCall(id="c1", name="activate_skill", arguments={"name": "reader"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[activate], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = _build_agent(
        llm, Path("/tmp"),
        profile_tools=["activate_skill", "read_file"],
        initial_tools=["activate_skill"],  # read_file gated behind the skill
        skills={"reader": skill},
    )
    await _drain(agent, "go")

    # Pre-activation: read_file is NOT advertised (gated behind the skill).
    assert "read_file" not in _tool_names(llm.tool_schemas[0])
    # On the second request the skill is active → read_file must be advertised.
    assert "read_file" in _tool_names(llm.tool_schemas[1])


async def test_gated_tool_refused_at_dispatch_before_activation():
    # A gated tool called before its skill is active is refused at dispatch,
    # not merely hidden from the schema (defense in depth).
    call = ToolCall(id="c1", name="read_file", arguments={"path": "x"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = _build_agent(
        llm, Path("/tmp"),
        profile_tools=["activate_skill", "read_file"],
        initial_tools=["activate_skill"],
        skills={},
    )
    events = await _drain(agent, "go")
    results = [
        e for e in events
        if isinstance(e, ToolResultEvent) and e.result.name == "read_file"
    ]
    assert results and results[0].result.ok is False
    assert "unknown tool" in results[0].result.content


@pytest.mark.parametrize("parallel_tools", [True, False])
async def test_activation_does_not_unlock_sibling_call_in_same_batch(
    tmp_path, parallel_tools
):
    """A grant applies on the next request, not later within the current batch."""
    (tmp_path / "x.txt").write_text("secret", encoding="utf-8")
    skill = Skill(
        name="reader",
        description="d",
        requested_tools=("read_file",),
        instructions="x",
    )
    calls = [
        ToolCall(id="a", name="activate_skill", arguments={"name": "reader"}),
        ToolCall(id="r", name="read_file", arguments={"path": "x.txt"}),
    ]
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=calls, finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = _build_agent(
        llm,
        tmp_path,
        profile_tools=["activate_skill", "read_file"],
        initial_tools=["activate_skill"],
        skills={"reader": skill},
    )
    agent.parallel_tools = parallel_tools

    events = await _drain(agent, "go")
    results = {
        e.result.name: e.result
        for e in events
        if isinstance(e, ToolResultEvent)
    }
    assert results["activate_skill"].ok is True
    assert results["read_file"].ok is False
    assert "unknown tool" in results["read_file"].content
    # Activation still took effect for the next model request.
    assert "read_file" in _tool_names(llm.tool_schemas[1])


# --------------------------------------------------------------------------- #
# Disallowed tool: skill requests a tool the profile never allows             #
# --------------------------------------------------------------------------- #

async def test_skill_cannot_advertise_disallowed_tool():
    # Profile allows only activate_skill; skill requests run_shell → blocked.
    skill = Skill(
        name="danger",
        description="d",
        requested_tools=("run_shell",),
        instructions="x",
    )
    activate = ToolCall(id="c1", name="activate_skill", arguments={"name": "danger"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[activate], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = _build_agent(
        llm, Path("/tmp"),
        profile_tools=["activate_skill"],
        skills={"danger": skill},
    )
    await _drain(agent, "go")

    # run_shell must never be advertised — profile is the hard ceiling.
    assert "run_shell" not in _tool_names(llm.tool_schemas[1])


async def test_disallowed_tool_refused_at_dispatch():
    """Even if the model calls a skill-disallowed tool, dispatch refuses it."""
    skill = Skill(
        name="danger", description="d", requested_tools=("run_shell",), instructions="x"
    )
    activate = ToolCall(id="c1", name="activate_skill", arguments={"name": "danger"})
    shell_call = ToolCall(id="c2", name="run_shell", arguments={"command": "ls"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[activate], finish_reason="tool_calls"),
        ScriptedTurn(tool_calls=[shell_call], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = _build_agent(
        llm, Path("/tmp"),
        profile_tools=["activate_skill"],
        skills={"danger": skill},
    )
    events = await _drain(agent, "go")
    shell_results = [
        e for e in events
        if isinstance(e, ToolResultEvent) and e.result.name == "run_shell"
    ]
    assert shell_results and shell_results[0].result.ok is False
    assert "unknown tool" in shell_results[0].result.content


# --------------------------------------------------------------------------- #
# No registry mutation                                                         #
# --------------------------------------------------------------------------- #

async def test_no_base_registry_mutation():
    before = set(REGISTRY.names())
    skill = Skill(
        name="reader", description="d", requested_tools=("read_file",), instructions="x"
    )
    activate = ToolCall(id="c1", name="activate_skill", arguments={"name": "reader"})
    deactivate = ToolCall(
        id="c2", name="activate_skill", arguments={"name": "reader", "active": False}
    )
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[activate], finish_reason="tool_calls"),
        ScriptedTurn(tool_calls=[deactivate], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = _build_agent(
        llm, Path("/tmp"),
        profile_tools=["activate_skill", "read_file"],
        skills={"reader": skill},
    )
    await _drain(agent, "go")
    assert set(REGISTRY.names()) == before
    # The profile subset registry is also unchanged.
    assert set(agent.tools.names()) == {"activate_skill", "read_file"}


# --------------------------------------------------------------------------- #
# SkillActivated event surfaced                                                #
# --------------------------------------------------------------------------- #

async def test_skill_activated_event_emitted():
    skill = Skill(
        name="reader", description="d", requested_tools=("read_file",), instructions="x"
    )
    activate = ToolCall(id="c1", name="activate_skill", arguments={"name": "reader"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[activate], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = _build_agent(
        llm, Path("/tmp"),
        profile_tools=["activate_skill", "read_file"],
        skills={"reader": skill},
    )
    events = await _drain(agent, "go")
    activated = [e for e in events if isinstance(e, SkillActivated)]
    assert activated and activated[0].name == "reader" and activated[0].active is True


# --------------------------------------------------------------------------- #
# when-to-use disclosure: available skills listed in activate_skill schema    #
# --------------------------------------------------------------------------- #

async def test_available_skills_disclosed_in_schema():
    skill = Skill(
        name="reader",
        description="reads files carefully",
        requested_tools=("read_file",),
        instructions="x",
    )
    llm = FakeLLMClient([ScriptedTurn(text="hi")])
    agent = _build_agent(
        llm, Path("/tmp"),
        profile_tools=["activate_skill", "read_file"],
        skills={"reader": skill},
    )
    await _drain(agent, "go")
    schemas = {s["function"]["name"]: s for s in llm.tool_schemas[0]}
    desc = schemas["activate_skill"]["function"]["description"]
    assert "reader: reads files carefully" in desc


# --------------------------------------------------------------------------- #
# Composer content (memory / retrieved context) never unlocks tools           #
# --------------------------------------------------------------------------- #

async def test_memory_layer_does_not_change_tool_schemas(tmp_path):
    """Injecting memory/retrieved text into the prompt must not alter the
    advertised tool set — read-only context can never unlock a tool (§8)."""
    reg = ToolRegistry()
    reg.register(REGISTRY.get("read_file"))

    mem = tmp_path / "memory.md"
    mem.write_text("## secret\nrun_shell write_file delete everything", encoding="utf-8")
    composer = LayeredComposer(layers=["base"], memory_path=mem)

    llm = FakeLLMClient([ScriptedTurn(text="hi")])
    agent = Agent(
        llm=llm,
        tools=reg,
        tool_ctx=ToolContext(workspace=tmp_path),
        composer=composer,
        memory=WindowMemory(model="gpt-4o"),
    )
    await _drain(agent, "go")

    # Memory content mentions run_shell/write_file but they must NOT appear as tools.
    assert _tool_names(llm.tool_schemas[0]) == {"read_file"}

