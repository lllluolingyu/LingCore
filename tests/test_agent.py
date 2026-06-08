"""Tests for the agent loop and memory (M3).

Driven entirely by a scripted FakeLLMClient — no network, no API cost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.composer import StaticComposer
from lingcore.events import Error, Final, TextDelta, ToolCallStarted, ToolResultEvent
from lingcore.memory import WindowMemory
from lingcore.message import Message, ToolCall
from lingcore.tools import ToolContext, ToolRegistry, tool
from lingcore.tools.builtin.fs import read_file
from tests.fakes import FakeLLMClient, ScriptedTurn


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    return tmp_path


def _agent(llm, workspace, tools=("read_file",), **kw) -> Agent:
    reg = ToolRegistry()
    from lingcore.tools import REGISTRY

    for name in tools:
        reg.register(REGISTRY.get(name))
    return Agent(
        llm=llm,
        tools=reg,
        tool_ctx=ToolContext(workspace=workspace),
        composer=StaticComposer("You are a coding agent."),
        memory=WindowMemory(model="gpt-4o"),
        **kw,
    )


async def _drain(agent, text):
    return [ev async for ev in agent.run(text)]


async def test_simple_text_reply_streams(workspace):
    llm = FakeLLMClient([ScriptedTurn(text="Hi there!")])
    events = await _drain(_agent(llm, workspace), "hello")
    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(d.text for d in deltas) == "Hi there!"
    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1 and finals[0].content == "Hi there!"


async def test_tool_call_then_final(workspace):
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="The file says hello."),
    ])
    agent = _agent(llm, workspace)
    events = await _drain(agent, "read a.txt")

    starts = [e for e in events if isinstance(e, ToolCallStarted)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert starts[0].call.name == "read_file"
    assert results[0].result.ok is True
    assert results[0].result.content == "hello"
    assert isinstance(events[-1], Final)

    # The model saw the tool result on its second turn.
    second_turn_msgs = llm.calls[1]
    assert any(m.role == "tool" and m.content == "hello" for m in second_turn_msgs)


async def test_tool_error_is_contained(workspace):
    call = ToolCall(id="c1", name="read_file", arguments={"path": "missing.txt"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="Sorry, that file is missing."),
    ])
    agent = _agent(llm, workspace)
    events = await _drain(agent, "read missing.txt")

    result = [e for e in events if isinstance(e, ToolResultEvent)][0].result
    assert result.ok is False
    assert "ERROR" in result.content
    # Loop did not crash; it produced a final reply.
    assert isinstance(events[-1], Final)


async def test_parallel_tool_calls(workspace):
    (workspace / "b.txt").write_text("world", encoding="utf-8")
    calls = [
        ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"}),
        ToolCall(id="c2", name="read_file", arguments={"path": "b.txt"}),
    ]
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=calls, finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = _agent(llm, workspace)
    events = await _drain(agent, "read both")
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert {r.result.content for r in results} == {"hello", "world"}


async def test_max_iters_emits_error(workspace):
    # A model that always asks for a tool never terminates -> cap hits.
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})
    turns = [ScriptedTurn(tool_calls=[call], finish_reason="tool_calls") for _ in range(10)]
    llm = FakeLLMClient(turns)
    agent = _agent(llm, workspace, max_iters=3)
    events = await _drain(agent, "loop forever")
    assert isinstance(events[-1], Error)
    assert "max iterations" in events[-1].message


async def test_unknown_tool_contained(workspace):
    call = ToolCall(id="c1", name="ghost_tool", arguments={})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="recovered"),
    ])
    agent = _agent(llm, workspace)
    events = await _drain(agent, "call ghost")
    result = [e for e in events if isinstance(e, ToolResultEvent)][0].result
    assert result.ok is False
    assert "unknown tool" in result.content
