from __future__ import annotations

import asyncio

import pytest

from lingcore.agent import Agent
from lingcore.background import BackgroundInput
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.events import Error, ToolResultEvent
from lingcore.execution import ExecutionBudget, ExecutionLimits, run_headless
from lingcore.llm import LLMChunk
from lingcore.message import ToolCall
from lingcore.tools import ToolRegistry
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure
from tests.test_trace_runtime import Metered, profile


def build(tmp_path, turns, limits, *, tracing=False):
    raw = Metered(turns)
    prof = profile(tmp_path)
    prof.tracing.enabled = tracing
    agent = Agent.from_profile(
        prof, llm=raw, execution_budget=ExecutionBudget(ExecutionLimits(**limits))
    )
    raw.agent = agent
    return agent, raw


@pytest.mark.parametrize("limit", [{"model_requests": 1}, {"reported_tokens": 19}])
async def test_request_and_token_budget_are_terminal(tmp_path, limit):
    agent, raw = build(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="a", name="read_file", arguments={"path": "missing"})
                ]
            ),
            ScriptedTurn(text="must not run"),
        ],
        limit,
    )
    events = [e async for e in agent.run("do it")]
    assert isinstance(events[-1], Error)
    assert agent.execution_outcome.status == "budget_exceeded"
    assert agent.execution_outcome.limit_type == next(iter(limit))
    assert len(raw.calls) == 1
    assert [m.role for m in agent.memory.messages] == ["user"]
    await agent.aclose()


async def test_rejected_call_counts_toward_tool_budget(tmp_path):
    agent, raw = build(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="a", name="unknown"),
                    ToolCall(id="b", name="also_unknown"),
                ]
            )
        ],
        {"tool_calls": 1},
    )
    events = [e async for e in agent.run("do it")]
    assert isinstance(events[-1], Error)
    assert agent.execution_outcome.limit_type == "tool_calls"
    assert not any(isinstance(e, ToolResultEvent) for e in events)
    assert len(raw.calls) == 1
    await agent.aclose()


async def test_retry_exhaustion_stops_before_second_request(tmp_path, monkeypatch):
    monkeypatch.setattr("lingcore.agent._backoff_seconds", lambda attempt: 0)
    agent, raw = build(
        tmp_path,
        [StreamFailure(text="partial"), ScriptedTurn(text="no")],
        {"model_requests": 1},
    )
    [e async for e in agent.run("go")]
    assert agent.execution_outcome.limit_type == "model_requests"
    assert len(raw.calls) == 1 and len(agent.request_ledger.records) == 1
    await agent.aclose()


async def test_strict_missing_usage_has_distinct_cause(tmp_path):
    raw = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="a", name="read_file", arguments={"path": "missing"})
                ]
            ),
            ScriptedTurn(text="no"),
        ]
    )
    agent = Agent.from_profile(
        profile(tmp_path),
        llm=raw,
        execution_budget=ExecutionBudget(ExecutionLimits(usage_missing="stop")),
    )
    [e async for e in agent.run("go")]
    outcome = agent.execution_outcome
    assert (
        outcome.status == "failed"
        and outcome.provider_error_category == "usage_unavailable"
    )
    assert outcome.limit_type is None and len(raw.calls) == 1
    await agent.aclose()


async def test_deadline_uses_stop_handshake_and_closes_agent(tmp_path):
    class Hanging:
        async def stream(self, messages, tools=None):
            yield LLMChunk(text_delta="partial")
            await asyncio.Event().wait()

    agent = Agent.from_profile(
        profile(tmp_path),
        llm=Hanging(),
        execution_budget=ExecutionBudget(ExecutionLimits(wall_seconds=0.03)),
    )
    result = await run_headless(agent, ["one", "never"])
    assert result.outcome.status == "timed_out"
    assert not agent.turn_pending_finalization and agent._closed
    assert [m.role for m in agent.memory.messages] == ["user"]
    assert result.trace.complete


async def test_scripted_driver_reuses_session_and_closes_on_normal_exit(tmp_path):
    agent, raw = build(
        tmp_path,
        [ScriptedTurn(text="one"), ScriptedTurn(text="two")],
        {"model_requests": 2},
    )
    result = await run_headless(agent, ["one", "two"])
    assert result.outcome.status == "completed" and agent._closed
    assert len(result.usage) == 2 and len(raw.calls[1]) > len(raw.calls[0])
    assert agent.background.enabled is False


async def test_automatic_turns_need_explicit_budget(tmp_path):
    agent, raw = build(tmp_path, [ScriptedTurn(text="never")], {})
    events = [e async for e in agent.run(BackgroundInput(text="monitor"))]
    assert isinstance(events[-1], Error) and not raw.calls
    assert agent.execution_outcome.limit_type == "automatic_turns"
    await agent.aclose()


async def test_separate_state_keeps_source_inputs_and_excludes_prior_memory(tmp_path):
    prof = profile(
        tmp_path,
        content="redacted",
        tools=["memory", "calendar_add"],
        plugins=["calendar"],
    )
    source = tmp_path / "source"
    (source / "memory.md").write_text("SOURCE PRIVATE MEMORY")
    (source / "world.md").write_text("READ ONLY SOURCE PROMPT")
    state = tmp_path / "state"
    raw = Metered(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="m",
                        name="memory",
                        arguments={
                            "action": "remember",
                            "key": "new",
                            "content": "fresh memory",
                        },
                    ),
                    ToolCall(
                        id="c",
                        name="calendar_add",
                        arguments={
                            "events": [
                                {
                                    "title": "hello",
                                    "start": "2026-10-29T12:00:00+08:00",
                                    "duration_minutes": 30,
                                }
                            ]
                        },
                    ),
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent.from_profile(prof, llm=raw, state_dir=state)
    raw.agent = agent
    [e async for e in agent.run("go")]
    assert "READ ONLY SOURCE PROMPT" in raw.calls[0][0].content
    assert "SOURCE PRIVATE MEMORY" not in raw.calls[0][0].content
    assert (state / "memory.md").exists() and (state / "calendar.ics").exists()
    assert (state / "sessions.db").exists() and (state / "traces").is_dir()
    assert not (source / "sessions.db").exists()
    assert (source / "memory.md").read_text() == "SOURCE PRIVATE MEMORY"
    await agent.aclose()


def test_injected_registry_has_no_builtin_fallback(tmp_path):
    prof = AgentProfile.model_validate({"llm": {"model": "m"}, "tools": ["read_file"]})
    with pytest.raises(ConfigError, match="ceiling exactly"):
        Agent.from_profile(
            prof, llm=FakeLLMClient([]), tool_registry=ToolRegistry(), base_dir=tmp_path
        )


def test_injected_registry_rejects_plugin_loading(tmp_path):
    prof = AgentProfile.model_validate({"llm": {"model": "m"}, "plugins": ["clock"]})
    with pytest.raises(ConfigError, match="plugins or skills"):
        Agent.from_profile(
            prof, llm=FakeLLMClient([]), tool_registry=ToolRegistry(), base_dir=tmp_path
        )
