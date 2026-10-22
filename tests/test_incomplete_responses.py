"""Malformed and budget-limited responses must never silently execute or finish."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.events import (
    Error,
    Final,
    StreamRetry,
    ToolCallStarted,
    ToolResultEvent,
    TurnStatsReported,
)
from lingcore.message import Message, ToolCall
from lingcore.sessions import SessionStore
from lingcore.tools import ToolContext, ToolRegistry, tool
from tests.fakes import FakeLLMClient, ScriptedTurn


@pytest.mark.parametrize("raw", ["", '{"value":', "[]", "null"])
async def test_invalid_arguments_never_execute_defaulted_tool(tmp_path, raw):
    class Args(BaseModel):
        value: str = "default"

    ran = []
    registry = ToolRegistry()

    # Use an explicit Tool here: the decorator resolves annotations from module globals.
    from lingcore.tools import Tool

    async def execute(args, ctx):
        ran.append(args.value)
        return "executed"

    registry.register(Tool("probe", "probe", Args, execute))
    calls = [
        ToolCall(id="bad", name="probe", invalid_arguments=raw),
        ToolCall(id="good", name="probe", arguments={"value": "explicit"}),
    ]
    llm = FakeLLMClient([ScriptedTurn(tool_calls=calls), ScriptedTurn(text="done")])
    async with Agent(
        llm, registry, ToolContext(workspace=tmp_path), system_prompt="test"
    ) as agent:
        events = [e async for e in agent.run("go")]
    assert ran == ["explicit"]
    results = [e.result for e in events if isinstance(e, ToolResultEvent)]
    assert [r.ok for r in results] == [False, True]
    assert "not run" in results[0].content
    assert isinstance(events[-1], Final)


@pytest.mark.parametrize("reason", ["length", "model_context_window_exceeded"])
@pytest.mark.parametrize("text", ["partial answer", ""])
async def test_incomplete_reply_survives_resume_and_can_be_continued(
    tmp_path, reason, text
):
    profile = AgentProfile(llm={"model": "fake"}, workspace=str(tmp_path), tools=[])
    llm = FakeLLMClient([ScriptedTurn(text=text, finish_reason=reason)])
    with SessionStore(tmp_path / "sessions.db") as store:
        async with Agent.from_profile(profile, llm=llm, session_store=store) as agent:
            events = [e async for e in agent.run("go")]
            sid = agent.tool_ctx.session_id
        assert isinstance(events[-1], Error)
        assert events[-1].finish_reason == reason
        assert events[-1].partial_content == text
        assert isinstance(events[-2], TurnStatsReported)
        assert not any(isinstance(e, Final | StreamRetry) for e in events)
        assert len(llm.calls) == 1
        saved = store.messages(sid)[-1]
        assert saved.content == text and saved.finish_reason == reason
        assert saved.finish_detail == events[-1].message
        assert not saved.tool_calls
        assert "finish_reason" not in saved.to_openai()
        assert "finish_reason" not in saved.to_anthropic()
        assert "finish_detail" not in saved.to_openai()
        assert "finish_detail" not in saved.to_anthropic()
        continuation = FakeLLMClient([ScriptedTurn(text="finished")])
        async with Agent.from_profile(
            profile, llm=continuation, session_store=store, session_id=sid
        ) as resumed:
            events = [e async for e in resumed.run("Continue")]
        assert isinstance(events[-1], Final)
        assert any(m.finish_reason == reason for m in continuation.calls[0])


async def test_cut_off_batch_runs_no_tools_even_with_complete_json(tmp_path):
    registry = ToolRegistry()
    ran = []

    @tool(registry=registry)
    async def probe(args: NoArgs, ctx: ToolContext) -> str:
        ran.append(True)
        return "executed"

    llm = FakeLLMClient(
        [
            ScriptedTurn(
                text="I will run these",
                finish_reason="length",
                tool_calls=[
                    ToolCall(id="complete", name="probe"),
                    ToolCall(id="partial", name="probe", invalid_arguments=""),
                ],
            )
        ]
    )
    async with Agent(
        llm, registry, ToolContext(workspace=tmp_path), system_prompt="test"
    ) as agent:
        events = [e async for e in agent.run("go")]
        assert not agent.memory.messages[-1].tool_calls
    assert not ran
    assert isinstance(events[-1], Error)
    assert "2 tool call(s) were not run" in events[-1].message
    assert not any(isinstance(e, ToolCallStarted | ToolResultEvent) for e in events)


class NoArgs(BaseModel):
    pass


async def test_stop_during_incomplete_turn_end_keeps_committed_outcome(tmp_path):
    from lingcore.plugins.hooks import HookRunner

    entered = asyncio.Event()

    class SlowHooks(HookRunner):
        async def turn_end(self, event):
            entered.set()
            await asyncio.Event().wait()

    ctx = ToolContext(workspace=tmp_path)
    agent = Agent(
        FakeLLMClient([ScriptedTurn(text="partial", finish_reason="length")]),
        ToolRegistry(),
        ctx,
        system_prompt="test",
        hooks=SlowHooks([], ctx),
    )

    async def drain():
        return [e async for e in agent.run("go")]

    task = asyncio.create_task(drain())
    await entered.wait()
    assert agent.cancel_turn()
    with pytest.raises(asyncio.CancelledError):
        await task
    terminal = agent.finalize_cancelled_turn()
    assert isinstance(terminal, Error) and terminal.finish_reason == "length"
    assert agent.memory.messages[-1] == Message.assistant(
        "partial", finish_reason="length", finish_detail=terminal.message
    )
    assert terminal.partial_content == "partial"
    await agent.aclose()


@pytest.mark.parametrize("guarded", ["redacted partial", ""])
async def test_incomplete_reply_is_guarded_before_commit_and_resume(tmp_path, guarded):
    from lingcore.guardrails import NoopGuardrail

    class Redactor(NoopGuardrail):
        async def post_output(self, text):
            assert text == "SECRET"
            return guarded

    profile = AgentProfile(llm={"model": "fake"}, workspace=str(tmp_path), tools=[])
    with SessionStore(tmp_path / "sessions.db") as store:
        async with Agent.from_profile(
            profile,
            llm=FakeLLMClient([ScriptedTurn(text="SECRET", finish_reason="length")]),
            session_store=store,
        ) as agent:
            agent.guardrail = Redactor()
            events = [event async for event in agent.run("go")]
            sid = agent.tool_ctx.session_id
        assert events[-1].partial_content == guarded
        saved = store.messages(sid)[-1]
        assert saved.content == guarded
        assert "SECRET" not in saved.model_dump_json()
        async with Agent.from_profile(
            profile, llm=FakeLLMClient([]), session_store=store, session_id=sid
        ) as resumed:
            assert resumed.memory.messages[-1].content == guarded
