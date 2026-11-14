"""Authorization snapshots, private terminal tools, and live cancellable progress."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

import pytest
from pydantic import BaseModel

import lingcore.tools.builtin  # noqa: F401
from lingcore._agent.dispatch import dispatch_authorization
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.events import Final, ToolProgress, ToolResultEvent, TurnCancelled
from lingcore.message import ToolCall
from lingcore.skills import Skill, SkillState
from lingcore.tools import REGISTRY, Tool, ToolContext, ToolOutput, ToolRegistry
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.delegation import EmptyArgs, build_agent


class ResultArgs(BaseModel):
    answer: int


async def submit(args, ctx):
    return ToolOutput(args.model_dump_json(), end_turn=True)


SUBMIT = Tool("test_submit", "Return a result", ResultArgs, submit)


@pytest.mark.parametrize("terminal", [False, True])
async def test_extra_tool_is_private_and_only_allowlisted_tool_ends_turn(
    tmp_path, terminal
):
    before = REGISTRY.names()
    agent, raw = build_agent(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="s", name=SUBMIT.name, arguments={"answer": 3})]
            ),
            ScriptedTurn(text="normal final"),
        ],
        extra_tools=(SUBMIT,),
        terminal_tools=frozenset({SUBMIT.name}) if terminal else frozenset(),
    )
    async with agent:
        events = [e async for e in agent.run("go")]
    assert len(raw.calls) == (1 if terminal else 2)
    assert events[-1] == Final('{"answer":3}' if terminal else "normal final")
    assert REGISTRY.names() == before
    result = next(e.result for e in events if isinstance(e, ToolResultEvent))
    assert "_end_turn" not in result.model_dump()
    assert [m.role for m in agent.memory.messages][:3] == ["user", "assistant", "tool"]


async def test_invalid_terminal_submission_retries_then_finishes_entire_batch(tmp_path):
    finished = []

    async def sibling(args, ctx):
        await asyncio.sleep(0)
        finished.append(True)
        return "sibling complete"

    sibling_tool = Tool("sibling", "sibling", EmptyArgs, sibling)
    agent, raw = build_agent(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall(id="bad", name=SUBMIT.name)]),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="good", name=SUBMIT.name, arguments={"answer": 7}),
                    ToolCall(id="sibling", name="sibling"),
                ]
            ),
        ],
        extra_tools=(SUBMIT, sibling_tool),
        terminal_tools=frozenset({SUBMIT.name}),
    )
    async with agent:
        events = [e async for e in agent.run("go")]
    results = [e.result for e in events if isinstance(e, ToolResultEvent)]
    assert [r.ok for r in results] == [False, True, True]
    assert len(raw.calls) == 2 and finished == [True]
    assert events[-1] == Final('{"answer":7}')


def test_terminal_tools_must_be_injected_and_cannot_override_ceiling(tmp_path):
    profile = AgentProfile.model_validate(
        {"llm": {"model": "fake"}, "tools": ["read_file"]}
    )
    with pytest.raises(ConfigError, match="extra_tools"):
        Agent.from_profile(
            profile, llm=FakeLLMClient([]), terminal_tools=frozenset({"read_file"})
        )
    collision = Tool("read_file", "collision", ResultArgs, submit)
    with pytest.raises(ConfigError, match="unique names"):
        Agent.from_profile(profile, llm=FakeLLMClient([]), extra_tools=(collision,))
    with pytest.raises(ConfigError, match="unique names"):
        build_agent(tmp_path, [], extra_tools=(SUBMIT, SUBMIT))


@pytest.mark.parametrize("parallel", [False, True])
async def test_dispatch_context_keeps_batch_snapshot_through_skill_activation(
    tmp_path, parallel
):
    state = SkillState(
        skills={
            "reader": Skill(
                name="reader",
                description="read",
                instructions="read",
                requested_tools=("hidden",),
            )
        },
        profile_tools=frozenset({"activate", "inspect", "hidden"}),
    )
    activated = asyncio.Event()
    snapshots = []

    async def activate(args, ctx):
        state.active.append("reader")
        activated.set()
        return "activated"

    async def inspect(args, ctx):
        await activated.wait()
        snapshots.append(dispatch_authorization.get())
        return "inspected"

    registry = ToolRegistry()
    for name, fn in [("activate", activate), ("inspect", inspect), ("hidden", inspect)]:
        registry.register(Tool(name, name, EmptyArgs, fn))
    agent = Agent(
        FakeLLMClient([]),
        registry,
        ToolContext(tmp_path),
        system_prompt="x",
        initial_tools=frozenset({"activate", "inspect"}),
        skill_state=state,
        parallel_tools=parallel,
    )
    async with agent:
        results = await agent._dispatch(
            [
                ToolCall(id="a", name="activate"),
                ToolCall(id="i", name="inspect"),
                ToolCall(id="h", name="hidden"),
            ]
        )
        assert snapshots == [frozenset({"activate", "inspect"})]
        assert not results[-1].ok and dispatch_authorization.get() is None
        await agent._dispatch([ToolCall(id="next", name="inspect")])
        assert "hidden" in snapshots[-1]


@pytest.mark.parametrize("stop", [False, True])
async def test_progress_arrives_before_result_and_stop_reaps_every_tool(tmp_path, stop):
    release = asyncio.Event()
    started = set()
    reaped = set()

    async def long_tool(args, ctx):
        started.add(args.answer)
        data = {"step": args.answer}
        ctx.progress(data)
        data["step"] = -1
        try:
            await release.wait()
            return "done"
        finally:
            await asyncio.sleep(0)
            reaped.add(args.answer)

    tool = Tool("long", "long", ResultArgs, long_tool)
    agent, raw = build_agent(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id=str(i), name="long", arguments={"answer": i})
                    for i in range(10)
                ]
            ),
            ScriptedTurn(text="finished"),
        ],
        extra_tools=(tool,),
    )
    seen = []
    progress_ready = asyncio.Event()

    async def consume():
        async with aclosing(agent.run("go")) as stream:
            async for event in stream:
                seen.append(event)
                if isinstance(event, ToolProgress):
                    progress_ready.set()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(progress_ready.wait(), 2)
        assert started == set(range(10)) and not reaped
        assert not any(isinstance(e, ToolResultEvent) for e in seen)
        if stop:
            assert agent.cancel_turn()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert isinstance(agent.finalize_cancelled_turn(), TurnCancelled)
        else:
            release.set()
            await asyncio.wait_for(task, 2)
            assert seen[-1] == Final("finished")
        assert reaped == set(range(10))
        assert all(
            e.payload["step"] >= 0 and e.call_id == str(e.payload["step"])
            for e in seen
            if isinstance(e, ToolProgress)
        )
        assert not any("step" in m.content for m in agent.memory.messages)
        assert dispatch_authorization.get() is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()


async def test_abandoning_stream_at_progress_reaps_batch(tmp_path):
    reaped = asyncio.Event()

    async def long_tool(args, ctx):
        try:
            ctx.progress({"waiting": True})
            await asyncio.Event().wait()
        finally:
            reaped.set()

    agent, _ = build_agent(
        tmp_path,
        [ScriptedTurn(tool_calls=[ToolCall(id="p", name="long")])],
        extra_tools=(Tool("long", "long", EmptyArgs, long_tool),),
    )
    async with agent:
        async with aclosing(agent.run("go")) as stream:
            async for event in stream:
                if isinstance(event, ToolProgress):
                    break
        assert reaped.is_set() and not agent.turn_pending_finalization
