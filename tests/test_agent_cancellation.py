"""Agent tests: cancellation."""

from __future__ import annotations

import asyncio

import pytest

import lingcore.tools.builtin  # noqa: F401  (registers builtins)
from lingcore.events import (
    Error,
    Final,
    TextDelta,
    TurnCancelled,
)
from lingcore.llm import LLMChunk
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.agent import _agent, _drain
from tests.support.agent import no_backoff as no_backoff
from tests.support.agent import workspace as workspace


async def test_guardrail_exception_releases_turn_lease(workspace):
    class FailOnceGuardrail:
        def __init__(self):
            self.failed = False

        async def pre_input(self, text):
            if not self.failed:
                self.failed = True
                raise RuntimeError("guardrail broke")
            return text

        async def post_output(self, text):
            return text

    agent = _agent(
        FakeLLMClient([ScriptedTurn(text="recovered")]),
        workspace,
        guardrail=FailOnceGuardrail(),
    )

    failed = await _drain(agent, "hello")
    assert len(failed) == 1
    assert isinstance(failed[0], Error)
    assert "RuntimeError: guardrail broke" in failed[0].message
    assert agent.memory.messages == []
    assert agent._turn_checkpoint is None
    assert agent.cancel_turn() is False

    recovered = await _drain(agent, "again")
    assert isinstance(recovered[-1], Final)
    assert recovered[-1].content == "recovered"


async def test_aclose_rolls_back_partial_turn_and_releases_lease(workspace):
    agent = _agent(FakeLLMClient([ScriptedTurn(text="partial reply")]), workspace)
    stream = agent.run("please answer")

    first = await anext(stream)
    assert isinstance(first, TextDelta)
    assert agent._turn_index == 1
    await stream.aclose()

    assert agent._turn_checkpoint is None
    assert agent._turn_index == 0
    assert [(m.role, m.content) for m in agent.memory.messages] == [
        ("user", "please answer")
    ]
    assert agent.cancel_turn() is False

    agent.llm = FakeLLMClient([ScriptedTurn(text="recovered")])
    recovered = await _drain(agent, "try again")
    assert isinstance(recovered[-1], Final)
    assert recovered[-1].content == "recovered"


async def test_stale_terminal_stream_cannot_roll_back_new_turn(workspace):
    agent = _agent(
        FakeLLMClient(
            [ScriptedTurn(text="first reply"), ScriptedTurn(text="second reply")]
        ),
        workspace,
    )
    old_stream = agent.run("first question")
    while not isinstance(await anext(old_stream), Final):
        pass

    # The terminal event releases the public turn slot, but its generator is
    # still suspended at the yield until the consumer exhausts or closes it.
    new_stream = agent.run("second question")
    while not isinstance(await anext(new_stream), TextDelta):
        pass
    checkpoint = agent._turn_checkpoint
    lease = agent._turn_lease
    turn_index = agent._turn_index

    await old_stream.aclose()

    assert agent._turn_checkpoint is checkpoint
    assert agent._turn_lease is lease
    assert agent._turn_index == turn_index
    remaining = [event async for event in new_stream]
    assert isinstance(remaining[-1], Final)
    assert [message.content for message in agent.memory.messages] == [
        "first question",
        "first reply",
        "second question",
        "second reply",
    ]


async def test_explicit_cancellation_rolls_back_partial_turn(workspace):
    class BlockingLLM:
        def __init__(self):
            self.waiting = asyncio.Event()

        async def stream(self, messages, tools=None):
            yield LLMChunk(text_delta="partial")
            self.waiting.set()
            await asyncio.Event().wait()

    llm = BlockingLLM()
    agent = _agent(llm, workspace)
    streamed = []

    async def drive():
        async for event in agent.run("please answer"):
            streamed.append(event)

    task = asyncio.create_task(drive())
    await llm.waiting.wait()
    assert any(isinstance(event, TextDelta) for event in streamed)
    with pytest.raises(RuntimeError, match="before its driver task stops"):
        agent.finalize_cancelled_turn()
    assert agent.turn_pending_finalization is False
    assert agent.cancel_turn() is True
    assert agent.turn_pending_finalization is False
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.turn_pending_finalization is True

    # A stopped driver is not a free turn slot until rollback has repaired its
    # checkpoint and durable branch.
    blocked = await _drain(agent, "too soon")
    assert len(blocked) == 1
    assert isinstance(blocked[0], Error)
    assert "another turn" in blocked[0].message

    cancelled = agent.finalize_cancelled_turn()
    assert isinstance(cancelled, TurnCancelled)
    assert agent.turn_pending_finalization is False
    assert cancelled.reason == "stopped by user"
    assert agent._turn_index == 0
    # Partial assistant output is void, but the user's submitted prompt remains
    # valid context for a follow-up or an edit.
    assert [(m.role, m.content) for m in agent.memory.messages] == [
        ("user", "please answer")
    ]
    assert agent.cancel_turn() is False

    agent.llm = FakeLLMClient([ScriptedTurn(text="recovered")])
    events = await _drain(agent, "try again")
    assert isinstance(events[-1], Final)
    assert agent.cancel_turn() is False


async def test_cancelled_driver_can_finalize_in_its_own_handler(workspace):
    class BlockingLLM:
        def __init__(self):
            self.waiting = asyncio.Event()

        async def stream(self, messages, tools=None):
            self.waiting.set()
            await asyncio.Event().wait()
            yield LLMChunk(finish_reason="stop")

    llm = BlockingLLM()
    agent = _agent(llm, workspace)

    async def drive():
        try:
            async for _ in agent.run("please answer"):
                pass
        except asyncio.CancelledError:
            return agent.finalize_cancelled_turn(reason="interrupted")

    task = asyncio.create_task(drive())
    await llm.waiting.wait()
    assert agent.cancel_turn() is True

    event = await task
    assert isinstance(event, TurnCancelled)
    assert event.reason == "interrupted"
    assert agent._turn_checkpoint is None
    assert [(message.role, message.content) for message in agent.memory.messages] == [
        ("user", "please answer")
    ]


async def test_cancel_tracks_task_driving_each_generator_step(workspace):
    class YieldThenBlockLLM:
        def __init__(self):
            self.blocking = asyncio.Event()

        async def stream(self, messages, tools=None):
            yield LLMChunk(text_delta="partial")
            self.blocking.set()
            await asyncio.Event().wait()

    llm = YieldThenBlockLLM()
    agent = _agent(llm, workspace)
    stream = agent.run("please answer")

    # Each anext is intentionally driven by a different short-lived task.
    first_task = asyncio.create_task(anext(stream))
    assert isinstance(await first_task, TextDelta)
    assert first_task.done()
    with pytest.raises(RuntimeError, match="no cancelled turn"):
        agent.finalize_cancelled_turn()
    second_task = asyncio.create_task(anext(stream))
    await llm.blocking.wait()

    assert agent.cancel_turn() is True
    with pytest.raises(asyncio.CancelledError):
        await second_task
    event = agent.finalize_cancelled_turn()

    assert isinstance(event, TurnCancelled)
    assert agent._turn_checkpoint is None
    assert [(message.role, message.content) for message in agent.memory.messages] == [
        ("user", "please answer")
    ]


async def test_cancel_during_frontend_work_preserves_finalize_handshake(workspace):
    frontend_busy = asyncio.Event()

    async def drive(agent):
        async for event in agent.run("please answer"):
            if isinstance(event, TextDelta):
                frontend_busy.set()
                await asyncio.Event().wait()

    agent = _agent(
        FakeLLMClient([ScriptedTurn(text="partial reply")]),
        workspace,
    )
    task = asyncio.create_task(drive(agent))
    await frontend_busy.wait()

    assert agent.cancel_turn() is True
    with pytest.raises(asyncio.CancelledError):
        await task
    # Give the loop a chance to finalize the abandoned async generator before
    # the frontend performs its explicit second-stage repair.
    await asyncio.sleep(0)
    event = agent.finalize_cancelled_turn()

    assert isinstance(event, TurnCancelled)
    assert agent._turn_checkpoint is None
    assert [(message.role, message.content) for message in agent.memory.messages] == [
        ("user", "please answer")
    ]
