"""Agent tests: retries."""

from __future__ import annotations

import lingcore.tools.builtin  # noqa: F401  (registers builtins)
from lingcore.events import (
    Error,
    Final,
    StreamRetry,
)
from lingcore.llm import LLMChunk
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure
from tests.support.agent import _agent, _drain
from tests.support.agent import no_backoff as no_backoff
from tests.support.agent import workspace as workspace


async def test_midstream_failure_recovers(workspace, no_backoff):
    # The reply dies after "par" already streamed; the loop discards the
    # partial turn, announces the retry, and re-requests successfully.
    llm = FakeLLMClient(
        [
            StreamFailure(text="par", reason="stream interrupted: connection lost"),
            ScriptedTurn(text="Hi there!"),
        ]
    )
    agent = _agent(llm, workspace)
    events = await _drain(agent, "hello")

    retries = [e for e in events if isinstance(e, StreamRetry)]
    assert len(retries) == 1
    assert retries[0].attempt == 1
    assert retries[0].max_attempts == agent.stream_retries
    assert retries[0].discarded_chars == 3
    assert "connection lost" in retries[0].reason
    assert isinstance(events[-1], Final)
    assert events[-1].content == "Hi there!"

    # The partial attempt was never committed: exactly one assistant message,
    # carrying only the successful reply.
    assistants = [m for m in agent.memory.messages if m.role == "assistant"]
    assert [m.content for m in assistants] == ["Hi there!"]
    # The re-request sent the identical conversation (nothing was appended
    # between attempts).
    assert len(llm.calls) == 2
    assert [(m.role, m.content) for m in llm.calls[0]] == [
        (m.role, m.content) for m in llm.calls[1]
    ]


async def test_midstream_failure_before_first_token_recovers(workspace, no_backoff):
    # Stream opened but died before emitting anything: recovery is invisible
    # apart from the StreamRetry event (nothing to discard).
    llm = FakeLLMClient([StreamFailure(), ScriptedTurn(text="ok")])
    agent = _agent(llm, workspace)
    events = await _drain(agent, "hello")
    retries = [e for e in events if isinstance(e, StreamRetry)]
    assert len(retries) == 1
    assert retries[0].discarded_chars == 0
    assert isinstance(events[-1], Final) and events[-1].content == "ok"


async def test_midstream_retries_exhausted_ends_with_error(workspace, no_backoff):
    llm = FakeLLMClient([StreamFailure(), StreamFailure(), StreamFailure()])
    agent = _agent(llm, workspace, stream_retries=2)
    events = await _drain(agent, "hello")

    assert len([e for e in events if isinstance(e, StreamRetry)]) == 2
    assert isinstance(events[-1], Error)
    assert "after 2 retries" in events[-1].message
    assert not any(isinstance(e, Final) for e in events)
    # The failed turn left no assistant message behind...
    assert [m.role for m in agent.memory.messages] == ["user"]
    assert agent._turn_index == 0

    # ...and the loop is still alive: the next turn completes normally.
    llm._turns.append(ScriptedTurn(text="recovered"))
    events2 = await _drain(agent, "again")
    assert isinstance(events2[-1], Final) and events2[-1].content == "recovered"
    assert agent._turn_index == 1


async def test_stream_retries_zero_disables_recovery(workspace, no_backoff):
    llm = FakeLLMClient([StreamFailure()])
    agent = _agent(llm, workspace, stream_retries=0)
    events = await _drain(agent, "hello")
    assert not any(isinstance(e, StreamRetry) for e in events)
    assert isinstance(events[-1], Error)


async def test_nonretryable_stream_error_fails_fast(workspace, no_backoff):
    llm = FakeLLMClient(
        [
            StreamFailure(reason="request failed: BadRequestError", retryable=False),
            ScriptedTurn(text="never reached"),
        ]
    )
    agent = _agent(llm, workspace, stream_retries=5)
    events = await _drain(agent, "hello")
    assert not any(isinstance(e, StreamRetry) for e in events)
    assert isinstance(events[-1], Error)
    assert "BadRequestError" in events[-1].message
    assert len(llm.calls) == 1  # no re-request for a non-retryable failure


async def test_foreign_llm_exception_contained(workspace):
    # A duck-typed backend that raises something other than LLMStreamError
    # must not crash the session: the turn ends with an Error event.
    class BoomClient:
        async def stream(self, messages, tools=None):
            yield LLMChunk(text_delta="x")
            raise ValueError("boom")

    agent = _agent(BoomClient(), workspace)
    events = await _drain(agent, "hello")
    assert isinstance(events[-1], Error)
    assert "ValueError" in events[-1].message
    assert agent._turn_index == 0
    # Still usable afterwards.
    agent.llm = FakeLLMClient([ScriptedTurn(text="fine")])
    events2 = await _drain(agent, "again")
    assert isinstance(events2[-1], Final) and events2[-1].content == "fine"
