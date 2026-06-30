"""Tests for the agent loop and memory (M3).

Driven entirely by a scripted FakeLLMClient — no network, no API cost.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from pydantic import BaseModel

from lingcore.agent import Agent
from lingcore.composer import StaticComposer
from lingcore.events import (
    Error,
    Final,
    StreamRetry,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.llm import LLMChunk
from lingcore.memory import WindowMemory
from lingcore.message import Attachment, Message, ToolCall, UserInput
from lingcore.modality import MediaAdapter
from lingcore.tools import ToolContext, ToolOutput, ToolRegistry, tool
from lingcore.tools.builtin.fs import read_file
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure
from tests.test_modality import make_pdf


class _EmptyArgs(BaseModel):
    pass


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
    assert results[0].result.content == "1\thello"  # read_file is line-numbered
    assert isinstance(events[-1], Final)

    # The model saw the tool result on its second turn.
    second_turn_msgs = llm.calls[1]
    assert any(m.role == "tool" and m.content == "1\thello" for m in second_turn_msgs)


async def test_tool_output_attachments_are_hoisted(workspace):
    local_reg = ToolRegistry()

    @tool(name="media_tool", registry=local_reg)
    async def media_tool(args: _EmptyArgs, ctx: ToolContext) -> ToolOutput:
        data = base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode("ascii")
        att = Attachment(kind="image", media_type="image/png", data=data, name="pic.png")
        return ToolOutput(text="attached pic", attachments=[att])

    call = ToolCall(id="c1", name="media_tool", arguments={})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = Agent(
        llm=llm,
        tools=local_reg,
        tool_ctx=ToolContext(workspace=workspace),
        composer=StaticComposer("sys"),
        memory=WindowMemory(model="gpt-4o"),
    )
    events = await _drain(agent, "test")
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert tool_results[0].result.content == "attached pic"
    assert len(tool_results[0].result.attachments) == 1

    user_msgs = [m for m in agent.memory.messages if m.role == "user"]
    assert len(user_msgs) == 2
    assert user_msgs[1].name == "media"
    assert len(user_msgs[1].attachments) == 1
    assert user_msgs[1].attachments[0].name == "pic.png"

    wire = llm.calls[1]
    last_user = [m for m in wire if m.role == "user"][-1]
    assert last_user.attachments


async def test_hoist_caps_aggregate_attachments(workspace):
    # Each result passes its own per-list caps (3 <= 8), but the round's
    # aggregate (9) exceeds MAX_ATTACHMENTS. The loop must cap the hoist and
    # finish the turn instead of letting the Message validator raise
    # (invariant 5).
    from lingcore.media_types import MAX_ATTACHMENTS

    local_reg = ToolRegistry()
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode("ascii")

    @tool(name="three_pics", registry=local_reg)
    async def three_pics(args: _EmptyArgs, ctx: ToolContext) -> ToolOutput:
        atts = [
            Attachment(kind="image", media_type="image/png", data=png, name=f"p{i}.png")
            for i in range(3)
        ]
        return ToolOutput(text="3 pics", attachments=atts)

    calls = [ToolCall(id=f"c{i}", name="three_pics", arguments={}) for i in range(3)]
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=calls, finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = Agent(
        llm=llm,
        tools=local_reg,
        tool_ctx=ToolContext(workspace=workspace),
        composer=StaticComposer("sys"),
        memory=WindowMemory(model="gpt-4o"),
    )
    events = await _drain(agent, "go")
    assert isinstance(events[-1], Final)  # the turn completed

    hoist = [m for m in agent.memory.messages if m.name == "media"][0]
    assert len(hoist.attachments) == MAX_ATTACHMENTS
    assert "1 attachment(s) dropped" in hoist.content


async def test_hoist_caps_aggregate_total_bytes(workspace, monkeypatch):
    # Two results within their own byte caps, together over the message total:
    # the second attachment is dropped, the turn still completes.
    import lingcore.agent as agent_mod

    local_reg = ToolRegistry()
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 64).decode("ascii")

    @tool(name="one_pic", registry=local_reg)
    async def one_pic(args: _EmptyArgs, ctx: ToolContext) -> ToolOutput:
        att = Attachment(kind="image", media_type="image/png", data=png, name="p.png")
        return ToolOutput(text="pic", attachments=[att])

    # Shrink the loop's total budget so one tiny PNG fits and the second does
    # not (the real 20MB constant would need huge fixtures).
    monkeypatch.setattr(agent_mod, "TOTAL_ATTACHMENT_MAX_BYTES", 100)

    calls = [ToolCall(id=f"c{i}", name="one_pic", arguments={}) for i in range(2)]
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=calls, finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = Agent(
        llm=llm,
        tools=local_reg,
        tool_ctx=ToolContext(workspace=workspace),
        composer=StaticComposer("sys"),
        memory=WindowMemory(model="gpt-4o"),
    )
    events = await _drain(agent, "go")
    assert isinstance(events[-1], Final)
    hoist = [m for m in agent.memory.messages if m.name == "media"][0]
    assert len(hoist.attachments) == 1
    assert "dropped" in hoist.content


async def test_modality_fallback_prepares_user_attachments(workspace):
    # A text-only model: the PDF the user attached is converted before the
    # message is committed, so every render of this turn sees the text.
    adapter = MediaAdapter(native=frozenset())
    llm = FakeLLMClient([ScriptedTurn(text="ok")])
    agent = _agent(llm, workspace, media_adapter=adapter)
    att = Attachment(
        kind="file",
        media_type="application/pdf",
        data=base64.b64encode(make_pdf("hidden rent figure")).decode("ascii"),
        name="d.pdf",
    )
    events = await _drain(agent, UserInput(text="read this", attachments=[att]))
    assert isinstance(events[-1], Final)
    user = [m for m in agent.memory.messages if m.role == "user"][0]
    assert "hidden rent figure" in user.attachments[0].fallback_text
    # The model-facing copy carried it too (FakeLLM records Message objects).
    assert llm.calls[0][-1].attachments[0].fallback_text


async def test_modality_fallback_prepares_hoisted_tool_media(workspace):
    adapter = MediaAdapter(native=frozenset())
    local_reg = ToolRegistry()
    pdf_b64 = base64.b64encode(make_pdf("quarterly numbers")).decode("ascii")

    @tool(name="fetch_doc", registry=local_reg)
    async def fetch_doc(args: _EmptyArgs, ctx: ToolContext) -> ToolOutput:
        att = Attachment(
            kind="file", media_type="application/pdf", data=pdf_b64, name="q.pdf"
        )
        return ToolOutput(text="attached q.pdf", attachments=[att])

    call = ToolCall(id="c1", name="fetch_doc", arguments={})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    agent = Agent(
        llm=llm,
        tools=local_reg,
        tool_ctx=ToolContext(workspace=workspace),
        composer=StaticComposer("sys"),
        memory=WindowMemory(model="gpt-4o"),
        media_adapter=adapter,
    )
    events = await _drain(agent, "get the doc")
    assert isinstance(events[-1], Final)
    hoist = [m for m in agent.memory.messages if m.name == "media"][0]
    assert "quarterly numbers" in hoist.attachments[0].fallback_text


async def test_no_adapter_leaves_attachments_untouched(workspace):
    llm = FakeLLMClient([ScriptedTurn(text="ok")])
    agent = _agent(llm, workspace)  # media_adapter defaults to None
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode("ascii")
    att = Attachment(kind="image", media_type="image/png", data=png, name="p.png")
    await _drain(agent, UserInput(text="look", attachments=[att]))
    user = [m for m in agent.memory.messages if m.role == "user"][0]
    assert user.attachments[0] is att
    assert user.attachments[0].fallback_text is None


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
    assert {r.result.content for r in results} == {"1\thello", "1\tworld"}


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


# --- mid-stream failure recovery -------------------------------------------


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr("lingcore.agent._backoff_seconds", lambda attempt: 0.0)


async def test_midstream_failure_recovers(workspace, no_backoff):
    # The reply dies after "par" already streamed; the loop discards the
    # partial turn, announces the retry, and re-requests successfully.
    llm = FakeLLMClient([
        StreamFailure(text="par", reason="stream interrupted: connection lost"),
        ScriptedTurn(text="Hi there!"),
    ])
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
    assert [
        (m.role, m.content) for m in llm.calls[0]
    ] == [(m.role, m.content) for m in llm.calls[1]]


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

    # ...and the loop is still alive: the next turn completes normally.
    llm._turns.append(ScriptedTurn(text="recovered"))
    events2 = await _drain(agent, "again")
    assert isinstance(events2[-1], Final) and events2[-1].content == "recovered"


async def test_stream_retries_zero_disables_recovery(workspace, no_backoff):
    llm = FakeLLMClient([StreamFailure()])
    agent = _agent(llm, workspace, stream_retries=0)
    events = await _drain(agent, "hello")
    assert not any(isinstance(e, StreamRetry) for e in events)
    assert isinstance(events[-1], Error)


async def test_nonretryable_stream_error_fails_fast(workspace, no_backoff):
    llm = FakeLLMClient([
        StreamFailure(reason="request failed: BadRequestError", retryable=False),
        ScriptedTurn(text="never reached"),
    ])
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
    # Still usable afterwards.
    agent.llm = FakeLLMClient([ScriptedTurn(text="fine")])
    events2 = await _drain(agent, "again")
    assert isinstance(events2[-1], Final) and events2[-1].content == "fine"
