"""Tests for LLMClient streaming + tool-call delta reassembly (M1)."""

from __future__ import annotations

import pytest

from lingcore.llm import LLMClient
from tests.fakes import (
    _Choice,
    _Delta,
    _Event,
    _Fn,
    _ToolCallDelta,
    make_openai_stream,
)


async def _collect(client: LLMClient, monkeypatch, events):
    """Patch the client's stream opener to return our fabricated events."""

    async def fake_open(messages, tools):
        return make_openai_stream(events)

    monkeypatch.setattr(client, "_open_stream", fake_open)
    return [chunk async for chunk in client.stream(messages=[])]


@pytest.fixture
def client():
    return LLMClient(model="x", api_key="sk-test", base_url="http://localhost/v1")


async def test_text_streaming(client, monkeypatch):
    events = [
        _Event([_Choice(_Delta(content="Hel"))]),
        _Event([_Choice(_Delta(content="lo"))]),
        _Event([_Choice(_Delta(), finish_reason="stop")]),
    ]
    chunks = await _collect(client, monkeypatch, events)
    text = "".join(c.text_delta for c in chunks)
    assert text == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].tool_calls is None


async def test_tool_call_fragments_reassembled(client, monkeypatch):
    # A single tool call whose JSON arguments arrive split across chunks.
    events = [
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, id="call_1", function=_Fn(name="read_file")),
        ]))]),
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, function=_Fn(arguments='{"pa')),
        ]))]),
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, function=_Fn(arguments='th": "a.txt"}')),
        ]))]),
        _Event([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    chunks = await _collect(client, monkeypatch, events)
    final = chunks[-1]
    assert final.finish_reason == "tool_calls"
    assert final.tool_calls is not None
    assert len(final.tool_calls) == 1
    call = final.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.txt"}


async def test_parallel_tool_calls_kept_separate(client, monkeypatch):
    events = [
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, id="c0", function=_Fn(name="read_file", arguments='{"path":"a"}')),
            _ToolCallDelta(index=1, id="c1", function=_Fn(name="list_dir", arguments='{"path":"."}')),
        ]))]),
        _Event([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    chunks = await _collect(client, monkeypatch, events)
    calls = chunks[-1].tool_calls
    assert [c.name for c in calls] == ["read_file", "list_dir"]
    assert calls[0].arguments == {"path": "a"}
    assert calls[1].arguments == {"path": "."}


async def test_malformed_args_become_empty_dict(client, monkeypatch):
    events = [
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, id="c0", function=_Fn(name="x", arguments="{not json")),
        ]))]),
        _Event([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    chunks = await _collect(client, monkeypatch, events)
    assert chunks[-1].tool_calls[0].arguments == {}
