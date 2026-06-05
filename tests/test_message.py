"""Tests for the message data model (M1)."""

from __future__ import annotations

from lingcore.message import Conversation, Message, ToolCall, ToolResult


def test_user_and_system_to_openai():
    assert Message.user("hi").to_openai() == {"role": "user", "content": "hi"}
    assert Message.system("sys").to_openai() == {"role": "system", "content": "sys"}


def test_assistant_with_tool_calls_to_openai():
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})
    msg = Message.assistant(content="", tool_calls=[call])
    wire = msg.to_openai()
    assert wire["role"] == "assistant"
    assert wire["tool_calls"][0]["id"] == "c1"
    assert wire["tool_calls"][0]["function"]["name"] == "read_file"
    # arguments must be a JSON *string* on the wire
    assert wire["tool_calls"][0]["function"]["arguments"] == '{"path": "a.txt"}'


def test_assistant_without_tool_calls_omits_key():
    assert "tool_calls" not in Message.assistant(content="hello").to_openai()


def test_tool_result_round_trip():
    result = ToolResult(call_id="c1", name="read_file", content="file body", ok=True)
    msg = Message.from_tool_result(result)
    wire = msg.to_openai()
    assert wire == {"role": "tool", "tool_call_id": "c1", "content": "file body"}


def test_conversation_to_openai_order():
    conv = Conversation()
    conv.add(Message.system("s"))
    conv.add(Message.user("u"))
    wire = conv.to_openai()
    assert [m["role"] for m in wire] == ["system", "user"]
    assert len(conv) == 2
