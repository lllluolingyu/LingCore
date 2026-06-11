"""Tests for the message data model (M1)."""

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from lingcore.message import (
    Attachment,
    Conversation,
    Message,
    ToolCall,
    ToolResult,
    UserInput,
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


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


def test_user_attachments_to_openai_parts():
    img_data = _b64(b"\x89PNG\r\n\x1a\nrest")
    pdf_data = _b64(b"%PDF-1.4\n")
    img = Attachment(kind="image", media_type="image/png", data=img_data, name="pic.png")
    pdf = Attachment(kind="file", media_type="application/pdf", data=pdf_data, name="doc.pdf")
    wire = Message.user("describe", attachments=[img, pdf]).to_openai()
    assert wire == {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data}"},
            },
            {
                "type": "file",
                "file": {
                    "filename": "doc.pdf",
                    "file_data": f"data:application/pdf;base64,{pdf_data}",
                },
            },
        ],
    }


def test_user_attachment_without_text_skips_text_part():
    data = _b64(b"\xff\xd8\xffrest")
    img = Attachment(kind="image", media_type="image/jpeg", data=data, name="p.jpg")
    wire = Message.user("", attachments=[img]).to_openai()
    assert wire["content"] == [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
    ]


def test_non_user_attachment_does_not_change_wire_shape():
    img = Attachment(
        kind="image",
        media_type="image/png",
        data=_b64(b"\x89PNG\r\n\x1a\nrest"),
        name="p.png",
    )
    msg = Message(role="assistant", content="done", attachments=[img])
    assert msg.to_openai() == {"role": "assistant", "content": "done"}


def test_user_input_defaults():
    assert UserInput().text == ""
    assert UserInput().attachments == []


def test_attachment_rejects_invalid_payloads():
    with pytest.raises(ValidationError, match="invalid attachment base64"):
        Attachment(kind="image", media_type="image/png", data="not base64!!")

    with pytest.raises(ValidationError, match="does not match media type"):
        Attachment(kind="image", media_type="image/png", data=_b64(b"%PDF-1.4\n"))

    with pytest.raises(ValidationError, match="does not match"):
        Attachment(
            kind="file",
            media_type="application/pdf",
            data=f"data:image/png;base64,{_b64(b'%PDF-1.4\n')}",
        )


def test_user_input_rejects_too_many_attachments():
    img = Attachment(
        kind="image",
        media_type="image/png",
        data=_b64(b"\x89PNG\r\n\x1a\nrest"),
        name="p.png",
    )
    with pytest.raises(ValidationError, match="too many attachments"):
        UserInput(attachments=[img] * 9)
