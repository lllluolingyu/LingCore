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


def test_anthropic_native_attachments_and_text_fallback():
    image_data = _b64(b"\x89PNG\r\n\x1a\nrest")
    pdf_data = _b64(b"%PDF-1.4\n")
    image = Attachment(kind="image", media_type="image/png", data=image_data)
    pdf = Attachment(kind="file", media_type="application/pdf", data=pdf_data)
    text = Attachment(
        kind="text",
        media_type="text/plain",
        data=_b64(b"notes"),
        name="notes.txt",
        fallback_text="notes",
    )
    wire = Message.user("inspect", attachments=[image, pdf, text]).to_anthropic()
    parts = wire["content"]
    assert wire["role"] == "user"
    assert parts[0]["type"] == "text"
    assert all(value in parts[0]["text"] for value in ["inspect", "notes.txt", "notes"])
    assert parts[1:] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_data},
        },
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_data,
            },
        },
    ]


def test_anthropic_narrowed_modalities_keep_fallback_without_payload():
    image = Attachment(
        kind="image",
        media_type="image/png",
        data=_b64(b"\x89PNG\r\n\x1a\nrest"),
        name="chart.png",
        fallback_text="a bar chart",
    )
    wire = Message.user("inspect", attachments=[image]).to_anthropic(
        attachment_modalities=frozenset()
    )
    assert all(part["type"] == "text" for part in wire["content"])
    assert "a bar chart" in wire["content"][0]["text"]
    assert image.data not in str(wire)


def test_anthropic_attachment_only_message_has_no_empty_text_block():
    image = Attachment(
        kind="image", media_type="image/png", data=_b64(b"\x89PNG\r\n\x1a\nrest")
    )
    assert [
        part["type"]
        for part in Message.user("", attachments=[image]).to_anthropic()["content"]
    ] == ["image"]


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
    img = Attachment(
        kind="image", media_type="image/png", data=img_data, name="pic.png"
    )
    pdf = Attachment(
        kind="file", media_type="application/pdf", data=pdf_data, name="doc.pdf"
    )
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
            data="data:image/png;base64," + _b64(b"%PDF-1.4\n"),
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


# --------------------------------------------------------------------------- #
# Modality-narrowed rendering (attachment_modalities=...)                      #
# --------------------------------------------------------------------------- #


def _png() -> Attachment:
    return Attachment(
        kind="image",
        media_type="image/png",
        data=_b64(b"\x89PNG\r\n\x1a\nrest"),
        name="p.png",
    )


def test_unsupported_modality_collapses_to_plain_string():
    pdf = Attachment(
        kind="file",
        media_type="application/pdf",
        data=_b64(b"%PDF-1.4\n"),
        name="doc.pdf",
        fallback_text="page one words",
    )
    wire = Message.user("summarize", attachments=[pdf]).to_openai(
        attachment_modalities=frozenset()
    )
    # No native part remains -> plain string (text-only servers may reject
    # a parts array outright).
    assert isinstance(wire["content"], str)
    assert "summarize" in wire["content"]
    assert "doc.pdf" in wire["content"]
    assert "page one words" in wire["content"]


def test_unsupported_modality_without_fallback_gets_placeholder():
    wire = Message.user("", attachments=[_png()]).to_openai(
        attachment_modalities=frozenset()
    )
    assert isinstance(wire["content"], str)
    assert "p.png" in wire["content"]
    assert "no text fallback is available" in wire["content"]


def test_mixed_modalities_merge_fallback_into_text_part():
    img = _png()
    pdf = Attachment(
        kind="file",
        media_type="application/pdf",
        data=_b64(b"%PDF-1.4\n"),
        name="doc.pdf",
        fallback_text="extracted words",
    )
    wire = Message.user("look", attachments=[img, pdf]).to_openai(
        attachment_modalities=frozenset({"image"})
    )
    parts = wire["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert "look" in parts[0]["text"] and "extracted words" in parts[0]["text"]
    assert parts[1]["image_url"]["url"] == f"data:image/png;base64,{img.data}"


def test_full_modalities_param_matches_default_render():
    msg = Message.user("d", attachments=[_png()])
    assert (
        msg.to_openai(attachment_modalities=frozenset({"image", "file"}))
        == msg.to_openai()
    )


def test_fallback_text_is_truncated_never_rejected():
    from lingcore.media_types import FALLBACK_TEXT_MAX_CHARS

    att = Attachment(
        kind="image",
        media_type="image/png",
        data=_b64(b"\x89PNG\r\n\x1a\nrest"),
        fallback_text="y" * (FALLBACK_TEXT_MAX_CHARS + 5_000),
    )
    assert len(att.fallback_text) == FALLBACK_TEXT_MAX_CHARS
    assert att.fallback_text.endswith("[fallback text truncated]")


# --------------------------------------------------------------------------- #
# text / binary attachment kinds                                               #
# --------------------------------------------------------------------------- #


def test_text_attachment_validates_and_inlines_content():
    att = Attachment(
        kind="text",
        media_type="text/x-python",
        data=_b64(b"print('hi')\n"),
        name="a.py",
        fallback_text="print('hi')\n",
    )
    wire = Message.user("explain", attachments=[att]).to_openai()
    # text is never a native modality, so the message collapses to a string
    # with the file's content inlined.
    assert isinstance(wire["content"], str)
    assert "explain" in wire["content"]
    assert "a.py" in wire["content"]
    assert "print('hi')" in wire["content"]


def test_binary_attachment_renders_its_note_verbatim():
    note = (
        "[binary file saved to attachments/x.bin (application/octet-stream, 4 bytes)]"
    )
    att = Attachment(
        kind="binary",
        media_type="application/octet-stream",
        data=_b64(b"\x00\x01\x02\x03"),
        name="x.bin",
        fallback_text=note,
    )
    wire = Message.user("", attachments=[att]).to_openai()
    assert wire["content"] == note


def test_text_attachment_rejects_nul_bytes():
    with pytest.raises(ValidationError, match="NUL"):
        Attachment(kind="text", media_type="text/plain", data=_b64(b"a\x00b"))


def test_text_attachment_rejects_malformed_media_type():
    with pytest.raises(ValidationError, match="invalid media type"):
        Attachment(kind="text", media_type="not a type", data=_b64(b"hello"))


def test_text_and_image_mix_keeps_native_image_plus_inlined_text():
    img = _png()
    txt = Attachment(
        kind="text",
        media_type="text/plain",
        data=_b64(b"notes"),
        name="n.txt",
        fallback_text="notes",
    )
    wire = Message.user("look", attachments=[img, txt]).to_openai()
    parts = wire["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert "look" in parts[0]["text"] and "notes" in parts[0]["text"]


def test_old_style_image_dict_still_validates():
    # Back-compat: a stored row carrying only image/file kinds loads unchanged.
    att = Attachment.model_validate(
        {
            "kind": "image",
            "media_type": "image/png",
            "data": _b64(b"\x89PNG\r\n\x1a\nrest"),
            "name": "p.png",
        }
    )
    assert att.kind == "image" and att.media_type == "image/png"


def test_reasoning_content_round_trips_and_renders_only_on_request() -> None:
    msg = Message.assistant("hi", reasoning_content="because")
    assert Message.model_validate_json(msg.model_dump_json()) == msg
    assert "reasoning_content" not in msg.to_openai()
    assert msg.to_openai(include_reasoning=True)["reasoning_content"] == "because"
    # Legacy rows lack the field entirely and still load.
    legacy = Message.model_validate_json('{"role": "assistant", "content": "x"}')
    assert legacy.to_openai(include_reasoning=True)["reasoning_content"] == ""
    assert "reasoning_content" not in Message.user("q").to_openai(
        include_reasoning=True
    )


def test_anthropic_content_roundtrips_without_leaking_to_openai():
    content = [
        {"type": "thinking", "thinking": "summary", "signature": "opaque-signature"},
        {"type": "text", "text": "Checking"},
        {"type": "redacted_thinking", "data": "opaque-data"},
        {"type": "tool_use", "id": "c", "name": "read_file", "input": {}},
    ]
    message = Message.assistant(
        "Checking",
        tool_calls=[ToolCall(id="c", name="read_file", arguments={})],
        anthropic_content=content,
    )
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored.to_anthropic() == {"role": "assistant", "content": content}
    openai_wire = restored.to_openai()
    assert openai_wire["content"] == "Checking"
    assert openai_wire["tool_calls"][0]["id"] == "c"
    assert "opaque" not in str(openai_wire)
    assert "anthropic_content" not in openai_wire

    wire = restored.to_anthropic()
    wire["content"][0]["signature"] = "changed"
    assert restored.anthropic_content == content
    assert (
        Message.model_validate_json(
            '{"role":"assistant","content":"legacy"}'
        ).anthropic_content
        is None
    )


def test_anthropic_content_is_only_replayed_for_assistants():
    message = Message(
        role="user",
        content="hi",
        anthropic_content=[
            {"type": "thinking", "thinking": "summary", "signature": "sig"}
        ],
    )
    assert message.to_anthropic() == {"role": "user", "content": "hi"}
