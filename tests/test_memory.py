"""Tests for WindowMemory (M3) — focus on block-aware trimming."""

from __future__ import annotations

from lingcore.memory import WindowMemory
from lingcore.message import Message, ToolCall, ToolResult


def test_render_prepends_system():
    mem = WindowMemory(model="gpt-4o")
    mem.add(Message.user("hi"))
    rendered = mem.render("SYS")
    assert rendered[0].role == "system"
    assert rendered[0].content == "SYS"
    assert rendered[1].content == "hi"


def test_message_count_cap_keeps_recent():
    mem = WindowMemory(max_messages=2, model="gpt-4o")
    for i in range(5):
        mem.add(Message.user(f"m{i}"))
    rendered = mem.render("SYS")
    # system + at most 2 most-recent user blocks
    bodies = [m.content for m in rendered if m.role == "user"]
    assert bodies == ["m3", "m4"]


def test_tool_block_never_orphaned():
    """A tool result must never be kept without its assistant tool_call."""
    mem = WindowMemory(max_messages=2, model="gpt-4o")
    mem.add(Message.user("old"))
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a"})
    mem.add(Message.assistant(content="", tool_calls=[call]))
    mem.add(Message.from_tool_result(ToolResult(call_id="c1", name="read_file", content="x")))

    rendered = mem.render("SYS")
    roles = [m.role for m in rendered]
    # If a tool message is present, the message before it must carry tool_calls.
    for i, m in enumerate(rendered):
        if m.role == "tool":
            assert rendered[i - 1].role == "assistant"
            assert rendered[i - 1].tool_calls, "tool result orphaned from its call"


def test_keeps_at_least_one_block_even_over_budget():
    mem = WindowMemory(max_messages=1, max_tokens=1, model="gpt-4o")
    mem.add(Message.user("a very long message that exceeds the tiny token budget"))
    rendered = mem.render("SYS")
    # system + the single block (we never return an empty conversation)
    assert any(m.role == "user" for m in rendered)


def test_attachment_estimates_flat_and_fallback_text():
    import base64

    from lingcore.message import Attachment

    mem = WindowMemory(model="gpt-4o")
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode("ascii")
    plain = Attachment(kind="image", media_type="image/png", data=png)
    converted = plain.model_copy(update={"fallback_text": "x" * 40_000})

    flat = mem._tokens(Message.user("m", attachments=[plain]))
    with_text = mem._tokens(Message.user("m", attachments=[converted]))
    assert flat >= 1_000  # flat per-image estimate
    # A large fallback may be what actually goes on the wire: the estimate
    # must grow with it (40k chars ≈ 10k tokens > the flat 1k).
    assert with_text >= 10_000
