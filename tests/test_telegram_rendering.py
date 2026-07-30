"""Deterministic Telegram rendering and confirmation tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from lingcore.events import Final, StreamRetry, TextDelta, ToolCallStarted, ToolResultEvent
from lingcore.integrations.telegram.confirmations import ConfirmationManager
from lingcore.integrations.telegram.protocol import InlineButtons
from lingcore.integrations.telegram.rendering import (
    DISCARDED_NOTICE,
    SUPERSEDED_NOTICE,
    TelegramTurnRenderer,
    chunk_text,
)
from lingcore.message import ToolCall, ToolResult


@dataclass
class Sent:
    message_id: int
    text: str


class FakeSender:
    def __init__(self):
        self.next_id = 1
        self.messages: dict[int, str] = {}
        self.sent: list[tuple[int, str, InlineButtons | None]] = []
        self.edits: list[tuple[int, str]] = []
        self.deleted: list[int] = []
        self.markup_edits: list[int] = []
        self.callback_answers = []
        self.fail_deletes: set[int] = set()

    async def send_message(self, chat_id, text, *, reply_markup=None):
        mid = self.next_id
        self.next_id += 1
        self.messages[mid] = text
        self.sent.append((chat_id, text, reply_markup))
        return Sent(mid, text)

    async def edit_message(
        self, chat_id, message_id, text, *, reply_markup=None
    ):
        self.messages[message_id] = text
        self.edits.append((message_id, text))

    async def edit_reply_markup(
        self, chat_id, message_id, *, reply_markup=None
    ):
        self.markup_edits.append(message_id)

    async def delete_message(self, chat_id, message_id):
        if message_id in self.fail_deletes:
            raise RuntimeError("delete failed")
        self.deleted.append(message_id)
        self.messages.pop(message_id, None)

    async def send_photo(self, chat_id, data, *, filename):
        pass

    async def send_document(self, chat_id, data, *, filename):
        pass

    async def answer_callback(
        self, callback_query_id, *, text=None, show_alert=False
    ):
        self.callback_answers.append((callback_query_id, text, show_alert))


def test_chunker_preserves_text_and_prefers_boundaries():
    text = "a" * 3_900 + "\n\n" + "b" * 200
    chunks = chunk_text(text)
    assert chunks[0].endswith("\n\n")
    assert "".join(chunks) == text
    assert all(len(chunk) <= 4_000 for chunk in chunks)


async def test_stream_throttles_active_edits_and_final_is_authoritative():
    sender = FakeSender()
    now = [10.0]
    renderer = TelegramTurnRenderer(
        sender, 5, edit_interval=1.0, clock=lambda: now[0]
    )
    await renderer.handle(TextDelta("first"))
    await renderer.handle(TextDelta(" second"))
    assert [text for _, text in sender.edits] == ["first"]

    await renderer.handle(Final("authoritative"))
    assert list(sender.messages.values()) == ["authoritative"]


async def test_stream_rollover_and_surplus_delete_fallback():
    sender = FakeSender()
    renderer = TelegramTurnRenderer(
        sender, 5, edit_interval=0, clock=lambda: 1.0
    )
    await renderer.handle(TextDelta("x" * 4_000))
    assert len(renderer.response_message_ids) == 2
    surplus_id = renderer.response_message_ids[1]
    sender.fail_deletes.add(surplus_id)

    await renderer.handle(Final("short"))

    assert sender.messages[renderer.response_message_ids[0]] == "short"
    assert sender.messages[surplus_id] == SUPERSEDED_NOTICE


async def test_retry_discards_old_attempt_and_starts_fresh_set():
    sender = FakeSender()
    renderer = TelegramTurnRenderer(sender, 5, edit_interval=0)
    await renderer.handle(TextDelta("partial"))
    old_id = renderer.response_message_ids[0]
    await renderer.handle(
        StreamRetry(attempt=1, max_attempts=3, reason="lost", discarded_chars=7)
    )
    assert sender.messages[old_id] == DISCARDED_NOTICE
    assert renderer.response_message_ids != (old_id,)
    await renderer.handle(Final("new answer"))
    assert sender.messages[renderer.response_message_ids[0]] == "new answer"


async def test_status_exposes_tool_name_and_outcome_not_body():
    sender = FakeSender()
    renderer = TelegramTurnRenderer(sender, 5, edit_interval=0)
    call = ToolCall(id="c", name="read_file", arguments={"path": "secret"})
    await renderer.handle(ToolCallStarted(call))
    await renderer.handle(
        ToolResultEvent(
            ToolResult(
                call_id="c",
                name="read_file",
                content="result-body-sentinel",
                ok=False,
            )
        )
    )
    output = "\n".join(sender.messages.values())
    assert "read_file" in output and "failed" in output
    assert "result-body-sentinel" not in output
    assert "secret" not in output


async def test_parallel_confirmations_wrong_user_and_independent_resolution():
    sender = FakeSender()
    manager = ConfirmationManager(60)
    first = asyncio.create_task(manager.request(1, 1, "one", sender))
    second = asyncio.create_task(manager.request(1, 1, "two", sender))
    await asyncio.sleep(0)
    first_data = sender.sent[0][2][0][0].callback_data
    second_data = sender.sent[1][2][0][1].callback_data

    answer, alert = await manager.resolve_callback(
        callback_data=first_data, user_id=2, chat_id=1
    )
    assert alert and "another user" in answer
    assert not first.done()

    await manager.resolve_callback(
        callback_data=first_data, user_id=1, chat_id=1
    )
    await manager.resolve_callback(
        callback_data=second_data, user_id=1, chat_id=1
    )
    assert await first is True
    assert await second is False
    assert len(sender.markup_edits) == 2


async def test_confirmation_timeout_and_shutdown_denial_without_real_sleep():
    sender = FakeSender()
    gates: list[asyncio.Future[None]] = []

    async def fake_sleep(_):
        gate = asyncio.get_running_loop().create_future()
        gates.append(gate)
        await gate

    manager = ConfirmationManager(60, sleep=fake_sleep)
    timed = asyncio.create_task(manager.request(1, 1, "one", sender))
    stopped = asyncio.create_task(manager.request(2, 2, "two", sender))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    gates[0].set_result(None)
    assert await timed is False
    await manager.deny_all()
    assert await stopped is False
