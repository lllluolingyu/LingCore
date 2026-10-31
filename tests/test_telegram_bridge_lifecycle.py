"""Telegram bridge tests: lifecycle."""

from __future__ import annotations

import asyncio
import sqlite3

from lingcore.events import Final
from lingcore.integrations.telegram.bridge import select_photo
from lingcore.integrations.telegram.protocol import TelegramFile, TelegramMessage
from lingcore.integrations.telegram.rendering import TelegramTurnRenderer
from lingcore.llm import LLMChunk
from lingcore.sessions import is_session_id
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.telegram_bridge import FakeSender, _finish, _setup


async def test_private_allowlist_groups_update_dedup_and_album_dedup(
    tmp_path, monkeypatch
):
    _, _, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    denied = TelegramMessage(1, 99, 99, "private", text="hello")
    await bridge.handle_message(denied, sender)
    await bridge.handle_message(denied, sender)
    assert len(sender.sent) == 1
    assert "99" in sender.sent[0][1]

    await bridge.handle_message(
        TelegramMessage(2, 11, -100, "group", text="ignored"), sender
    )
    assert len(sender.sent) == 1

    await bridge.handle_message(
        TelegramMessage(
            3,
            11,
            11,
            "private",
            photos=(TelegramFile("a", "photo.jpg"),),
            media_group_id="album",
        ),
        sender,
    )
    await bridge.handle_message(
        TelegramMessage(
            4,
            11,
            11,
            "private",
            photos=(TelegramFile("b", "photo.jpg"),),
            media_group_id="album",
        ),
        sender,
    )
    album_denials = [text for _, text in sender.sent if "Albums" in text]
    assert len(album_denials) == 1
    await bridge.shutdown()


async def test_stop_cancels_attachment_download_before_agent_checkpoint(
    tmp_path, monkeypatch
):
    _, _, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    sender.downloads["doc"] = b"data"
    sender.download_gate = asyncio.Event()
    await bridge.handle_message(
        TelegramMessage(
            1,
            11,
            11,
            "private",
            text="read",
            document=TelegramFile("doc", "d.txt", 4),
        ),
        sender,
    )
    await sender.download_started.wait()
    runtime = await bridge.runtime_for(11, 11, sender)
    assert runtime.agent.turn_pending_finalization is False

    await bridge.handle_message(
        TelegramMessage(2, 11, 11, "private", text="/stop"), sender
    )
    assert any("before the agent turn began" in text for _, text in sender.sent)
    assert not bridge.background_tasks
    await bridge.shutdown()


async def test_stop_after_checkpoint_finalizes_and_keeps_user_message(
    tmp_path, monkeypatch
):
    started = asyncio.Event()

    class BlockingLLM:
        async def stream(self, messages, tools=None):
            started.set()
            await asyncio.Event().wait()
            yield LLMChunk(text_delta="unreachable")

    _, _, bridge = _setup(tmp_path, monkeypatch, llm_factory=lambda _: BlockingLLM())
    sender = FakeSender()
    await bridge.handle_message(
        TelegramMessage(1, 11, 11, "private", text="long"), sender
    )
    await started.wait()
    runtime = await bridge.runtime_for(11, 11, sender)
    await bridge.handle_message(
        TelegramMessage(2, 11, 11, "private", text="/new"), sender
    )
    await bridge.handle_message(
        TelegramMessage(3, 11, 11, "private", text="/resume invalid"), sender
    )
    assert sum("Cannot switch sessions" in text for _, text in sender.sent) == 2
    await bridge.handle_message(
        TelegramMessage(4, 11, 11, "private", text="/stop"), sender
    )

    assert runtime.agent.turn_pending_finalization is False
    assert [(m.role, m.content) for m in runtime.agent.memory.messages] == [
        ("user", "long")
    ]
    assert any("stopped by user" in text for _, text in sender.sent)
    await bridge.shutdown()


async def test_stop_during_final_delivery_waits_for_committed_reply(
    tmp_path, monkeypatch
):
    final_started = asyncio.Event()
    delivery_gate = asyncio.Event()
    original_handle = TelegramTurnRenderer.handle

    async def gated_handle(self, event):
        if isinstance(event, Final):
            final_started.set()
            await delivery_gate.wait()
        await original_handle(self, event)

    monkeypatch.setattr(TelegramTurnRenderer, "handle", gated_handle)
    _, config, bridge = _setup(
        tmp_path,
        monkeypatch,
        llm_factory=lambda _: FakeLLMClient(
            [ScriptedTurn(text="authoritative response")]
        ),
    )
    # Leave only the first streamed fragment visible until Final reconciliation.
    config.stream_edit_interval = 3_600
    sender = FakeSender()
    await bridge.handle_message(
        TelegramMessage(1, 11, 11, "private", text="question"), sender
    )
    await final_started.wait()
    runtime = await bridge.runtime_for(11, 11, sender)
    assert [
        (message.role, message.content) for message in runtime.agent.memory.messages
    ] == [
        ("user", "question"),
        ("assistant", "authoritative response"),
    ]
    assert "authoritative response" not in sender.messages.values()

    stopping = asyncio.create_task(
        bridge.handle_message(
            TelegramMessage(2, 11, 11, "private", text="/stop"), sender
        )
    )
    await asyncio.sleep(0)
    assert not stopping.done()
    delivery_gate.set()
    await asyncio.wait_for(stopping, timeout=1)
    await _finish(bridge)

    assert "authoritative response" in sender.messages.values()
    assert any("already completed" in text for _, text in sender.sent)
    assert not any("before the agent turn began" in text for _, text in sender.sent)
    session_id = runtime.agent.memory.session_id
    assert [
        (message.role, message.content)
        for message in runtime.store.messages(session_id)
    ] == [
        ("user", "question"),
        ("assistant", "authoritative response"),
    ]
    await bridge.shutdown()


async def test_renderer_failure_falls_back_without_rolling_back_turn(
    tmp_path, monkeypatch
):
    class BrokenEditSender(FakeSender):
        async def edit_message(self, chat_id, message_id, text, *, reply_markup=None):
            raise RuntimeError("Flood control exceeded. Retry in 5 seconds")

    _, _, bridge = _setup(
        tmp_path,
        monkeypatch,
        llm_factory=lambda _: FakeLLMClient([ScriptedTurn(text="complete response")]),
    )
    sender = BrokenEditSender()
    await bridge.handle_message(
        TelegramMessage(1, 11, 11, "private", text="question"), sender
    )
    await _finish(bridge)

    runtime = await bridge.runtime_for(11, 11, sender)
    assert [
        (message.role, message.content) for message in runtime.agent.memory.messages
    ] == [
        ("user", "question"),
        ("assistant", "complete response"),
    ]
    assert "complete response" in sender.messages.values()
    assert "…" not in sender.messages.values()
    assert (
        sum("Live Telegram updates were interrupted" in text for _, text in sender.sent)
        == 1
    )
    session_id = runtime.agent.memory.session_id
    assert [
        (message.role, message.content)
        for message in runtime.store.messages(session_id)
    ] == [
        ("user", "question"),
        ("assistant", "complete response"),
    ]
    await bridge.shutdown()


async def test_corrupt_active_session_self_heals_for_messages_and_new(
    tmp_path, monkeypatch, caplog
):
    _, _, bridge = _setup(tmp_path, monkeypatch)
    with sqlite3.connect(bridge._state.path) as connection:
        connection.executemany(
            "INSERT INTO active_sessions (user_id, session_id) VALUES (?, ?)",
            ((11, "corrupt"), (22, "also-corrupt")),
        )

    sender = FakeSender()
    await bridge.handle_message(
        TelegramMessage(1, 11, 11, "private", text="recover"), sender
    )
    await bridge.handle_message(
        TelegramMessage(2, 22, 22, "private", text="/new"), sender
    )
    await _finish(bridge)

    eleven = await bridge.runtime_for(11, 11, sender)
    twenty_two = await bridge.runtime_for(22, 22, sender)
    assert is_session_id(eleven.agent.memory.session_id)
    assert is_session_id(twenty_two.agent.memory.session_id)
    assert bridge._state.active_session(11) == eleven.agent.memory.session_id
    assert bridge._state.active_session(22) == twenty_two.agent.memory.session_id
    assert any("Started session" in text for _, text in sender.sent)
    assert not any(
        "Could not start your LingCore runtime" in text for _, text in sender.sent
    )
    assert (
        sum(
            "Ignoring invalid Telegram active-session selection" in record.message
            for record in caplog.records
        )
        == 2
    )
    await bridge.shutdown()


async def test_shutdown_cancels_long_turn_instead_of_waiting(tmp_path, monkeypatch):
    started = asyncio.Event()

    class BlockingLLM:
        async def stream(self, messages, tools=None):
            started.set()
            await asyncio.Event().wait()
            yield LLMChunk(text_delta="unreachable")

    _, _, bridge = _setup(tmp_path, monkeypatch, llm_factory=lambda _: BlockingLLM())
    sender = FakeSender()
    await bridge.handle_message(
        TelegramMessage(1, 11, 11, "private", text="long"), sender
    )
    await started.wait()
    await asyncio.wait_for(bridge.shutdown(), timeout=1)
    assert not bridge.background_tasks


def test_photo_selection_uses_largest_representation_within_limit():
    photos = (
        TelegramFile("small", "photo.jpg", 1_000, width=100, height=100),
        TelegramFile("fit", "photo.jpg", 4_000, width=500, height=500),
        TelegramFile("huge", "photo.jpg", 9_000, width=1_000, height=1_000),
    )
    assert select_photo(photos, max_bytes=5_000).file_id == "fit"
