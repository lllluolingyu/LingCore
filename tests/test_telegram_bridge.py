"""Offline Telegram bridge isolation, routing, state, and Stop coverage."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from lingcore.config import AgentProfile
from lingcore.events import Final
from lingcore.integrations.telegram.bridge import TelegramBridge, select_photo
from lingcore.integrations.telegram.config import load_telegram_config
from lingcore.integrations.telegram.protocol import TelegramFile, TelegramMessage
from lingcore.integrations.telegram.rendering import TelegramTurnRenderer
from lingcore.llm import LLMChunk
from lingcore.sessions import is_session_id
from tests.fakes import FakeLLMClient, ScriptedTurn

PROFILE = """
name: bridge-test
llm:
  model: test-model
  base_url: http://localhost:11434/v1
tools: [memory]
tool_options:
  memory:
    max_bytes: 1234
"""


@dataclass
class Sent:
    message_id: int


class FakeSender:
    def __init__(self):
        self.next_id = 1
        self.messages: dict[int, str] = {}
        self.sent: list[tuple[int, str]] = []
        self.downloads: dict[str, bytes] = {}
        self.download_started = asyncio.Event()
        self.download_gate: asyncio.Event | None = None
        self.callback_answers = []

    async def send_message(self, chat_id, text, *, reply_markup=None):
        mid = self.next_id
        self.next_id += 1
        self.messages[mid] = text
        self.sent.append((chat_id, text))
        return Sent(mid)

    async def edit_message(self, chat_id, message_id, text, *, reply_markup=None):
        self.messages[message_id] = text

    async def edit_reply_markup(self, chat_id, message_id, *, reply_markup=None):
        return None

    async def delete_message(self, chat_id, message_id):
        self.messages.pop(message_id, None)

    async def send_photo(self, chat_id, data, *, filename):
        return None

    async def send_document(self, chat_id, data, *, filename):
        return None

    async def download_file(self, file_id, *, max_bytes):
        self.download_started.set()
        if self.download_gate is not None:
            await self.download_gate.wait()
        data = self.downloads[file_id]
        if len(data) > max_bytes:
            raise ValueError("too large")
        return data

    async def answer_callback(self, callback_query_id, *, text=None, show_alert=False):
        self.callback_answers.append((callback_query_id, text, show_alert))


def _setup(tmp_path: Path, monkeypatch, *, llm_factory=None):
    root = tmp_path / "profile"
    root.mkdir()
    (root / "config.yaml").write_text(PROFILE, encoding="utf-8")
    (root / "telegram.yaml").write_text(
        """
token_env: TELEGRAM_TEST_TOKEN
allowed_user_ids: [11, 22]
stream_edit_interval: 0
""",
        encoding="utf-8",
    )
    profile = AgentProfile.load(root)
    config = load_telegram_config(profile, require_secrets=False)

    class Encoding:
        def encode(self, text, *, disallowed_special=()):
            return list(text)

    monkeypatch.setattr("lingcore.memory._encoding", lambda _: Encoding())

    async def direct_to_thread(function, *args, **kwargs):
        # Keep this suite strictly single-threaded/offline. Agent's production
        # path still uses asyncio.to_thread; ingest itself has dedicated tests.
        return function(*args, **kwargs)

    monkeypatch.setattr("lingcore.agent.asyncio.to_thread", direct_to_thread)
    if llm_factory is None:

        def llm_factory(_):
            return FakeLLMClient([ScriptedTurn(text="reply")] * 10)

    return profile, config, TelegramBridge(profile, config, llm_factory=llm_factory)


async def _finish(bridge: TelegramBridge):
    tasks = list(bridge.background_tasks)
    if tasks:
        await asyncio.gather(*tasks)
    await asyncio.sleep(0)


async def test_two_users_are_isolated_in_state_profile_and_attachments(
    tmp_path, monkeypatch
):
    profile, config, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    sender.downloads["doc"] = b"private user eleven"

    await bridge.handle_message(
        TelegramMessage(
            update_id=1,
            user_id=11,
            chat_id=11,
            chat_type="private",
            text="one",
            document=TelegramFile("doc", "notes.txt", len(sender.downloads["doc"])),
        ),
        sender,
    )
    await bridge.handle_message(
        TelegramMessage(2, 22, 22, "private", text="two"), sender
    )
    await _finish(bridge)

    one = await bridge.runtime_for(11, 11, sender)
    two = await bridge.runtime_for(22, 22, sender)
    assert one.profile is not two.profile
    assert one.profile.tool_options is not two.profile.tool_options
    assert one.profile.tool_options["memory"] is not two.profile.tool_options["memory"]
    assert one.profile.tool_options["memory"]["path"].endswith(
        ".lingcore/telegram/users/11/memory.md"
    )
    assert "allow_absolute_path" not in one.profile.tool_options["memory"]
    assert one.profile.sessions.path.endswith(".lingcore/telegram/users/11/sessions.db")
    assert one.profile.sessions.allow_absolute_path is False
    assert one.agent.tool_ctx.workspace == config.state_path / "users/11/workspace"
    assert two.agent.tool_ctx.workspace == config.state_path / "users/22/workspace"
    assert (one.agent.tool_ctx.workspace / "attachments/notes.txt").is_file()
    assert not (two.agent.tool_ctx.workspace / "attachments/notes.txt").exists()
    assert one.store.db_path != two.store.db_path
    assert profile.tool_options["memory"] == {"max_bytes": 1234}
    await bridge.shutdown()


async def test_absolute_state_consent_cascades_only_to_derived_paths(
    tmp_path, monkeypatch
):
    root = tmp_path / "profile"
    root.mkdir()
    (root / "config.yaml").write_text(PROFILE, encoding="utf-8")
    state = tmp_path / "outside-state"
    (root / "telegram.yaml").write_text(
        f"""
token_env: TELEGRAM_TEST_TOKEN
allowed_user_ids: [11]
state_dir: {state}
allow_absolute_state_dir: true
""",
        encoding="utf-8",
    )
    profile = AgentProfile.load(root)
    config = load_telegram_config(profile, require_secrets=False)

    class Encoding:
        def encode(self, text, *, disallowed_special=()):
            return list(text)

    monkeypatch.setattr("lingcore.memory._encoding", lambda _: Encoding())
    bridge = TelegramBridge(
        profile,
        config,
        llm_factory=lambda _: FakeLLMClient([ScriptedTurn(text="ok")]),
    )
    sender = FakeSender()
    runtime = await bridge.runtime_for(11, 11, sender)
    assert Path(runtime.profile.tool_options["memory"]["path"]).is_absolute()
    assert runtime.profile.tool_options["memory"]["allow_absolute_path"] is True
    assert Path(runtime.profile.sessions.path).is_absolute()
    assert runtime.profile.sessions.allow_absolute_path is True
    # Unrelated profile permissions remain unchanged.
    assert runtime.profile.memory == profile.memory
    assert runtime.profile.loop == profile.loop
    await bridge.shutdown()


async def test_resume_selection_survives_silent_restart(tmp_path, monkeypatch):
    profile, config, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    await bridge.handle_message(
        TelegramMessage(1, 11, 11, "private", text="first"), sender
    )
    await _finish(bridge)
    runtime = await bridge.runtime_for(11, 11, sender)
    first_id = runtime.agent.memory.session_id

    await bridge.handle_message(
        TelegramMessage(2, 11, 11, "private", text="/new"), sender
    )
    second_id = runtime.agent.memory.session_id
    assert second_id != first_id
    await bridge.handle_message(
        TelegramMessage(3, 11, 11, "private", text=f"/resume {first_id[:8]}"),
        sender,
    )
    assert runtime.agent.memory.session_id == first_id
    await bridge.shutdown()

    restarted = TelegramBridge(
        profile,
        config,
        llm_factory=lambda _: FakeLLMClient([ScriptedTurn(text="after")]),
    )
    restored = await restarted.runtime_for(11, 11, sender)
    assert restored.agent.memory.session_id == first_id
    await restarted.shutdown()


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


async def test_plugin_command_alias_preserves_caption_attachment_and_closes(
    tmp_path, monkeypatch
):
    from lingcore.plugins.commands import Command, CommandCatalog

    _, _, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    sender.downloads["doc"] = b"notes"
    runtime = await bridge.runtime_for(11, 11, sender)
    catalog = CommandCatalog([Command("review", "Review $ARGUMENTS", "code-review")])
    bridge.commands = runtime.agent.commands = catalog
    captured = []

    async def run(incoming):
        captured.append(incoming)
        yield Final("done")

    closed = []

    async def close():
        closed.append(True)

    monkeypatch.setattr(runtime.agent, "run", run)
    monkeypatch.setattr(runtime.agent, "aclose", close)
    raw = "/code_review_review@Bot src"
    await bridge.handle_message(
        TelegramMessage(
            update_id=91,
            user_id=11,
            chat_id=11,
            chat_type="private",
            text=raw,
            document=TelegramFile("doc", "notes.txt", 5),
        ),
        sender,
    )
    await _finish(bridge)
    assert captured[0].text == "Review src"
    assert captured[0].display_text == raw
    assert captured[0].attachments[0].name == "notes.txt"
    await bridge.handle_message(
        TelegramMessage(
            update_id=92, user_id=11, chat_id=11, chat_type="private", text="/new"
        ),
        sender,
    )
    assert closed == [True]
    replacement = runtime.agent
    replacement_closed = []
    original_close = replacement.aclose

    async def close_replacement():
        replacement_closed.append(True)
        await original_close()

    monkeypatch.setattr(replacement, "aclose", close_replacement)
    await bridge.shutdown()
    await bridge.shutdown()
    assert closed == [True]
    assert replacement_closed == [True]


@pytest.mark.parametrize("shutdown", [False, True])
async def test_stop_drains_plugin_notices_before_finalize(
    tmp_path, monkeypatch, shutdown
):
    from lingcore.events import PluginNotice

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
    notices = [PluginNotice("policy", "before_tool", "asked", "Approve?")]

    def drain():
        result = notices.copy()
        notices.clear()
        return result

    original_finalize = runtime.agent.finalize_cancelled_turn

    def finalize(*args, **kwargs):
        notices.clear()
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(runtime.agent, "drain_plugin_notices", drain)
    monkeypatch.setattr(runtime.agent, "finalize_cancelled_turn", finalize)
    if shutdown:
        await bridge.shutdown()
    else:
        await bridge.handle_message(
            TelegramMessage(2, 11, 11, "private", text="/stop"), sender
        )
    visible = [text for _, text in sender.sent]
    notice_index = next(i for i, text in enumerate(visible) if "Plugin policy" in text)
    terminal_index = next(i for i, text in enumerate(visible) if "⏹️" in text)
    assert notice_index < terminal_index
    assert notices == []
    await bridge.shutdown()


async def test_mixed_case_plugin_command_is_not_unknown(tmp_path, monkeypatch):
    from lingcore.plugins.commands import Command, CommandCatalog

    _, _, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    runtime = await bridge.runtime_for(11, 11, sender)
    catalog = CommandCatalog([Command("review", "Review $ARGUMENTS")])
    bridge.commands = runtime.agent.commands = catalog
    captured = []

    async def run(incoming):
        captured.append(incoming)
        yield Final("done")

    monkeypatch.setattr(runtime.agent, "run", run)
    await bridge.handle_message(
        TelegramMessage(
            update_id=93,
            user_id=11,
            chat_id=11,
            chat_type="private",
            text="/Review src",
        ),
        sender,
    )
    await _finish(bridge)
    assert captured[0].text == "Review src"
    assert captured[0].display_text == "/Review src"
