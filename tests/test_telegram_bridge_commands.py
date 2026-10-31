"""Telegram bridge tests: commands."""

from __future__ import annotations

import asyncio

import pytest

from lingcore.events import Final
from lingcore.integrations.telegram.protocol import TelegramFile, TelegramMessage
from lingcore.llm import LLMChunk
from tests.support.telegram_bridge import FakeSender, _finish, _setup


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
