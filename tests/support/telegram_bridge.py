"""Fixtures and builders for telegram bridge tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from lingcore.config import AgentProfile
from lingcore.integrations.telegram.bridge import TelegramBridge
from lingcore.integrations.telegram.config import load_telegram_config
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


def _setup(tmp_path: Path, monkeypatch, *, llm_factory=None, profile_text=PROFILE):
    root = tmp_path / "profile"
    root.mkdir()
    (root / "config.yaml").write_text(profile_text, encoding="utf-8")
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
