"""Offline Lark bridge fixtures and message builders."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from lingcore.config import AgentProfile
from lingcore.integrations.lark.bridge import LarkBridge
from lingcore.integrations.lark.config import load_lark_config
from lingcore.integrations.lark.protocol import LarkMessage
from tests.fakes import FakeLLMClient, ScriptedTurn

PROFILE = """name: lark-test
llm:
  model: test-model
  base_url: http://localhost:11434/v1
tools: [memory]
"""
CONFIG = """allowed_user_ids: [ou_alice, ou_bob]
allowed_chat_ids: [oc_group]
"""


def loaded(tmp_path, *, profile_text=PROFILE, config_text=CONFIG):
    root = tmp_path / "profile"
    root.mkdir(exist_ok=True)
    (root / "config.yaml").write_text(profile_text)
    (root / "lark.yaml").write_text(config_text)
    profile = AgentProfile.load(root)
    return profile, load_lark_config(profile, require_secrets=False)


def message(mid="om_1", text="hello", **kwargs):
    return replace(LarkMessage(mid, "ou_alice", "oc_dm", "p2p", text=text), **kwargs)


class FakeSender:
    def __init__(self):
        self.sent = []
        self.files = []
        self.downloads = {}
        self.download_started = asyncio.Event()
        self.download_gate = None
        self.sent_event = asyncio.Event()

    async def send_text(self, message, text):
        self.sent.append((message, text))
        self.sent_event.set()

    async def send_file(self, message, data, *, name, image):
        self.files.append((message, data, name, image))

    async def download(self, message, resource):
        self.download_started.set()
        if self.download_gate is not None:
            await self.download_gate.wait()
        return self.downloads[resource.key]


def setup_bridge(
    tmp_path, monkeypatch, *, llm_factory=None, sender=None, profile_text=PROFILE
):
    profile, config = loaded(tmp_path, profile_text=profile_text)

    class Encoding:
        def encode(self, text, *, disallowed_special=()):
            return list(text)

    monkeypatch.setattr("lingcore.memory._encoding", lambda _: Encoding())
    if llm_factory is None:

        def llm_factory(_):
            return FakeLLMClient([ScriptedTurn(text="reply")] * 10)

    sender = sender or FakeSender()
    return (
        profile,
        config,
        LarkBridge(profile, config, sender, llm_factory=llm_factory),
        sender,
    )


async def finish(bridge):
    await asyncio.wait_for(asyncio.gather(*bridge.background_tasks), 5)
    await asyncio.sleep(0)
