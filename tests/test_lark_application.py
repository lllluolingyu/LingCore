"""Official SDK contracts and WebSocket lifecycle without live credentials."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace

import pytest
from lark_channel import (
    Conversation,
    Identity,
    InboundMessage,
    ResourceDescriptor,
    SendResult,
)

from lingcore.errors import ToolError
from lingcore.integrations.lark.application import (
    SDKSender,
    create_lark_channel,
    normalize_message,
    serve_lark,
)
from lingcore.integrations.lark.protocol import LarkResource
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.lark_bridge import loaded, message


def sdk_message(**kwargs):
    return replace(
        InboundMessage(
            id="om_1",
            create_time=0,
            conversation=Conversation("oc_dm", "p2p"),
            sender=Identity("ou_alice", sender_type="user"),
            body_text="hello",
            content_text="<at>bot</at> hello",
            raw_content_type="text",
        ),
        **kwargs,
    )


@pytest.mark.parametrize("domain", ["lark", "feishu"])
def test_real_sdk_factory_options_are_supported(tmp_path, domain):
    _, config = loaded(tmp_path)
    config.domain = domain
    config._profile_env = {"LARK_APP_ID": "cli_test", "LARK_APP_SECRET": "test"}
    channel = create_lark_channel(config)
    assert channel._config.domain == config.api_domain
    assert channel._config.transport.kind == "ws"
    assert channel._config.policy.group_allowlist == ["oc_group"]
    assert channel._config.policy.require_mention
    assert not channel._config.safety.chat_queue.enabled
    assert channel._config.security.mode == "strict"


def test_normalization_preserves_user_identity_and_removes_bot_mention():
    normalized = normalize_message(sdk_message(mentioned_bot=True))
    assert normalized == replace(message(), mentioned_bot=True)
    media = normalize_message(
        sdk_message(
            raw_content_type="file",
            resources=[ResourceDescriptor("file", "file_key", "a.txt")],
        )
    )
    assert media.text == "" and media.resources == (
        LarkResource("file_key", "file", "a.txt"),
    )


@pytest.mark.parametrize("kind", ["text", "post"])
async def test_sdk_normalizes_wire_event_and_bot_mention(tmp_path, kind):
    _, config = loaded(tmp_path)
    config._profile_env = {"LARK_APP_ID": "cli_test", "LARK_APP_SECRET": "test"}
    channel = create_lark_channel(config)
    channel._pipeline.set_bot_open_id("ou_bot")
    content = (
        {"text": "@_user_1 /help"}
        if kind == "text"
        else {
            "title": "",
            "content": [
                [{"tag": "at", "user_id": "ou_bot"}, {"tag": "text", "text": "/help"}]
            ],
        }
    )
    normalized = await channel._pipeline.normalize(
        message_event={
            "message_id": "om_wire",
            "chat_id": "oc_group",
            "chat_type": "group",
            "message_type": kind,
            "content": json.dumps(content),
            "mentions": [
                {"key": "@_user_1", "id": {"open_id": "ou_bot"}, "name": "Bot"}
            ],
        },
        sender={"sender_id": {"open_id": "ou_alice"}, "sender_type": "user"},
    )
    incoming = normalize_message(normalized)
    assert incoming.user_id == "ou_alice" and incoming.sender_type == "user"
    assert incoming.mentioned_bot and incoming.text.strip() == "/help"


async def test_sender_splits_unicode_text_and_checks_send_result():
    calls = []

    class Channel:
        async def send(self, *args):
            calls.append(args)
            return SendResult(success=True, message_id="om_reply")

    sender = SDKSender(Channel())
    text = "你好🌍" * 2500
    await sender.send_text(message(), text)
    assert len(calls) == 3
    assert "".join(args[1]["text"] for args in calls) == text
    assert all(args[2] == {"reply_to": "om_1"} for args in calls)

    class FailedChannel:
        async def send(self, *args):
            return SendResult(success=False)

    with pytest.raises(ToolError, match="delivery failed"):
        await SDKSender(FailedChannel()).send_text(message(), "test")


async def test_sender_uses_message_resource_endpoint_and_file_payload():
    calls = []

    class Channel:
        async def download_resource(self, *args, **kwargs):
            calls.append((args, kwargs))
            return b"file bytes"

        async def send(self, *args):
            calls.append(args)
            return SendResult(success=True)

    sender = SDKSender(Channel())
    assert (
        await sender.download(message(), LarkResource("key", "file")) == b"file bytes"
    )
    assert calls.pop() == (("key", "file"), {"message_id": "om_1"})
    await sender.send_file(message(), b"bytes", name="a.txt", image=False)
    assert calls[-1][1] == {"file": {"source": b"bytes", "file_name": "a.txt"}}
    await sender.send_file(message(), b"image", name="a.png", image=True)
    assert calls[-1][1] == {"image": {"source": b"image"}}


@pytest.mark.parametrize("fail", [False, True])
async def test_lifecycle_hands_off_sdk_thread_and_always_disconnects(
    tmp_path, monkeypatch, fail
):
    profile, config = loaded(tmp_path)
    main_thread = threading.get_ident()
    handlers = {}
    calls = []

    class Bridge:
        def __init__(self, *args, **kwargs):
            calls.append("created")

        async def handle_message(self, msg):
            assert threading.get_ident() == main_thread
            assert msg.text == "hello"
            calls.append("message")

        async def shutdown(self):
            calls.append("shutdown")

    class Channel:
        def on(self, name, handler):
            handlers[name] = handler

        async def connect(self):
            await asyncio.to_thread(
                lambda: asyncio.run(handlers["message"](sdk_message()))
            )
            if fail:
                raise RuntimeError("failed startup")

        async def disconnect(self):
            calls.append("disconnect")

    monkeypatch.setattr("lingcore.integrations.lark.application.LarkBridge", Bridge)
    if fail:
        with pytest.raises(RuntimeError, match="startup"):
            await serve_lark(profile, config, channel=Channel())
    else:
        await serve_lark(profile, config, channel=Channel())
    assert calls == ["created", "message", "shutdown", "disconnect"]


async def test_application_drives_real_bridge(tmp_path, monkeypatch):
    profile, config = loaded(tmp_path)
    handlers = {}
    reply = asyncio.Event()
    sent = []

    class Encoding:
        def encode(self, text, *, disallowed_special=()):
            return list(text)

    monkeypatch.setattr("lingcore.memory._encoding", lambda _: Encoding())

    class Channel:
        def on(self, name, handler):
            handlers[name] = handler

        async def connect(self):
            await handlers["message"](sdk_message())
            await asyncio.wait_for(reply.wait(), 2)

        async def send(self, chat, content, opts):
            sent.append((chat, content, opts))
            reply.set()
            return SendResult(success=True)

        async def disconnect(self):
            pass

    await serve_lark(
        profile,
        config,
        channel=Channel(),
        llm_factory=lambda _: FakeLLMClient([ScriptedTurn(text="reply")]),
    )
    assert sent == [("oc_dm", {"text": "reply"}, {"reply_to": "om_1"})]
