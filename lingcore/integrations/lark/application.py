"""Thin adapter over the optional official Lark Channel SDK."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, ToolError
from lingcore.integrations.lark.bridge import LarkBridge, LLMFactory
from lingcore.integrations.lark.config import LarkConfig, load_lark_config
from lingcore.integrations.lark.protocol import LarkMessage, LarkResource

_LOG = logging.getLogger(__name__)


def normalize_message(message: Any) -> LarkMessage:
    """Convert a normalized SDK message without exposing the SDK to the bridge."""
    return LarkMessage(
        message_id=message.message_id,
        user_id=message.sender_id,
        chat_id=message.chat_id,
        chat_type=message.chat_type,
        text=message.body_text if message.raw_content_type in {"text", "post"} else "",
        message_type=message.raw_content_type,
        mentioned_bot=message.mentioned_bot,
        sender_type=message.sender_type,
        resources=tuple(
            LarkResource(item.file_key, item.type, item.file_name)
            for item in message.resources
        ),
    )


class SDKSender:
    def __init__(self, channel: Any) -> None:
        self.channel = channel

    @staticmethod
    def _check(result: Any) -> None:
        # Failure is usually a SendResult, not a raised exception. Avoid
        # exposing its raw response or upstream exception in logs or chat.
        if not result.success:
            raise ToolError("Lark message delivery failed")

    async def send_text(self, message: LarkMessage, text: str) -> None:
        for start in range(0, len(text), 3000):
            result = await self.channel.send(
                message.chat_id,
                {"text": text[start : start + 3000]},
                {"reply_to": message.message_id},
            )
            self._check(result)

    async def send_file(
        self, message: LarkMessage, data: bytes, *, name: str, image: bool
    ) -> None:
        payload = (
            {"image": {"source": data}}
            if image
            else {"file": {"source": data, "file_name": name}}
        )
        self._check(
            await self.channel.send(
                message.chat_id, payload, {"reply_to": message.message_id}
            )
        )

    async def download(self, message: LarkMessage, resource: LarkResource) -> bytes:
        # Message attachments require the message resource endpoint; standalone
        # image/file download endpoints cannot fetch users' attachments.
        data = await self.channel.download_resource(
            resource.key, resource.kind, message_id=message.message_id
        )
        if not isinstance(data, bytes):
            raise ToolError("Could not download the Lark attachment")
        return data


def create_lark_channel(config: LarkConfig) -> Any:
    from lark_channel import (
        ChatQueueConfig,
        FeishuChannel,
        InboundConfig,
        PolicyConfig,
        SafetyConfig,
        SecurityConfig,
    )

    app_id, app_secret = config.resolve_credentials()
    return FeishuChannel(
        app_id=app_id,
        app_secret=app_secret,
        domain=config.api_domain,
        transport="ws",
        # Routing uses open_ids, so contact-name API lookups are unnecessary.
        name_lookup=lambda ids: {},
        # DMs reach the bridge so an unknown user can learn their app-scoped
        # open_id. The bridge enforces the sender allowlist in every chat.
        policy=PolicyConfig(
            dm_policy="open",
            group_policy="allowlist" if config.allowed_chat_ids else "disabled",
            group_allowlist=config.allowed_chat_ids,
            require_mention=True,
            respond_to_mention_all=False,
        ),
        inbound=InboundConfig(
            expand_merge_forward=False,
            fetch_interactive_card=False,
            reaction_notifications="off",
        ),
        safety=SafetyConfig(
            chat_queue=ChatQueueConfig(enabled=False),
        ),
        security=SecurityConfig(
            mode="strict",
            max_ws_fragment_parts=128,
            max_ws_fragment_bytes=8 * 1024 * 1024,
            max_concurrent_ws_handlers=32,
        ),
    )


async def serve_lark(
    profile: AgentProfile,
    config: LarkConfig,
    *,
    channel: Any = None,
    llm_factory: LLMFactory | None = None,
) -> None:
    if channel is None:
        channel = create_lark_channel(config)
    loop = asyncio.get_running_loop()
    bridge = LarkBridge(profile, config, SDKSender(channel), llm_factory=llm_factory)

    async def on_message(raw: Any) -> None:
        try:
            message = normalize_message(raw)
            # The SDK invokes handlers on its own background loop. SQLite, Agent,
            # asyncio locks and approval futures all belong to our owning loop.
            if asyncio.get_running_loop() is loop:
                await bridge.handle_message(message)
            else:
                future = asyncio.run_coroutine_threadsafe(
                    bridge.handle_message(message), loop
                )
                await asyncio.wrap_future(future)
        except Exception as exc:
            # Keep raw exceptions out of the SDK's handler traceback logger.
            _LOG.warning("Lark message handling failed: %s", type(exc).__name__)

    async def on_error(error: Any) -> None:
        _LOG.warning("Lark SDK error: %s", type(error).__name__)

    channel.on("message", on_message)
    channel.on("error", on_error)
    try:
        await channel.connect()
    finally:
        try:
            await bridge.shutdown()
        finally:
            await channel.disconnect()


def run_lark(
    profile_path: str | Path, *, lark_config_path: str | Path | None = None
) -> int:
    # Fail with the CLI's install hint before attempting to load credentials.
    import lark_channel  # noqa: F401

    profile = AgentProfile.load(profile_path)
    config = load_lark_config(profile, lark_config_path)
    try:
        asyncio.run(serve_lark(profile, config))
    except (KeyboardInterrupt, ConfigError):
        raise
    except Exception as exc:
        raise ConfigError(
            f"Lark connection failed ({type(exc).__name__}); check app credentials, "
            "permissions, domain and WebSocket event subscription"
        ) from None
    return 0
