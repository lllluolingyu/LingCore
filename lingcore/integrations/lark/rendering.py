"""Deliver authoritative replies and tool attachments to Lark."""

from __future__ import annotations

import base64
import logging

from lingcore.events import (
    AgentEvent,
    Error,
    Final,
    PluginNotice,
    ToolResultEvent,
    TurnCancelled,
)
from lingcore.integrations.lark.protocol import LarkMessage, LarkSender

_LOG = logging.getLogger(__name__)


async def deliver(event: AgentEvent, message: LarkMessage, sender: LarkSender) -> None:
    # Buffer text until Final/Error: only post-output-guardrail content is sent.
    if isinstance(event, Final):
        await sender.send_text(message, event.content or "(empty response)")
    elif isinstance(event, Error):
        text = "\n\n".join(
            part for part in (event.partial_content, f"Error: {event.message}") if part
        )
        await sender.send_text(message, text)
    elif isinstance(event, TurnCancelled):
        await sender.send_text(message, f"Stopped: {event.reason}")
    elif isinstance(event, PluginNotice):
        # Routine context/result updates are diagnostics, not chat replies.
        if event.action == "modified":
            _LOG.debug(
                "Plugin %s · %s · %s: %s",
                event.plugin,
                event.hook,
                event.action,
                event.message,
            )
        else:
            await sender.send_text(
                message,
                f"Plugin {event.plugin} · {event.hook} · {event.action}: {event.message}",
            )
    elif isinstance(event, ToolResultEvent):
        for attachment in event.result.attachments:
            await sender.send_file(
                message,
                base64.b64decode(attachment.data, validate=True),
                name=attachment.name or "attachment",
                image=attachment.kind == "image",
            )
