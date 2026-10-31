"""Best-effort Telegram delivery and cancellation rendering."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from lingcore.events import AgentEvent, Error, Final, PluginNotice, TurnCancelled
from lingcore.integrations.telegram.protocol import TelegramSender
from lingcore.integrations.telegram.rendering import (
    EMPTY_RESPONSE,
    TelegramTurnRenderer,
    chunk_text,
    error_text,
)

if TYPE_CHECKING:
    from lingcore.integrations.telegram.bridge import (
        TelegramBridge,
        TelegramUserRuntime,
    )
_LOG = logging.getLogger(__name__)


async def _fallback_delivery(
    self: TelegramBridge,
    event: AgentEvent,
    *,
    renderer: TelegramTurnRenderer | None,
    sender: TelegramSender,
    chat_id: int,
    warning_sent: bool,
) -> bool:
    """Best-effort delivery after one renderer operation failed."""
    if isinstance(event, Final) or (
        isinstance(event, Error) and event.partial_content is not None
    ):
        # Remove stale placeholders/partial chunks where possible before
        # sending the authoritative response as fresh plain messages.
        for response_id in renderer.response_message_ids if renderer else ():
            try:
                await sender.delete_message(chat_id, response_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        await self._best_effort_plain(
            sender,
            chat_id,
            (event.content or EMPTY_RESPONSE)
            if isinstance(event, Final)
            else error_text(event),
        )
        return True
    if isinstance(event, Error):
        await self._best_effort_plain(sender, chat_id, error_text(event))
        return True
    if isinstance(event, TurnCancelled):
        await self._best_effort_plain(sender, chat_id, f"⏹️ {event.reason}")
        return True
    if not warning_sent:
        await self._best_effort_plain(
            sender,
            chat_id,
            "⚠️ Live Telegram updates were interrupted. "
            "The final response will be sent as a new message if needed.",
        )
    return True


async def _best_effort_plain(
    sender: TelegramSender,
    chat_id: int,
    text: str,
) -> None:
    for chunk in chunk_text(text):
        try:
            await sender.send_message(chat_id, chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Exception strings may contain the bot token.
            _LOG.warning(
                "Telegram fallback delivery failed with %s",
                type(exc).__name__,
            )
            return


async def _deliver_cancelled_events(
    self: TelegramBridge,
    runtime: TelegramUserRuntime,
    terminal: AgentEvent | None,
    notices: list[PluginNotice],
) -> None:
    """Deliver captured policy decisions before a cancellation terminal."""
    renderer = runtime.renderer
    runtime.renderer = None
    sender = self._senders.get(runtime.user_id)
    chat_id = self._chat_ids.get(runtime.user_id)
    pending = [*getattr(runtime.agent, "drain_usage", lambda: [])(), *notices]
    if terminal is not None:
        pending.append(terminal)
    for event in pending:
        if renderer is not None:
            try:
                await renderer.handle(event)
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _LOG.warning(
                    "Telegram cancellation delivery failed with %s",
                    type(exc).__name__,
                )
        if sender is None or chat_id is None:
            continue
        if isinstance(event, PluginNotice):
            text = f"Plugin {event.plugin} · {event.hook} · {event.action}: {event.message}"
        elif isinstance(event, Final | Error | TurnCancelled):
            await self._fallback_delivery(
                event,
                renderer=renderer,
                sender=sender,
                chat_id=chat_id,
                warning_sent=False,
            )
            continue
        else:
            continue
        await self._best_effort_plain(sender, chat_id, text)
