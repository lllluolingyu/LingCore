"""PTB-light rendering of LingCore events into Telegram messages."""

from __future__ import annotations

import base64
import time
from collections import OrderedDict
from collections.abc import Callable

from lingcore.events import (
    AgentEvent,
    Compacted,
    Error,
    Final,
    PluginNotice,
    SkillActivated,
    StreamRetry,
    TextDelta,
    TodoUpdated,
    ToolCallStarted,
    ToolResultEvent,
    TurnCancelled,
    UsageReported,
)
from lingcore.integrations.telegram.protocol import TelegramSender, message_id

TELEGRAM_TEXT_LIMIT = 4_000
EMPTY_RESPONSE = "(empty response)"
SUPERSEDED_NOTICE = "(superseded)"
DISCARDED_NOTICE = "⚠️ Discarded partial response."
STREAM_PLACEHOLDER = "…"


def chunk_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Split deterministically, preferring paragraph/newline/space boundaries."""
    if limit <= 0:
        raise ValueError("chunk limit must be positive")
    if not text:
        return []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut >= 0:
            cut += 2
        else:
            cut = window.rfind("\n")
            if cut >= 0:
                cut += 1
            else:
                # Keep the whitespace itself in the frozen chunk so joining the
                # result reproduces the authoritative content exactly.
                whitespace = max(
                    window.rfind(" "),
                    window.rfind("\t"),
                    window.rfind("\r"),
                )
                cut = whitespace + 1 if whitespace >= 0 else limit
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        chunks.append(remaining)
    return chunks


def _not_modified(exc: Exception) -> bool:
    return "message is not modified" in str(exc).lower()


class TelegramTurnRenderer:
    """Render one attempt while retaining message ids for Final reconciliation."""

    def __init__(
        self,
        sender: TelegramSender,
        chat_id: int,
        *,
        edit_interval: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sender = sender
        self.chat_id = chat_id
        self.edit_interval = edit_interval
        self._clock = clock
        self._text = ""
        self._messages: list[int] = []
        self._sent_text: list[str] = []
        self._last_active_edit = float("-inf")
        self._status_message: int | None = None
        self._status_text = ""
        self._tools: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._activity: list[str] = []

    @property
    def response_message_ids(self) -> tuple[int, ...]:
        return tuple(self._messages)

    async def handle(self, event: AgentEvent) -> None:
        if isinstance(event, TextDelta):
            self._text += event.text
            await self._render_stream()
        elif isinstance(event, Final):
            await self.reconcile_final(event.content)
        elif isinstance(event, StreamRetry):
            await self._discard_attempt(event)
        elif isinstance(event, ToolCallStarted):
            self._tools[event.call.id] = (event.call.name, "running")
            await self._render_status()
        elif isinstance(event, ToolResultEvent):
            self._tools[event.result.call_id] = (
                event.result.name,
                "succeeded" if event.result.ok else "failed",
            )
            await self._render_status()
            for attachment in event.result.attachments:
                payload = base64.b64decode(attachment.data, validate=True)
                filename = attachment.name or "attachment"
                if attachment.kind == "image":
                    await self.sender.send_photo(
                        self.chat_id, payload, filename=filename
                    )
                else:
                    await self.sender.send_document(
                        self.chat_id, payload, filename=filename
                    )
        elif isinstance(event, PluginNotice):
            await self.sender.send_message(
                self.chat_id,
                f"Plugin {event.plugin} · {event.hook} · {event.action}: {event.message}",
            )
        elif isinstance(event, SkillActivated):
            action = "activated" if event.active else "deactivated"
            self._activity.append(f"skill {event.name}: {action}")
            await self._render_status()
        elif isinstance(event, Compacted):
            self._activity.append(
                f"context compacted: {event.summarized_messages} messages"
            )
            await self._render_status()
        elif isinstance(event, TodoUpdated):
            done = sum(1 for item in event.todos if item.status == "completed")
            current = next(
                (item.content for item in event.todos if item.status == "in_progress"),
                None,
            )
            summary = f"todos: {done}/{len(event.todos)} done"
            if current:
                summary += f" — now: {current}"
            self._activity.append(summary)
            await self._render_status()
        elif isinstance(event, TurnCancelled):
            await self._discard_for_terminal(f"⏹️ {event.reason}")
        elif isinstance(event, UsageReported):
            pass  # accounting only; never rendered into a chat
        elif isinstance(event, Error):
            await self._discard_for_terminal(f"❌ {event.message}")

    async def reconcile_final(self, content: str) -> None:
        """Make current-attempt messages exactly match authoritative Final text."""
        chunks = chunk_text(content or EMPTY_RESPONSE)
        for index, chunk in enumerate(chunks):
            if index < len(self._messages):
                await self._edit_response(index, chunk)
            else:
                sent = await self.sender.send_message(self.chat_id, chunk)
                self._messages.append(message_id(sent))
                self._sent_text.append(chunk)

        surplus = list(range(len(chunks), len(self._messages)))
        for index in reversed(surplus):
            mid = self._messages[index]
            try:
                await self.sender.delete_message(self.chat_id, mid)
            except Exception:
                await self._edit_response(index, SUPERSEDED_NOTICE)
        if surplus:
            del self._messages[len(chunks) :]
            del self._sent_text[len(chunks) :]
        self._text = content or EMPTY_RESPONSE

    async def _render_stream(self) -> None:
        chunks = chunk_text(self._text)
        if not chunks:
            return
        # A full active chunk is frozen immediately and followed by a fresh
        # placeholder. Final reconciliation removes that placeholder when the
        # authoritative answer ends exactly on the boundary.
        desired_slots = len(chunks) + (
            1 if len(chunks[-1]) == TELEGRAM_TEXT_LIMIT else 0
        )
        while len(self._messages) < desired_slots:
            sent = await self.sender.send_message(self.chat_id, STREAM_PLACEHOLDER)
            self._messages.append(message_id(sent))
            self._sent_text.append(STREAM_PLACEHOLDER)

        for index, chunk in enumerate(chunks[:-1]):
            await self._edit_response(index, chunk)

        active_index = len(chunks) - 1
        now = self._clock()
        active_filled = len(chunks[-1]) == TELEGRAM_TEXT_LIMIT
        if active_filled or now - self._last_active_edit >= self.edit_interval:
            await self._edit_response(active_index, chunks[-1])
            self._last_active_edit = now

    async def _edit_response(self, index: int, text: str) -> None:
        if self._sent_text[index] == text:
            return
        try:
            await self.sender.edit_message(self.chat_id, self._messages[index], text)
        except Exception as exc:
            if not _not_modified(exc):
                raise
        self._sent_text[index] = text

    async def _discard_attempt(self, event: StreamRetry) -> None:
        for index in range(len(self._messages)):
            await self._edit_response(index, DISCARDED_NOTICE)
        reason = " ".join(event.reason.split())[:500]
        await self.sender.send_message(
            self.chat_id,
            f"🔄 Response interrupted; retrying "
            f"({event.attempt}/{event.max_attempts})."
            + (f"\n{reason}" if reason else ""),
        )
        self._text = ""
        self._messages = []
        self._sent_text = []
        self._last_active_edit = float("-inf")
        sent = await self.sender.send_message(self.chat_id, STREAM_PLACEHOLDER)
        self._messages.append(message_id(sent))
        self._sent_text.append(STREAM_PLACEHOLDER)

    async def _render_status(self) -> None:
        lines = ["⚙️ Activity"]
        tools = list(self._tools.values())
        for name, state in tools[:20]:
            clean_name = " ".join(name.split())[:80] or "(unnamed tool)"
            lines.append(f"• {clean_name}: {state}")
        if len(tools) > 20:
            lines.append(f"• +{len(tools) - 20} more tool calls")
        lines.extend(
            f"• {' '.join(entry.split())[:120]}" for entry in self._activity[-4:]
        )
        text = "\n".join(lines)
        if text == self._status_text:
            return
        if self._status_message is None:
            sent = await self.sender.send_message(self.chat_id, text)
            self._status_message = message_id(sent)
        else:
            try:
                await self.sender.edit_message(self.chat_id, self._status_message, text)
            except Exception as exc:
                if not _not_modified(exc):
                    raise
        self._status_text = text

    async def _discard_for_terminal(self, notice: str) -> None:
        for index in range(len(self._messages)):
            await self._edit_response(index, DISCARDED_NOTICE)
        for chunk in chunk_text(notice):
            await self.sender.send_message(self.chat_id, chunk)
