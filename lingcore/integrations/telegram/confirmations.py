"""Concurrent, user-bound Telegram confirmation prompts."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from lingcore.integrations.telegram.protocol import (
    InlineButton,
    TelegramSender,
    message_id,
)
from lingcore.integrations.telegram.rendering import chunk_text

_LOG = logging.getLogger(__name__)
_CALLBACK_PREFIX = "lc"

Sleep = Callable[[float], Awaitable[None]]


@dataclass(slots=True)
class _Pending:
    id: str
    user_id: int
    chat_id: int
    sender: TelegramSender
    future: asyncio.Future[bool]
    message_id: int | None = None
    buttons_removed: bool = False


class ConfirmationManager:
    """Own independent approval futures for parallel risky tool calls."""

    def __init__(
        self,
        timeout: float,
        *,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.timeout = timeout
        self._sleep = sleep
        self._pending: dict[str, _Pending] = {}

    async def request(
        self,
        user_id: int,
        chat_id: int,
        command: str,
        sender: TelegramSender,
    ) -> bool:
        callback_id = self._new_id()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        pending = _Pending(
            id=callback_id,
            user_id=user_id,
            chat_id=chat_id,
            sender=sender,
            future=future,
        )
        self._pending[callback_id] = pending
        buttons = (
            (
                InlineButton("Approve", f"{_CALLBACK_PREFIX}:a:{callback_id}"),
                InlineButton("Deny", f"{_CALLBACK_PREFIX}:d:{callback_id}"),
            ),
        )
        timeout_task = asyncio.create_task(self._expire(callback_id))
        timeout_task.add_done_callback(self._timeout_done)
        try:
            chunks = chunk_text(
                f"Allow this high-risk action?\n\n{command or '(empty action)'}"
            )
            for chunk in chunks[:-1]:
                await sender.send_message(chat_id, chunk)
            sent = await sender.send_message(
                chat_id, chunks[-1], reply_markup=buttons
            )
            pending.message_id = message_id(sent)
            return await future
        finally:
            timeout_task.cancel()
            await asyncio.gather(timeout_task, return_exceptions=True)
            if not future.done():
                future.cancel()
            self._pending.pop(callback_id, None)
            await self._remove_buttons(pending)

    async def resolve_callback(
        self,
        *,
        callback_data: str | None,
        user_id: int | None,
        chat_id: int | None,
    ) -> tuple[str, bool]:
        """Resolve one callback and return ``(answer text, show_alert)``."""
        parsed = self._parse(callback_data)
        if parsed is None:
            return "This confirmation is no longer valid.", False
        action, callback_id = parsed
        pending = self._pending.get(callback_id)
        if pending is None or pending.future.done():
            return "This confirmation is stale.", False
        if user_id != pending.user_id or chat_id != pending.chat_id:
            return "This confirmation belongs to another user.", True
        pending.future.set_result(action == "a")
        await self._remove_buttons(pending)
        return ("Approved." if action == "a" else "Denied."), False

    async def deny_user(self, user_id: int) -> None:
        await self._deny(
            [pending for pending in self._pending.values() if pending.user_id == user_id]
        )

    async def deny_all(self) -> None:
        await self._deny(list(self._pending.values()))

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _new_id(self) -> str:
        while True:
            candidate = secrets.token_urlsafe(6)
            if candidate not in self._pending:
                return candidate

    @staticmethod
    def _parse(data: str | None) -> tuple[str, str] | None:
        if not data:
            return None
        parts = data.split(":")
        if len(parts) != 3 or parts[0] != _CALLBACK_PREFIX:
            return None
        if parts[1] not in {"a", "d"} or not parts[2]:
            return None
        return parts[1], parts[2]

    async def _expire(self, callback_id: str) -> None:
        try:
            await self._sleep(self.timeout)
        except asyncio.CancelledError:
            return
        pending = self._pending.get(callback_id)
        if pending is None or pending.future.done():
            return
        pending.future.set_result(False)
        await self._remove_buttons(pending)

    async def _deny(self, pending_items: list[_Pending]) -> None:
        for pending in pending_items:
            if not pending.future.done():
                pending.future.set_result(False)
            await self._remove_buttons(pending)

    async def _remove_buttons(self, pending: _Pending) -> None:
        if pending.buttons_removed or pending.message_id is None:
            return
        try:
            await pending.sender.edit_reply_markup(
                pending.chat_id,
                pending.message_id,
                reply_markup=None,
            )
        except Exception:
            _LOG.debug("failed to remove Telegram confirmation buttons", exc_info=True)
        else:
            pending.buttons_removed = True

    @staticmethod
    def _timeout_done(task: asyncio.Task[None]) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            _LOG.error(
                "Telegram confirmation timeout task failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
