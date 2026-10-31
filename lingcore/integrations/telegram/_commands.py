"""Telegram slash-command dispatch and session selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lingcore.errors import SessionError
from lingcore.integrations.telegram.protocol import TelegramSender
from lingcore.sessions import new_session_id

if TYPE_CHECKING:
    from lingcore.integrations.telegram.bridge import TelegramBridge

_HELP = """LingCore Telegram

Send text, one photo, or one document.

/new — start a new session
/sessions — show your 10 most recent sessions
/resume <id-prefix> — switch to a stored session
/stop — cancel your active turn
/help — show this help"""


_RESERVED_COMMANDS = {"start", "help", "new", "sessions", "resume", "stop"}


async def _handle_command(
    self: TelegramBridge,
    command: str,
    argument: str,
    user_id: int,
    chat_id: int,
    sender: TelegramSender,
) -> None:
    if command in {"start", "help"}:
        extra = "\n".join(
            f"/{c.telegram_name} {c.argument_hint} — {c.description}"
            for c in self.commands.telegram_commands(reserved=_RESERVED_COMMANDS)
        )
        await sender.send_message(chat_id, _HELP + ("\n\n" + extra if extra else ""))
        return
    if command == "stop":
        await self._stop(user_id, chat_id, sender)
        return
    if command not in {"new", "sessions", "resume"}:
        await sender.send_message(chat_id, "Unknown command. Use /help.")
        return
    try:
        runtime = await self.runtime_for(user_id, chat_id, sender)
    except Exception as exc:
        await sender.send_message(
            chat_id,
            f"Could not start your LingCore runtime: {self._safe_runtime_error(exc)}",
        )
        return
    if command == "sessions":
        await self._sessions(runtime, chat_id, sender)
        return
    active = self._user_tasks.get(user_id)
    if active is not None and not active.done():
        await sender.send_message(
            chat_id, "Cannot switch sessions while a turn is running."
        )
        return
    if command == "new":
        await self._replace_session(
            runtime, new_session_id(), chat_id, sender, resumed=False
        )
    else:
        if not argument:
            await sender.send_message(chat_id, "Usage: /resume <session-id-prefix>")
            return
        try:
            meta = runtime.store.resolve_prefix(argument.lower())
        except SessionError as exc:
            await sender.send_message(chat_id, str(exc))
            return
        await self._replace_session(runtime, meta.id, chat_id, sender, resumed=True)
