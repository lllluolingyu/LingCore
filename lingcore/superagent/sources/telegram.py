"""Telegram event source and channel sharing the installed PTB transport."""

from __future__ import annotations

import time
from typing import Any

from lingcore.superagent.core.models import Trigger
from lingcore.superagent.outbox import VerifiedAction
from lingcore.superagent.sources.base import bounded_trigger


class TelegramSource:
    name = "telegram"

    def __init__(self, runtime: Any, application: Any) -> None:
        self.runtime, self.application = runtime, application
        self.username = ""

    def normalize(self, update: Any) -> Trigger | None:
        message = update.effective_message
        actor = update.effective_user
        if message is None or actor is None or actor.is_bot:
            return None
        text = message.text or message.caption or ""
        thread = getattr(message, "message_thread_id", None) or message.chat_id
        return bounded_trigger(
            source=self.name,
            external_id=str(update.update_id),
            actor=str(actor.id),
            thread=f"telegram:{message.chat_id}:{thread}",
            address={
                "chat_id": str(message.chat_id),
                "message_id": str(message.message_id),
            },
            text=text,
            addressed=message.chat.type == "private"
            or (bool(self.username) and f"@{self.username}" in text),
            bot=False,
            timestamp=time.time(),
        )

    async def handle_message(self, update: Any, context: Any = None) -> None:
        trigger = self.normalize(update)
        if trigger is not None:
            await self.runtime.ingest(trigger)

    async def handle_action(self, update: Any, context: Any = None) -> None:
        query = update.callback_query
        if query is None or not query.data or not query.data.startswith("lc:"):
            return
        _, token, decision = query.data.split(":", 2)
        if decision not in {"yes", "no"}:
            return
        await self.runtime.action(
            VerifiedAction(
                str(query.from_user.id),
                token,
                decision == "yes",
                self.name,
                str(query.message.chat_id),
            )
        )
        await query.answer()

    async def start(self) -> None:
        from telegram.ext import CallbackQueryHandler, MessageHandler, filters

        self.application.add_handler(
            CallbackQueryHandler(self.handle_action, pattern=r"^lc:")
        )
        self.application.add_handler(MessageHandler(filters.ALL, self.handle_message))
        await self.application.initialize()
        self.username = (await self.application.bot.get_me()).username or ""
        await self.application.start()
        await self.application.updater.start_polling()

    async def close(self) -> None:
        if self.application.updater.running:
            await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()

    async def send(
        self,
        trigger: Trigger,
        text: str,
        *,
        key: str,
        approval: dict[str, Any] | None = None,
    ) -> str:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        markup = (
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            label, callback_data=f"lc:{approval['token']}:{value}"
                        )
                        for label, value in [("Approve", "yes"), ("Deny", "no")]
                    ]
                ]
            )
            if approval
            else None
        )
        sent = await self.application.bot.send_message(
            chat_id=int(trigger.address["chat_id"]),
            text=text[:4000],
            reply_to_message_id=int(trigger.address["message_id"])
            if trigger.address.get("message_id")
            else None,
            reply_markup=markup,
        )
        return str(sent.chat_id)
