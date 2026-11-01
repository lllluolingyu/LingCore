"""Expiring, one-use approvals bound to the requesting user and chat."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass

from lingcore.integrations.lark.protocol import LarkMessage, LarkSender


@dataclass
class Pending:
    scope: tuple[str, str]
    future: asyncio.Future[bool]


class Confirmations:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.pending: dict[str, Pending] = {}

    async def request(
        self, message: LarkMessage, command: str, sender: LarkSender
    ) -> bool:
        token = secrets.token_hex(8)
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.pending[token] = Pending(message.scope, future)
        try:
            # Never approve a truncated command: the user must see the whole request.
            await sender.send_text(
                message,
                f"Approval requested by {message.user_id}:\n\n{command}\n\n"
                f"Reply /approve {token} or /deny {token} within {self.timeout:g}s. "
                "In a group, mention the bot in your reply.",
            )
            try:
                async with asyncio.timeout(self.timeout):
                    return await future
            except TimeoutError:
                await sender.send_text(message, "Approval expired; action denied.")
                return False
        finally:
            self.pending.pop(token, None)
            if not future.done():
                future.cancel()

    def resolve(self, message: LarkMessage, token: str, *, approve: bool) -> bool:
        pending = self.pending.get(token)
        if pending is None or pending.scope != message.scope or pending.future.done():
            return False
        pending.future.set_result(approve)
        return True

    def deny(self, scope: tuple[str, str] | None = None) -> None:
        for pending in self.pending.values():
            if (scope is None or scope == pending.scope) and not pending.future.done():
                pending.future.set_result(False)
