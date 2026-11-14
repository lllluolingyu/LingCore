"""Lark source/channel over the shared SDK transport, without per-user Agents."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from lingcore.errors import ToolError
from lingcore.superagent.core.models import Trigger
from lingcore.superagent.outbox import VerifiedAction
from lingcore.superagent.sources.base import bounded_trigger


class LarkSource:
    name = "lark"

    def __init__(self, runtime: Any, channel: Any) -> None:
        self.runtime, self.channel = runtime, channel
        self.loop = asyncio.get_running_loop()

    def normalize(self, raw: Any) -> Trigger | None:
        if raw.sender_type != "user":
            return None
        thread = getattr(raw, "thread_id", None) or raw.chat_id
        return bounded_trigger(
            source=self.name,
            external_id=raw.message_id,
            actor=raw.sender_id,
            thread=f"lark:{thread}",
            address={"chat_id": raw.chat_id, "message_id": raw.message_id},
            text=raw.body_text or "",
            addressed=raw.chat_type == "p2p" or raw.mentioned_bot,
            bot=False,
            timestamp=time.time(),
        )

    async def handle_message(self, raw: Any) -> None:
        trigger = self.normalize(raw)
        if trigger is not None:
            await self.runtime.ingest(trigger)

    async def handle_action(self, raw: Any) -> None:
        value = raw.action.value
        if not isinstance(value, dict) or value.get("kind") != "lingcore_approval":
            return
        if type(value.get("allow")) is not bool:
            return
        await self.runtime.action(
            VerifiedAction(
                raw.operator.open_id,
                value["token"],
                value["allow"],
                self.name,
                raw.chat_id,
            )
        )

    async def _handoff(self, fn: Any, value: Any) -> None:
        if asyncio.get_running_loop() is self.loop:
            await fn(value)
        else:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(fn(value), self.loop)
            )

    async def start(self) -> None:
        self.channel.on("message", lambda raw: self._handoff(self.handle_message, raw))
        self.channel.on(
            "cardAction", lambda raw: self._handoff(self.handle_action, raw)
        )
        await self.channel.connect_until_ready()

    async def close(self) -> None:
        await self.channel.disconnect()

    async def send(
        self,
        trigger: Trigger,
        text: str,
        *,
        key: str,
        approval: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any]
        if approval:
            payload = {
                "card": {
                    "config": {"wide_screen_mode": True},
                    "elements": [
                        {"tag": "div", "text": {"tag": "plain_text", "content": text}},
                        {
                            "tag": "action",
                            "actions": [
                                {
                                    "tag": "button",
                                    "text": {"tag": "plain_text", "content": label},
                                    "type": style,
                                    "value": {
                                        "kind": "lingcore_approval",
                                        "token": approval["token"],
                                        "allow": allow,
                                    },
                                }
                                for label, style, allow in [
                                    ("Approve", "primary", True),
                                    ("Deny", "danger", False),
                                ]
                            ],
                        },
                    ],
                }
            }
        else:
            payload = {"text": text}
        result = await self.channel.send(
            trigger.address["chat_id"],
            payload,
            {
                "reply_to": trigger.address.get("message_id"),
                "uuid": key,
                **(
                    {"receive_id_type": "open_id"}
                    if trigger.address.get("recipient")
                    else {}
                ),
            },
        )
        if not result.success:
            raise ToolError("Lark delivery failed")
        conversation = (
            (getattr(result, "raw", None) or {}).get("data", {}).get("chat_id")
        )
        return str(
            conversation
            or ("" if trigger.address.get("recipient") else trigger.address["chat_id"])
        )
