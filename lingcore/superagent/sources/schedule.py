"""Deterministic interval triggers; checkpointed positions prevent duplicate ticks."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from lingcore.superagent.sources.base import bounded_trigger


class ScheduleSource:
    name = "schedule"

    def __init__(self, runtime: Any, config: Any) -> None:
        self.runtime, self.config = runtime, config
        self.task: asyncio.Task[None] | None = None

    async def poll(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        tick = int(now // self.config.interval_seconds)
        position = self.runtime.store.read().source_positions.get(self.name)
        if position is not None and int(position) >= tick:
            return
        options = self.config.options
        await self.runtime.ingest(
            bounded_trigger(
                source=self.name,
                external_id=str(tick),
                actor=options.get("requester", "scheduler"),
                thread="schedule:" + options.get("name", "default"),
                address={
                    "channel": options.get("channel", ""),
                    "chat_id": options.get("chat_id", ""),
                },
                text=options.get("prompt", "Scheduled check"),
                addressed=True,
                bot=False,
                timestamp=now,
            )
        )

    async def start(self) -> None:
        async def run() -> None:
            while True:
                await self.poll()
                await asyncio.sleep(min(60, self.config.interval_seconds))

        self.task = asyncio.create_task(run())

    async def close(self) -> None:
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def send(
        self,
        trigger: Any,
        text: str,
        *,
        key: str,
        approval: dict[str, Any] | None = None,
    ) -> str | None:
        channel = self.runtime.channels.get(trigger.address["channel"])
        if channel is None or channel is self:
            raise ValueError("schedule delivery channel is not configured")
        result = await channel.send(trigger, text, key=key, approval=approval)
        return str(result) if result is not None else None
