"""Durable effect execution after commit, with bounded concurrency and backoff."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from lingcore.delegation.worker_run import owned_io
from lingcore.errors import BudgetExceeded
from lingcore.superagent.core.models import Effect, SlotsFull


class EffectRunner:
    def __init__(
        self, runtime: Any, handlers: dict[str, Callable[[Effect], Awaitable[None]]]
    ) -> None:
        self.runtime, self.handlers = runtime, handlers
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.slots = {
            kind: asyncio.Semaphore(
                runtime.config.executor.max_workers + 1
                if kind == "launch"
                else 1
                if kind in {"assessment", "reply"}
                else 4
            )
            for kind in handlers
        }
        self._next_tick = 0.0

    async def once(self) -> None:
        if time.monotonic() >= self._next_tick:
            await owned_io(
                self.runtime.command, lambda engine: engine.tick(time.time())
            )
            self._next_tick = time.monotonic() + 1
        state = await owned_io(self.runtime.store.read)
        ordered = sorted(state.effects.values(), key=lambda e: (e.created_at, e.id))
        for effect in ordered:
            if len(self.tasks) >= 128:
                break
            if (
                effect.status != "pending"
                or effect.due_at > time.time()
                or effect.id in self.tasks
            ):
                continue
            if effect.kind == "reply":
                earlier = [
                    e
                    for e in ordered
                    if e.kind == "reply"
                    and e.id != effect.id
                    and (e.created_at, e.id) < (effect.created_at, effect.id)
                    and e.status != "done"
                    and e.payload.get("ticket_id") == effect.payload.get("ticket_id")
                    and e.payload.get("recipient") == effect.payload.get("recipient")
                ]
                if earlier:
                    continue
            self.tasks[effect.id] = asyncio.create_task(self._run(effect))
        for key, task in list(self.tasks.items()):
            if task.done():
                await asyncio.gather(task, return_exceptions=True)
                del self.tasks[key]

    async def _run(self, effect: Effect) -> None:
        async with self.slots[effect.kind]:
            self.runtime.command(lambda e: e.claim_effect(effect.id))
            try:
                await self.handlers[effect.kind](effect)
            except SlotsFull:
                self.runtime.command(lambda e: e.defer_effect(effect.id, time.time()))
            except asyncio.CancelledError:
                self.runtime.command(
                    lambda e: e.finish_effect(effect.id, time.time(), "interrupted")
                )
                raise
            except (Exception, BudgetExceeded) as exc:
                error_name = type(exc).__name__
                self.runtime.command(
                    lambda e: e.finish_effect(effect.id, time.time(), error_name)
                )
            else:
                self.runtime.command(
                    lambda e: e.finish_effect(effect.id, time.time(), None)
                )

    async def close(self) -> None:
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
