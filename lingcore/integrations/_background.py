"""Owned inbox pumps for transports that can confirm while a user is idle."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import Any

from lingcore.events import Error, Final


def bind_background(bridge: Any, runtime: Any, channel: str) -> None:
    agent = runtime.agent
    inbox = getattr(agent, "background", None)
    if inbox is None:
        return
    inbox.enabled = True
    agent.tool_ctx.supports_async_confirm = True
    if agent.tool_ctx.delegation is not None:
        agent.tool_ctx.delegation.broker.async_confirm = True
    if not hasattr(bridge, "_inbox_pumps"):
        bridge._inbox_pumps = set()

    async def drive(incoming: Any) -> None:
        if channel == "telegram":
            from lingcore.integrations.telegram.rendering import TelegramTurnRenderer

            renderer = TelegramTurnRenderer(
                bridge._senders[runtime.user_id],
                bridge._chat_ids[runtime.user_id],
                edit_interval=bridge.config.stream_edit_interval,
            )
            render = renderer.handle
            runtime.renderer = renderer
        else:

            async def render(event: Any) -> None:
                await bridge._deliver(event, runtime.message)

        try:
            async with aclosing(agent.run(incoming)) as turn:
                async for event in turn:
                    if isinstance(event, Final | Error):
                        runtime.final_delivery = True
                    await render(event)
        finally:
            runtime.final_delivery = False
            if agent.turn_pending_finalization:
                event = agent.finalize_cancelled_turn()
                for notice in agent.drain_plugin_notices():
                    await render(notice)
                agent.drain_usage()
                await render(event)

    async def pump() -> None:
        while not bridge._closing and not inbox.closed and runtime.agent is agent:
            await inbox.wait()
            if inbox.closed:
                break
            lock = runtime.lock if channel == "telegram" else bridge._dispatch_lock
            async with lock:
                active = (
                    bridge._user_tasks.get(runtime.user_id)
                    if channel == "telegram"
                    else runtime.task
                )
                if (
                    not bridge._closing
                    and not inbox.paused
                    and (active is None or active.done())
                ):
                    incoming = inbox.take()
                    if incoming is not None:
                        task = asyncio.create_task(drive(incoming))
                        bridge._tasks.add(task)
                        if channel == "telegram":
                            bridge._user_tasks[runtime.user_id] = task
                            task.add_done_callback(
                                lambda done: bridge._task_done(runtime.user_id, done)
                            )
                        else:
                            runtime.task = task
                            task.add_done_callback(bridge._task_done)
            await asyncio.sleep(0.05)

    task = asyncio.create_task(pump())
    bridge._inbox_pumps.add(task)
    task.add_done_callback(bridge._inbox_pumps.discard)


async def close_background(bridge: Any) -> None:
    tasks = list(getattr(bridge, "_inbox_pumps", ()))
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
