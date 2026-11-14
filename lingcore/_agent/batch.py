"""Own a tool-batch task while delivering bounded live progress."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from lingcore.events import ToolProgress
from lingcore.message import ToolCall, ToolResult
from lingcore.tools._progress import progress_sink

if TYPE_CHECKING:
    from lingcore.agent import Agent


class ToolBatch:
    def __init__(self) -> None:
        self.results: list[ToolResult] = []
        self._queue: asyncio.Queue[ToolProgress | None] = asyncio.Queue(maxsize=64)
        self._closed = False

    def publish(self, event: ToolProgress | None) -> None:
        if self._closed:
            return
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(event)

    async def progress(self) -> AsyncIterator[ToolProgress]:
        while (event := await self._queue.get()) is not None:
            yield event


@asynccontextmanager
async def tool_batch(agent: Agent, calls: list[ToolCall]) -> AsyncIterator[ToolBatch]:
    batch = ToolBatch()

    async def run() -> None:
        token = progress_sink.set(batch.publish)
        try:
            batch.results = await agent._dispatch(calls)
        finally:
            progress_sink.reset(token)
            batch.publish(None)

    task = asyncio.create_task(run())
    try:
        yield batch
        await task
    finally:
        batch._closed = True
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
