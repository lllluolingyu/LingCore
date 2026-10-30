"""One private Unix connection, with a short path on macOS as well as Linux."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from lingcore.evaluation.harbor.protocol import (
    MAX_FRAME_BYTES,
    Channel,
    Hello,
    ProtocolFailure,
)


class Listener:
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.directory = Path(tempfile.mkdtemp(prefix="lc-hb-", dir="/tmp"))
        os.chmod(self.directory, 0o700)
        self.path = self.directory / "s"
        self._server: asyncio.Server | None = None
        self._accepted: asyncio.Future[tuple[Channel, Hello]] = (
            asyncio.get_running_loop().create_future()
        )
        self._claimed = False
        self._channel: Channel | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(
            self._connect, path=self.path, limit=MAX_FRAME_BYTES
        )
        os.chmod(self.path, 0o600)

    def _connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._claimed:
            writer.close()
            return
        self._claimed = True
        assert self._server is not None
        self._server.close()
        self.path.unlink(missing_ok=True)
        task = asyncio.create_task(self._handshake(Channel(reader, writer)))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handshake(self, channel: Channel) -> None:
        try:
            async with asyncio.timeout(30):
                hello = await channel.receive()
            if not isinstance(hello, Hello) or hello.execution_id != self.execution_id:
                raise ProtocolFailure("bridge hello does not match this execution")
            self._channel = channel
            if not self._accepted.done():
                self._accepted.set_result((channel, hello))
        except (ProtocolFailure, TimeoutError) as exc:
            await channel.aclose()
            if not self._accepted.done():
                self._accepted.set_exception(
                    ProtocolFailure(
                        "bridge hello timed out"
                        if isinstance(exc, TimeoutError)
                        else str(exc)
                    )
                )
        except asyncio.CancelledError:
            await channel.aclose()
            raise

    async def accept(self, timeout: float) -> tuple[Channel, Hello]:
        try:
            async with asyncio.timeout(timeout):
                return await self._accepted
        except TimeoutError:
            raise ProtocolFailure("bridge hello timed out") from None

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
        for task in tuple(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._channel is not None:
            await self._channel.aclose()
        if self._server is not None:
            await self._server.wait_closed()
        if not self._accepted.done():
            self._accepted.cancel()
        elif not self._accepted.cancelled():
            self._accepted.exception()
        try:
            shutil.rmtree(self.directory)
        except FileNotFoundError:
            pass
