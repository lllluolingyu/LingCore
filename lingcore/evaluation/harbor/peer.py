"""One reader owns the wire, including unsolicited harness cancellation."""

from __future__ import annotations

import asyncio

from lingcore.evaluation.harbor.protocol import (
    Abort,
    Channel,
    Exec,
    ExecResult,
    Finish,
    Finished,
    Frame,
    Interrupt,
    ProtocolFailure,
)


class Peer:
    def __init__(self, channel: Channel):
        self.channel = channel
        self.signal: asyncio.Future[Abort | ProtocolFailure] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending: dict[str, asyncio.Future[ExecResult]] = {}
        self._finished: asyncio.Future[Finished] | None = None
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            while True:
                frame = await self.channel.receive()
                if isinstance(frame, Abort):
                    self._fail(frame)
                    return
                if isinstance(frame, ExecResult) and frame.call_id in self._pending:
                    future = self._pending.pop(frame.call_id)
                    if not future.done():
                        future.set_result(frame)
                elif (
                    isinstance(frame, Finished)
                    and self._finished is not None
                    and not self._finished.done()
                ):
                    self._finished.set_result(frame)
                    return
                else:
                    raise ProtocolFailure("unexpected bridge response")
        except ProtocolFailure as exc:
            self._fail(exc)
        except Exception:
            self._fail(ProtocolFailure("bridge reader failed"))

    def _fail(self, signal: Abort | ProtocolFailure) -> None:
        if not self.signal.done():
            self.signal.set_result(signal)
        failure = (
            signal
            if isinstance(signal, ProtocolFailure)
            else ProtocolFailure("bridge aborted: " + signal.reason)
        )
        for future in [*self._pending.values(), self._finished]:
            if future is not None and not future.done():
                future.set_exception(failure)
                future.exception()  # The owner may already have been cancelled.

    async def send(self, frame: Frame) -> None:
        await self.channel.send(frame)

    async def execute(self, request: Exec) -> ExecResult:
        if self.signal.done():
            raise ProtocolFailure("bridge is no longer accepting commands")
        future: asyncio.Future[ExecResult] = asyncio.get_running_loop().create_future()
        self._pending[request.call_id] = future
        try:
            await self.send(request)
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Stop only this command. Drain its reply before allowing finish or
            # another call; the reader remains active throughout cancellation.
            async def interrupt() -> None:
                async with asyncio.timeout(30):
                    await self.send(Interrupt(call_id=request.call_id))
                    await asyncio.shield(future)

            cleanup = asyncio.create_task(interrupt())
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        # Keep the call registered until its reply is drained,
                        # even if an outer timeout cancels Stop a second time.
                        continue
                cleanup.result()
            except (ProtocolFailure, TimeoutError):
                self._fail(ProtocolFailure("command interruption failed"))
            raise
        finally:
            self._pending.pop(request.call_id, None)

    async def finish(self, frame: Finish) -> Finished:
        if self.signal.done():
            raise ProtocolFailure("bridge ended before finish")
        self._finished = asyncio.get_running_loop().create_future()
        await self.send(frame)
        return await asyncio.shield(self._finished)

    async def aclose(self) -> None:
        self._reader.cancel()
        await asyncio.gather(self._reader, return_exceptions=True)
        self._fail(ProtocolFailure("bridge connection closed"))
        await self.channel.aclose()
