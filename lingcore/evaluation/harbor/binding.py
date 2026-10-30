"""Terminal requests and trial-scoped bounded captured output."""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from typing import Any
from uuid import uuid4

from lingcore.errors import ToolError
from lingcore.evaluation.harbor.peer import Peer
from lingcore.evaluation.harbor.protocol import Channel, Exec, ProtocolFailure, encode
from lingcore.terminal_options import TerminalOptions as TerminalOptions
from lingcore.tracing.content import redact_known_secrets


class HarborTerminal:
    binding_id = "harbor-terminal-v1"

    def __init__(
        self,
        channel: Channel | Peer,
        options: TerminalOptions | None = None,
        *,
        store_bytes: int = 32 * 1024 * 1024,
        secrets: tuple[str, ...] = (),
    ):
        self.peer = channel if isinstance(channel, Peer) else Peer(channel)
        self.options = options or TerminalOptions()
        self.store_bytes = store_bytes
        self.secrets = secrets
        self._outputs: OrderedDict[str, bytes] = OrderedDict()
        self._size = 0
        self._lock = asyncio.Lock()

    async def execute(
        self, command: str, cwd: str | None, timeout: float | None
    ) -> dict[str, Any]:
        if any(
            redact_known_secrets(value, self.secrets) != value
            for value in (command, cwd or "")
        ):
            raise ToolError("terminal request contains a private credential")
        async with self._lock:
            call_id = uuid4().hex
            request = Exec(
                call_id=call_id,
                command=command,
                cwd=cwd,
                timeout=min(timeout or self.options.timeout, self.options.max_timeout),
                capture_bytes=self.options.capture_bytes,
            )
            try:
                encode(request)
            except ProtocolFailure:
                raise ToolError("terminal command exceeds the wire limit") from None
            response = await self.peer.execute(request)
            try:
                data = base64.b64decode(response.output, validate=True)
            except ValueError:
                raise ProtocolFailure("invalid bridge output encoding") from None
            if (
                len(data) > request.capture_bytes
                or response.total_output_bytes is not None
                and response.total_output_bytes < len(data)
                or response.truncated
                != (
                    response.total_output_bytes is None
                    or response.total_output_bytes > len(data)
                )
            ):
                raise ProtocolFailure("bridge output exceeds capture bounds")
            display = data[: self.options.display_bytes]
            handle = None
            if len(data) > len(display):
                if len(data) <= self.store_bytes:
                    while self._size + len(data) > self.store_bytes:
                        _, removed = self._outputs.popitem(last=False)
                        self._size -= len(removed)
                    handle = uuid4().hex
                    self._outputs[handle] = data
                    self._size += len(data)
            return {
                "exit_code": response.exit_code,
                "output": display.decode("utf-8", errors="replace"),
                "duration": response.duration,
                "total_output_bytes": response.total_output_bytes,
                "truncated": response.truncated or len(data) > len(display),
                "timed_out": response.timed_out,
                "handle": handle,
            }

    def read_output(self, handle: str, offset: int, limit: int) -> dict[str, Any]:
        if offset < 0 or limit <= 0:
            raise ToolError("invalid terminal output range")
        data = self._outputs.get(handle)
        if data is None:
            raise ToolError("unknown or evicted terminal output handle")
        chunk = data[offset : offset + min(limit, self.options.display_bytes)]
        return {
            "output": chunk.decode("utf-8", errors="replace"),
            "offset": offset,
            "bytes": len(chunk),
            "captured_bytes": len(data),
            "eof": offset + len(chunk) >= len(data),
        }


class RefusingTerminal:
    binding_id = "harbor-terminal-v1"

    def __init__(self, options: TerminalOptions | None = None):
        self.options = options or TerminalOptions()

    async def execute(
        self, command: str, cwd: str | None, timeout: float | None
    ) -> dict[str, Any]:
        raise AssertionError("offline binding cannot execute commands")

    def read_output(self, handle: str, offset: int, limit: int) -> dict[str, Any]:
        raise AssertionError("offline binding cannot read output")
