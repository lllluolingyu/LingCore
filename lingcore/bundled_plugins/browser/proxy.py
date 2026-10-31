"""Vetted SOCKS proxy and sandbox port-bridge protocols."""

from __future__ import annotations

import asyncio
import functools
import ipaddress
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse

from lingcore.errors import ToolError

if TYPE_CHECKING:
    pass


_SOCKS_HANDSHAKE_TIMEOUT = 10.0
_RELAY_CHUNK = 64 * 1024
_SOCKS_OK, _SOCKS_NOT_ALLOWED, _SOCKS_UNREACHABLE = 0, 2, 4
_SOCKS_BAD_COMMAND, _SOCKS_BAD_ADDRESS = 7, 8
StreamPair = tuple[asyncio.StreamReader, asyncio.StreamWriter]
BridgeOpener = Callable[[], Awaitable[StreamPair]]


class PortBridges(Protocol):
    """A plugin that serves some ``host:port`` targets itself (e.g. ``process``).

    ``bridge_opener`` returns a connector for a target it owns, else ``None``.
    The browser uses it only for a browser it launched: the connector replaces
    the proxy's TCP connection, so the target needs no private-host approval.
    """

    def bridge_opener(self, host: str, port: int) -> BridgeOpener | None: ...


def _normal_host(host: str) -> str:
    return host.rstrip(".").lower()


def _http_form(url: str) -> str | None:
    """``url`` with ws/wss mapped to http/https; ``None`` for any other scheme."""
    parsed = urlparse(url)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    if scheme not in ("http", "https"):
        return None
    return parsed._replace(scheme=scheme).geturl()


async def _relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy one direction of a tunnel, passing on a half-close."""
    try:
        while data := await reader.read(_RELAY_CHUNK):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except OSError:
        writer.transport.abort()


class _VettingProxy:
    """A loopback SOCKS5 proxy that applies the network policy per connection.

    Chromium sends it unresolved hostnames, so ``vet`` sees every host the
    browser connects to — including redirect targets Playwright's routing
    never reports — and returns the address to pin, or ``None`` to refuse.
    ``bridge`` is consulted first: a target it owns is connected through its
    connector instead of TCP (a sandboxed dev server's port bridge).
    """

    def __init__(
        self,
        vet: Callable[[str, int], Awaitable[str | None]],
        connect_timeout: float,
        bridge: Callable[[str, int], BridgeOpener | None] | None = None,
    ) -> None:
        self._vet = vet
        self._connect_timeout = connect_timeout
        self._bridge = bridge
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> str:
        """Listen on an ephemeral loopback port; return the proxy URL."""
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"socks5://127.0.0.1:{port}"

    async def close(self) -> None:
        """Stop listening and drop every tunnel; safe to retry if cancelled."""
        if self._server is not None:
            self._server.close()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        upstream: asyncio.StreamWriter | None = None
        try:
            async with asyncio.timeout(_SOCKS_HANDSHAKE_TIMEOUT):
                target = await self._handshake(reader, writer)
            if target is None:
                return
            host, port = target
            opener = self._bridge(host, port) if self._bridge is not None else None
            if opener is None:
                code = _SOCKS_NOT_ALLOWED
                try:
                    address = await self._vet(host, port)
                except ToolError:  # unresolvable
                    address, code = None, _SOCKS_UNREACHABLE
                if address is None:
                    await self._reply(writer, code)
                    return
                opener = functools.partial(asyncio.open_connection, address, port)
            try:
                async with asyncio.timeout(self._connect_timeout):
                    upstream_reader, upstream = await opener()
            except (OSError, TimeoutError):
                await self._reply(writer, _SOCKS_UNREACHABLE)
                return
            await self._reply(writer, _SOCKS_OK)
            await asyncio.gather(
                _relay(reader, upstream), _relay(upstream_reader, writer)
            )
        except (OSError, TimeoutError, asyncio.IncompleteReadError):
            pass
        finally:
            for stream in (writer, upstream):
                if stream is not None:
                    stream.close()

    async def _handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> tuple[str, int] | None:
        """Read a no-auth SOCKS5 CONNECT; ``None`` after refusing anything else."""
        version, count = await reader.readexactly(2)
        methods = await reader.readexactly(count)
        if version != 5 or 0 not in methods:
            writer.write(b"\x05\xff")
            await writer.drain()
            return None
        writer.write(b"\x05\x00")
        await writer.drain()
        version, command, _, kind = await reader.readexactly(4)
        host: str | None = None
        if kind == 1:
            host = str(ipaddress.IPv4Address(await reader.readexactly(4)))
        elif kind == 4:
            host = str(ipaddress.IPv6Address(await reader.readexactly(16)))
        elif kind == 3:
            name = await reader.readexactly((await reader.readexactly(1))[0])
            if name.isascii():  # Chromium sends IDNA (punycode) names
                host = name.decode("ascii")
        port = int.from_bytes(await reader.readexactly(2), "big")
        if version != 5 or command != 1:
            await self._reply(writer, _SOCKS_BAD_COMMAND)
            return None
        if not host:
            await self._reply(writer, _SOCKS_BAD_ADDRESS)
            return None
        return host, port

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, code: int) -> None:
        writer.write(bytes([5, code, 0, 1, 0, 0, 0, 0, 0, 0]))
        await writer.drain()
