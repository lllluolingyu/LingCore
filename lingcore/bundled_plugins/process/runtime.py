"""Bounded process logs, socket bridges, and shared lifecycle helpers."""

from __future__ import annotations

import asyncio
import codecs
import itertools
import os
import re
import signal
import socket
import tempfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import regex

from lingcore.background import EventSource
from lingcore.errors import ToolError
from lingcore.sandbox import (
    ShellExecution,
)

_REGEX_TIMEOUT = 0.05  # per line, so a pathological pattern cannot stall a tool
_ANSI = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])"
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_BREAK = re.compile(r"\r\n|\r|\n")
_LOCAL_URL = re.compile(
    r"\b(?:localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0):([0-9]{1,5})\b", re.IGNORECASE
)


class LogBuffer:
    """Normalized output lines, bounded in bytes, numbered from 1 forever.

    Bytes are decoded incrementally (a UTF-8 sequence may span chunks), lines
    end at ``\\n``, ``\\r\\n`` or a lone ``\\r``, escapes and control characters
    are removed, and a line longer than ``max_line_chars`` is cut with a
    marker. Old lines are dropped from the front once the byte cap is reached;
    their numbers are never reused.
    """

    def __init__(
        self,
        max_bytes: int,
        max_line_chars: int,
        on_line: Callable[[str], None] | None = None,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_line_chars = max_line_chars
        # Raw text kept per line before stripping: escapes inflate it.
        self._raw_cap = max_line_chars * 4
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._lines: deque[str] = deque()
        self._sizes: deque[int] = deque()
        self._bytes = 0
        self.first = 1  # number of the oldest retained line
        self.next = 1  # number the next stored line gets
        self._partial: list[str] = []
        self._partial_len = 0
        self._overlong = False
        self._after_cr = False
        self._closed = False
        self._on_line = on_line

    @property
    def total(self) -> int:
        """Lines stored so far, including dropped ones."""
        return self.next - 1

    def feed(self, data: bytes) -> None:
        self._text(self._decoder.decode(data))

    def close(self) -> None:
        """Flush a final unterminated line; later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        self._text(self._decoder.decode(b"", final=True))
        if self._partial or self._overlong:
            self._end_line(from_cr=False)

    def lines_from(self, start: int) -> list[tuple[int, str]]:
        """Retained ``(number, line)`` pairs from ``start`` (clamped) onward."""
        offset = max(start, self.first) - self.first
        return [
            (self.first + offset + index, line)
            for index, line in enumerate(itertools.islice(self._lines, offset, None))
        ]

    def _text(self, text: str) -> None:
        if self._after_cr and text.startswith("\n"):
            text = text[1:]  # the second half of a \r\n split across chunks
        self._after_cr = False
        position = 0
        for match in _BREAK.finditer(text):
            self._append(text[position : match.start()])
            self._end_line(from_cr=match.group() == "\r")
            position = match.end()
        self._append(text[position:])
        if text.endswith("\r"):
            self._after_cr = True

    def _append(self, piece: str) -> None:
        if not piece or self._overlong:
            return
        room = self._raw_cap - self._partial_len
        if len(piece) > room:
            piece = piece[:room]
            self._overlong = True
        self._partial.append(piece)
        self._partial_len += len(piece)

    def _end_line(self, *, from_cr: bool) -> None:
        raw = "".join(self._partial)
        overlong = self._overlong
        self._partial, self._partial_len, self._overlong = [], 0, False
        line = _CONTROL.sub("", _ANSI.sub("", raw)).rstrip()
        if len(line) > self._max_line_chars:
            line = line[: self._max_line_chars]
            overlong = True
        if overlong:
            line += " …[line truncated]"
        if not line and from_cr:
            return  # a progress bar redrawing its line
        size = len(line.encode("utf-8")) + 1
        self._lines.append(line)
        self._sizes.append(size)
        self._bytes += size
        self.next += 1
        if self._on_line is not None:
            self._on_line(line)
        while self._bytes > self._max_bytes and len(self._lines) > 1:
            self._lines.popleft()
            self._bytes -= self._sizes.popleft()
            self.first += 1


StreamPair = tuple[asyncio.StreamReader, asyncio.StreamWriter]


class Bridge:
    """The host end of one process's relay control socket.

    ``open`` hands the relay a fresh socketpair end and returns the connected
    stream once the relay reports that the server accepted the connection.

    Bubblewrap's relay inherits its end of a socketpair, so the control socket
    exists from the start. An OCI relay connects to a listening socket instead
    (``pending`` accepts it, see ``_accept_relay``); ``open`` waits for that
    connection.
    """

    def __init__(
        self,
        control: socket.socket | None,
        port: int,
        *,
        pending: asyncio.Task[socket.socket] | None = None,
        timeout: float = 5.0,
    ) -> None:
        if control is not None:
            control.setblocking(False)
        self._control = control
        self._timeout = timeout
        self._pending = pending
        self.port = port
        self.closed = False
        self._writers: set[asyncio.StreamWriter] = set()

    async def _connected(self) -> socket.socket:
        if self._control is not None:
            return self._control
        pending = self._pending
        assert pending is not None
        # wait() neither cancels the accept on timeout nor turns close()'s
        # cancellation of it into the caller's own cancellation.
        await asyncio.wait({pending}, timeout=self._timeout)
        if self.closed:
            raise ConnectionRefusedError("the background process has stopped")
        if not pending.done() or pending.cancelled() or pending.exception():
            raise ConnectionRefusedError("the relay has not connected to the bridge")
        if self._control is None:
            self._control = pending.result()
            self._control.setblocking(False)
        return self._control

    async def open(self) -> StreamPair:
        if self.closed:
            raise ConnectionRefusedError("the background process has stopped")
        control = await self._connected()
        host, guest = socket.socketpair()
        try:
            try:
                socket.send_fds(control, [b"c"], [guest.fileno()])
            except BlockingIOError:
                raise ConnectionRefusedError("the relay is not accepting") from None
        except BaseException:
            host.close()
            raise
        finally:
            guest.close()
        reader, writer = await asyncio.open_connection(sock=host)
        try:
            async with asyncio.timeout(self._timeout):
                status = await reader.readexactly(1)
        except (asyncio.IncompleteReadError, TimeoutError):
            writer.close()
            raise ConnectionRefusedError("the relay did not answer") from None
        except BaseException:
            writer.close()
            raise
        if status != b"\x00":
            writer.close()
            raise ConnectionRefusedError(f"nothing is listening on port {self.port}")
        self._writers = {w for w in self._writers if not w.is_closing()}
        self._writers.add(writer)
        return reader, writer

    def close(self) -> None:
        """Refuse new connections and drop the open ones."""
        if self.closed:
            return
        self.closed = True
        pending = self._pending
        if pending is not None and not pending.done():
            pending.cancel()  # its finally removes the listener
        elif (
            pending is not None and not pending.cancelled() and not pending.exception()
        ):
            pending.result().close()
        if self._control is not None:
            self._control.close()
        for writer in self._writers:
            writer.transport.abort()
        self._writers.clear()


def _relay_listener() -> tuple[Path, socket.socket]:
    """A listening socket for one OCI relay, in a fresh private directory.

    The directory is 0700, so no other host user can reach the socket; the
    socket itself is world-connectable because only it is mounted into the
    container, where the image's user may have any uid.
    """
    directory = Path(tempfile.mkdtemp(prefix="lingcore-bridge-"))
    path = directory / "bridge.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        os.chmod(path, 0o666)
        listener.listen(1)
        listener.setblocking(False)
    except BaseException:
        listener.close()
        _remove_listener_dir(directory)
        raise
    return directory, listener


async def _accept_relay(listener: socket.socket, directory: Path) -> socket.socket:
    """Accept the relay's one connection, then stop listening for good.

    The relay connects before it starts the command, so the first connection
    is always the relay's; closing the listener right after it means the
    command (or anything else) can never connect in its place.
    """
    try:
        control, _ = await asyncio.get_running_loop().sock_accept(listener)
    finally:
        listener.close()
        _remove_listener_dir(directory)
    return control


def _remove_listener_dir(directory: Path) -> None:
    try:
        (directory / "bridge.sock").unlink(missing_ok=True)
        directory.rmdir()
    except OSError:
        pass


def _retrieve(task: asyncio.Task[socket.socket]) -> None:
    # A relay that never connects leaves the accept cancelled or failed; mark
    # the outcome retrieved so asyncio does not log it as unhandled.
    if not task.cancelled():
        task.exception()


@dataclass(eq=False)
class Process:
    id: str
    name: str | None
    command: str
    runner: str
    port: int | None
    execution: ShellExecution
    log: LogBuffer
    started: float  # time.monotonic()
    lifetime: float
    bridge: Bridge | None = None
    state: Literal["running", "exited"] = "running"
    exit_code: int | None = None
    ended: float | None = None
    note: str | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    supervisor: asyncio.Task[None] | None = None
    source: EventSource | None = None

    def state_text(self) -> str:
        if self.state == "running":
            return "running"
        code = "unknown" if self.exit_code is None else str(self.exit_code)
        return f"exited (code {code})"


async def _abort(execution: ShellExecution) -> None:
    try:
        await asyncio.shield(execution.abort())
    except ToolError:
        # OCI removal failures leave the watchdog armed; nothing else to do.
        pass


def _kill_group(execution: ShellExecution) -> None:
    try:
        os.killpg(execution.process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _search(pattern: regex.Pattern[str], line: str) -> bool:
    try:
        return pattern.search(line, timeout=_REGEX_TIMEOUT) is not None
    except TimeoutError:
        return False


def _compile(pattern: str, label: str) -> regex.Pattern[str]:
    try:
        return regex.compile(pattern, regex.VERSION0)
    except regex.error as exc:
        raise ToolError(f"invalid {label} regex: {exc}") from None


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s" if seconds else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _label(proc: Process) -> str:
    return f"{proc.id} ({proc.name})" if proc.name else proc.id


def _url(port: int) -> str:
    return f"http://localhost:{port}/"


def _announced_ports(lines: list[tuple[int, str]]) -> set[int]:
    return {
        int(match.group(1))
        for _, line in lines
        for match in _LOCAL_URL.finditer(line)
        if 0 < int(match.group(1)) < 65536
    }


def _render_lines(entries: list[tuple[int, str]], *, numbered: bool) -> str:
    if numbered:
        return "\n".join(f"{number}| {line}" for number, line in entries)
    return "\n".join(line for _, line in entries)
