"""Background processes for the bundled ``process`` plugin.

``process_start`` runs a long-lived command (a dev server, a watcher) in the
same sandbox as ``run_shell`` and keeps it running across tool calls and turns;
``process_logs``/``process_stop``/``process_list`` inspect and end it. The
user-facing contract is the "Bundled process plugin" section of
``docs/plugins.md``.

- ``ProcessHooks`` is the per-Agent plugin instance and owns the registry, so
  the plugin must be enabled with ``plugins: [process]``. Every process is
  stopped on ``Agent.aclose``, at ``max_lifetime``, and when the turn that
  started it is rolled back (``turn_rolled_back``), because the model's record
  of starting it is voided with the turn.
- A configured Bubblewrap or OCI sandbox is required. A host process group can
  outlive a crashed LingCore; Bubblewrap's ``--die-with-parent`` and the OCI
  watchdog cannot.
- Output is drained continuously into a byte-bounded ring buffer of
  normalized lines: ANSI escapes stripped, carriage returns split, long lines
  cut, so ``ready_pattern`` and ``grep`` see what a person would read.
- A declared ``port`` needs the sandbox's network off: Bubblewrap, or a Linux
  OCI container on a Linux host. The command then runs under ``relay.py``,
  which LingCore hands one end of a fresh socketpair per browser connection;
  the browser plugin routes ``http://localhost:<port>`` through it. The server
  reaches no network, and the browser reaches exactly that port. Bubblewrap
  inherits the relay's control socket; an OCI container cannot, so its relay
  connects once to a private socket LingCore listens on (see ``Bridge``).
"""

from __future__ import annotations

import asyncio
import secrets
import shlex
import socket
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import regex

from lingcore.background import EventSource
from lingcore.bundled_plugins.process import actions
from lingcore.bundled_plugins.process.options import OPTION_PATH
from lingcore.bundled_plugins.process.options import IdArgs as IdArgs
from lingcore.bundled_plugins.process.options import LogsArgs as LogsArgs
from lingcore.bundled_plugins.process.options import MonitorArgs as MonitorArgs
from lingcore.bundled_plugins.process.options import NoArgs as NoArgs
from lingcore.bundled_plugins.process.options import ProcessOptions as ProcessOptions
from lingcore.bundled_plugins.process.options import StartArgs as StartArgs
from lingcore.bundled_plugins.process.options import (
    parse_process_options as parse_process_options,
)
from lingcore.bundled_plugins.process.runtime import Bridge as Bridge
from lingcore.bundled_plugins.process.runtime import LogBuffer as LogBuffer
from lingcore.bundled_plugins.process.runtime import Process as Process
from lingcore.bundled_plugins.process.runtime import (
    StreamPair,
    _abort,
    _accept_relay,
    _duration,
    _kill_group,
    _relay_listener,
    _retrieve,
    _search,
)
from lingcore.errors import ToolError
from lingcore.plugins import PluginContext, PluginHooks, TurnRolledBackEvent
from lingcore.sandbox import (
    OciSandbox,
    ReadOnlyMount,
    SandboxConfig,
    ShellExecution,
    ShellOptions,
    launch_shell,
)
from lingcore.tools import ToolContext, tool

GUEST_ENV = {"NO_COLOR": "1", "FORCE_COLOR": "0", "PYTHONUNBUFFERED": "1"}
GUEST_PYTHON = "/usr/bin/python3"
OCI_GUEST_PYTHON = "python3"
RELAY_TARGET = "/run/lingcore/relay.py"
CONTROL_TARGET = "/run/lingcore/bridge.sock"
LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_READ_CHUNK = 64 * 1024
_STOP_TIMEOUT = 10.0
_BRIDGE_TIMEOUT = 5.0


class ProcessHooks(PluginHooks):
    """Owns this Agent's background processes.

    One lock serializes registry changes. Waiting for readiness happens outside
    it, so ``process_logs`` works while a start is still waiting. Stop and close
    run as owned tasks that finish even when their caller is cancelled.
    """

    def __init__(self, ctx: PluginContext) -> None:
        super().__init__(ctx)
        self._processes: dict[str, Process] = {}  # insertion order = start order
        # Pending events can outlive a removed/pruned process record. Keep its
        # rollback ownership without retaining finished sources after delivery.
        self._monitor_sources: weakref.WeakKeyDictionary[EventSource, float] = (
            weakref.WeakKeyDictionary()
        )
        self._used_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._reaping: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False
        self._dropped_exited = 0

    # -- registry ------------------------------------------------------------

    @asynccontextmanager
    async def use(self) -> AsyncIterator[ProcessHooks]:
        async with self._lock:
            if self._reaping:
                # Processes from a rolled-back turn are still being killed; a
                # killed process must never be reported as running.
                await asyncio.shield(asyncio.gather(*self._reaping))
            if self._closed:
                raise ToolError("the process plugin is closed")
            yield self

    def get(self, process_id: str) -> Process:
        # Tolerate an id copied with the punctuation that follows it in results.
        proc = self._processes.get(process_id.strip().strip(":.,;"))
        if proc is None:
            raise ToolError(
                f"no background process {process_id!r}. Processes end when the "
                "agent restarts (a new or resumed session, an edited message) and "
                "when the turn that started them is stopped, so older ids no "
                "longer exist; process_list shows the current ones"
            )
        return proc

    def processes(self) -> list[Process]:
        return list(self._processes.values())

    def remove(self, proc: Process) -> None:
        self._processes.pop(proc.id, None)

    def prune(self, max_exited: int) -> None:
        exited = [p for p in self._processes.values() if p.state == "exited"]
        exited.sort(key=lambda p: p.ended or 0.0)
        for proc in exited[: max(0, len(exited) - max_exited)]:
            del self._processes[proc.id]
            self._dropped_exited += 1

    def take_dropped_note(self) -> str | None:
        if not self._dropped_exited:
            return None
        count, self._dropped_exited = self._dropped_exited, 0
        return f"({count} older exited process(es) were removed to stay within max_exited.)"

    def check_capacity(self, options: ProcessOptions, port: int | None) -> None:
        running = [p for p in self._processes.values() if p.state == "running"]
        if len(running) >= options.max_processes:
            raise ToolError(
                f"{len(running)} background processes are already running "
                f"({OPTION_PATH}.max_processes); stop one with process_stop first"
            )
        if port is not None:
            for proc in self._processes.values():
                if proc.port == port:
                    raise ToolError(
                        f"port {port} is already bridged for {proc.id}; "
                        "process_stop it first"
                    )

    def _new_id(self) -> str:
        while True:
            candidate = f"p-{secrets.token_hex(3)}"
            if candidate not in self._used_ids:
                self._used_ids.add(candidate)
                return candidate

    async def launch(
        self,
        ctx: ToolContext,
        options: ProcessOptions,
        sandbox: SandboxConfig,
        command: str,
        *,
        name: str | None,
        port: int | None,
        monitor_timeout: float | None = None,
    ) -> Process:
        """Start the command and register it. Callers hold the lock."""
        bridge: Bridge | None = None
        if port is None:
            execution = await launch_shell(
                command,
                workspace=ctx.workspace,
                options=ShellOptions(sandbox=sandbox),
                getenv=ctx.getenv,
                extra_env=GUEST_ENV,
            )
        elif isinstance(sandbox, OciSandbox):
            execution, bridge = await self._launch_oci_bridged(
                ctx, sandbox, command, port
            )
        else:
            relay = Path(__file__).with_name("relay.py").resolve()
            config = sandbox.model_copy(
                update={
                    "read_only_mounts": [
                        *sandbox.read_only_mounts,
                        ReadOnlyMount(source=str(relay), target=RELAY_TARGET),
                    ]
                }
            )
            host_end, guest_end = socket.socketpair()
            fd = guest_end.fileno()
            wrapped = (
                f"exec {GUEST_PYTHON} -I -S {RELAY_TARGET} {fd} {port} "
                f"{shlex.quote(command)}"
            )
            try:
                execution = await launch_shell(
                    wrapped,
                    workspace=ctx.workspace,
                    options=ShellOptions(sandbox=config),
                    getenv=ctx.getenv,
                    pass_fds=(fd,),
                    extra_env=GUEST_ENV,
                )
            except BaseException:
                host_end.close()
                raise
            finally:
                guest_end.close()
            bridge = Bridge(host_end, port, timeout=_BRIDGE_TIMEOUT)
        ident = self._new_id()
        source = EventSource(ctx.background, ident) if monitor_timeout else None
        proc = Process(
            id=ident,
            name=name,
            command=command,
            runner=execution.runner,
            port=port,
            execution=execution,
            log=LogBuffer(
                options.max_log_bytes,
                options.max_line_chars,
                source.publish if source else None,
            ),
            started=time.monotonic(),
            lifetime=min(monitor_timeout, options.max_lifetime)
            if monitor_timeout is not None
            else options.max_lifetime,
            bridge=bridge,
            source=source,
        )
        if source is not None:
            self._monitor_sources[source] = proc.started
        self._processes[proc.id] = proc
        proc.supervisor = asyncio.create_task(self._supervise(proc))
        return proc

    async def _launch_oci_bridged(
        self, ctx: ToolContext, sandbox: OciSandbox, command: str, port: int
    ) -> tuple[ShellExecution, Bridge]:
        """Run ``command`` under the relay in a container, bridged to ``port``.

        A container cannot inherit a descriptor, so the relay connects to a
        socket LingCore listens on, mounted into the container as one file.
        """
        relay = Path(__file__).with_name("relay.py").resolve()
        directory, listener = _relay_listener()
        config = sandbox.model_copy(
            update={
                "read_only_mounts": [
                    *sandbox.read_only_mounts,
                    ReadOnlyMount(source=str(relay), target=RELAY_TARGET),
                    ReadOnlyMount(
                        source=str(directory / "bridge.sock"), target=CONTROL_TARGET
                    ),
                ]
            }
        )
        pending = asyncio.create_task(_accept_relay(listener, directory))
        pending.add_done_callback(_retrieve)
        wrapped = (
            f"exec {OCI_GUEST_PYTHON} -I -S {RELAY_TARGET} {CONTROL_TARGET} {port} "
            f"{shlex.quote(command)}"
        )
        try:
            execution = await launch_shell(
                wrapped,
                workspace=ctx.workspace,
                options=ShellOptions(sandbox=config),
                getenv=ctx.getenv,
                extra_env=GUEST_ENV,
            )
        except BaseException:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            raise
        return execution, Bridge(None, port, pending=pending, timeout=_BRIDGE_TIMEOUT)

    async def _supervise(self, proc: Process) -> None:
        """Drain output until exit, enforcing the lifetime; then mark exited."""
        stdout = proc.execution.process.stdout
        code: int | None = None
        try:
            async with asyncio.timeout(proc.lifetime):
                if stdout is not None:
                    while chunk := await stdout.read(_READ_CHUNK):
                        proc.log.feed(chunk)
                        proc.changed.set()
                code = await proc.execution.finish()
        except TimeoutError:
            proc.note = f"stopped: reached max_lifetime ({_duration(proc.lifetime)})"
            await _abort(proc.execution)
            code = proc.execution.process.returncode
        except ToolError as exc:
            proc.note = f"exit status unavailable: {exc}"
            code = proc.execution.process.returncode
        finally:
            proc.log.close()
            if proc.bridge is not None:
                proc.bridge.close()
            proc.state = "exited"
            proc.exit_code = code if code is not None else proc.exit_code
            proc.ended = time.monotonic()
            proc.changed.set()
            if proc.source is not None:
                proc.source.publish(f"Monitor ended: {proc.note or proc.state_text()}")
                proc.source.finish()

    async def stop(self, proc: Process) -> None:
        """Kill and reap ``proc``; completes even if the caller is cancelled."""
        await asyncio.shield(asyncio.ensure_future(self._stop(proc)))

    async def _stop(self, proc: Process) -> None:
        # Stopping a live watch drops its queued events; an exited watch's
        # last lines and end notice are still delivered.
        if proc.source is not None and proc.state == "running":
            proc.source.cancel()
        if proc.bridge is not None:
            proc.bridge.close()
        if proc.state == "running":
            await _abort(proc.execution)
        supervisor = proc.supervisor
        if supervisor is not None and not supervisor.done():
            try:
                async with asyncio.timeout(_STOP_TIMEOUT):
                    await asyncio.shield(supervisor)
            except TimeoutError:
                supervisor.cancel()
                await asyncio.gather(supervisor, return_exceptions=True)
        if proc.exit_code is None:
            proc.exit_code = proc.execution.process.returncode

    async def wait_ready(
        self, proc: Process, pattern: regex.Pattern[str] | None, timeout: float
    ) -> tuple[int, str] | None:
        """The first log line matching ``pattern``; ``None`` on timeout or exit.

        Without a pattern this only waits (up to ``timeout``) for an exit.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        cursor = 1
        while True:
            if pattern is not None:
                for number, line in proc.log.lines_from(cursor):
                    if _search(pattern, line):
                        return number, line
                cursor = proc.log.next
            if proc.state == "exited":
                return None
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            # No await separates the scan from the clear, so no line is missed.
            proc.changed.clear()
            try:
                async with asyncio.timeout(remaining):
                    await proc.changed.wait()
            except TimeoutError:
                return None

    # -- browser bridge ------------------------------------------------------

    def bridge_opener(
        self, host: str, port: int
    ) -> Callable[[], Awaitable[StreamPair]] | None:
        """How the browser reaches ``host:port`` if a process owns it.

        A listed process with that port shadows host loopback for the agent's
        browser even after it exits (its bridge then refuses), so a stopped dev
        server never silently falls through to whatever else holds the port.
        """
        if host.rstrip(".").lower().strip("[]") not in LOOPBACK_NAMES:
            return None
        for proc in self._processes.values():
            if proc.port == port and proc.bridge is not None:
                return proc.bridge.open
        return None

    # -- lifecycle hooks -----------------------------------------------------

    def turn_rolled_back(self, event: TurnRolledBackEvent) -> None:
        for source, started in list(self._monitor_sources.items()):
            if started >= event.started_at:
                source.cancel()
        doomed = [
            proc
            for proc in self._processes.values()
            if proc.started >= event.started_at
        ]
        if not doomed:
            return
        for proc in doomed:
            del self._processes[proc.id]
            if proc.bridge is not None:
                proc.bridge.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop to reap in: at least kill the trees
            for proc in doomed:
                _kill_group(proc.execution)
            return
        task = loop.create_task(self._reap(doomed))
        self._reaping.add(task)
        task.add_done_callback(self._reaping.discard)

    async def _reap(self, doomed: list[Process]) -> None:
        for proc in reversed(doomed):
            await self._stop(proc)

    async def aclose(self) -> None:
        self._closed = True  # queued tool calls now refuse
        for source in list(self._monitor_sources):
            source.cancel()
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close_all())
        await asyncio.shield(self._close_task)

    async def _close_all(self) -> None:
        if self._reaping:
            await asyncio.gather(*self._reaping, return_exceptions=True)
        for proc in reversed(list(self._processes.values())):
            await self._stop(proc)
            self._processes.pop(proc.id, None)


def _hooks(ctx: ToolContext) -> ProcessHooks:
    hooks = ctx.plugins.get("process")
    if not isinstance(hooks, ProcessHooks):
        raise ToolError(
            "the process plugin is not enabled: add `process` to the "
            "profile's plugins: list so this agent gets its own process registry"
        )
    return hooks


@tool(
    name="process_start",
    high_risk=True,
    description=(
        "Start a long-running command (a dev server, a watcher) in the background, "
        "in the same sandbox as run_shell, and return its first log lines. Use "
        "run_shell for commands that finish (builds, tests). Declare port for a "
        "server the browser should open: the browser tools then reach it at "
        "http://localhost:<port>/ and nothing else can. Use ready_pattern to wait "
        "until it is up. Every start needs the user's confirmation; processes end "
        "with process_stop, when the agent restarts, or at a lifetime limit."
    ),
)
async def process_start(args: StartArgs, ctx: ToolContext) -> str:
    return await actions._start_process(
        args,
        ctx,
        _hooks(ctx),
        guest_python=GUEST_PYTHON,
        oci_guest_python=OCI_GUEST_PYTHON,
    )


@tool(
    name="process_monitor",
    high_risk=True,
    description=(
        "Watch events in the background: start a sandboxed command and deliver "
        "each output line to this agent without polling tools. Returns immediately. "
        "Use a script that filters/deduplicates changes (logs, files, CI status) "
        "and flushes output. Events queue while the agent is busy and wake it "
        "when idle, even with the web UI closed. Default deadline 5 minutes, "
        "maximum 30; one end notice. Uses process_start's permissions and limits. "
        "Inspect/stop with process_logs, process_list, process_stop."
    ),
)
async def process_monitor(args: MonitorArgs, ctx: ToolContext) -> str:
    if not ctx.background.enabled or ctx.background.closed:
        raise ToolError("this frontend does not support monitor event delivery")
    return await actions._start_process(
        StartArgs(command=args.command, name=args.name),
        ctx,
        _hooks(ctx),
        guest_python=GUEST_PYTHON,
        oci_guest_python=OCI_GUEST_PYTHON,
        monitor_timeout=args.timeout,
    )


@tool(
    name="process_logs",
    description=(
        "Read a background process's log. Pass since (the previous result's next "
        "cursor) to read only new lines, tail for the last N, grep to filter."
    ),
)
async def process_logs(args: LogsArgs, ctx: ToolContext) -> str:
    return await actions.process_logs(args, ctx, _hooks(ctx))


@tool(
    name="process_stop",
    description=(
        "Stop a background process (kill its whole tree) and return its exit "
        "status and last log lines. Also removes an exited process from the list."
    ),
)
async def process_stop(args: IdArgs, ctx: ToolContext) -> str:
    return await actions.process_stop(args, ctx, _hooks(ctx))


@tool(
    name="process_list",
    description="List this agent's background processes with their state and port.",
)
async def process_list(args: NoArgs, ctx: ToolContext) -> str:
    return await actions.process_list(args, ctx, _hooks(ctx))
