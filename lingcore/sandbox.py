"""Sandbox backends for :mod:`lingcore.tools.builtin.shell`.

The shell tool deliberately keeps its confirmation and output policy separate
from process isolation.  This module owns the typed backend configuration and
turns one command into a supervised process.  A missing sandbox configuration
means the legacy host runner; a configured backend never falls back to it.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lingcore._sandbox_bubblewrap import (
    BUBBLEWRAP_HOME,
    _alias_guest_environment,
    _bubblewrap_argv,
    bubblewrap_system_mount_args,
)
from lingcore._sandbox_oci import (
    OCI_IMAGE_INSPECT_FORMAT,
    OciImageMetadata,
    _create_oci_environment_file,
    _discard_environment_file,
    _oci_create_argv,
    parse_oci_image_metadata,
)
from lingcore.errors import ToolError
from lingcore.sandbox_config import (
    _CONTAINER_NAME,
    _ENV_NAME,
    DEFAULT_MAX_CAPTURE_BYTES,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_SHELL_TIMEOUT,
    BubblewrapSandbox,
    OciResources,
    OciSandbox,
    ReadOnlyMount,
    SandboxConfig,
    ShellOptions,
)

__all__ = [
    "BUBBLEWRAP_HOME",
    "BubblewrapSandbox",
    "DEFAULT_MAX_CAPTURE_BYTES",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_SHELL_TIMEOUT",
    "OCI_IMAGE_INSPECT_FORMAT",
    "OciImageMetadata",
    "OciResources",
    "OciSandbox",
    "ReadOnlyMount",
    "SandboxConfig",
    "ShellOptions",
    "bubblewrap_system_mount_args",
    "parse_oci_image_metadata",
]

_CONTROL_OUTPUT_BYTES = 256 * 1024


@dataclass(slots=True)
class ShellExecution:
    """A running command plus backend-specific supervision hooks."""

    process: asyncio.subprocess.Process
    runner: str
    _exit_code: Callable[[], Awaitable[int]]
    _cleanup: Callable[[bool], Awaitable[None]]

    async def finish(self) -> int:
        """Obtain the command exit code and release backend resources."""
        try:
            code = await self._exit_code()
        except BaseException:
            try:
                await asyncio.shield(self._cleanup(False))
            except ToolError:
                # Cleanup is secondary: preserve cancellation or the exit-code
                # diagnostic. OCI cleanup has already armed its watchdog.
                pass
            raise
        else:
            await asyncio.shield(self._cleanup(False))
            return code

    async def abort(self) -> None:
        """Kill the command tree/container and release backend resources."""
        await self._cleanup(True)


@dataclass(frozen=True, slots=True)
class _ControlResult:
    returncode: int
    output: bytes


@dataclass(slots=True)
class _Watchdog:
    process: subprocess.Popen[bytes]

    def disarm(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.write(b"D")
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def trigger(self) -> None:
        """Close without disarming so the watchdog performs forced removal."""
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass


def parse_shell_options(raw: Mapping[str, Any] | object) -> ShellOptions:
    """Validate raw tool options for profile loading and direct tool calls."""
    if not isinstance(raw, Mapping):
        raise ValueError("tool_options.run_shell must be a mapping")
    return ShellOptions.model_validate(raw)


def sandbox_environment_names(options: ShellOptions) -> tuple[str, ...]:
    if options.sandbox is None:
        return ()
    return tuple(options.sandbox.pass_env)


async def launch_shell(
    command: str,
    *,
    workspace: Path,
    options: ShellOptions,
    getenv: Callable[[str], str | None],
    pass_fds: Sequence[int] = (),
    extra_env: Mapping[str, str] | None = None,
) -> ShellExecution:
    """Launch a command with the selected runner, without fallback.

    Every runner gives the command ``/dev/null`` as stdin: an inherited stdin
    is the frontend's terminal, which a command (or a dev server's keyboard
    shortcuts) would otherwise read from and switch to raw mode.

    ``pass_fds`` keeps those descriptors open, at the same numbers, in the
    Bubblewrap guest; no other runner can carry them. ``extra_env`` holds
    fixed, non-secret guest variables (e.g. ``NO_COLOR``); secrets go through
    the sandbox's ``pass_env``.
    """
    sandbox = options.sandbox
    if pass_fds and not isinstance(sandbox, BubblewrapSandbox):
        raise ToolError("only the bubblewrap runner can pass descriptors")
    extra = dict(extra_env or {})
    for name, value in extra.items():
        if not _ENV_NAME.fullmatch(name) or "\x00" in value or "\n" in value:
            raise ToolError(f"invalid extra environment variable {name!r}")
    if sandbox is None:
        return await _launch_host(command, workspace, extra)
    passed = _passed_environment(sandbox.pass_env, getenv)
    if isinstance(sandbox, BubblewrapSandbox):
        return await _launch_bubblewrap(
            command,
            workspace,
            sandbox,
            passed,
            pass_fds=tuple(pass_fds),
            extra_env=extra,
        )
    return await _launch_oci(command, workspace, sandbox, passed, extra_env=extra)


def _passed_environment(
    names: list[str], getenv: Callable[[str], str | None]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in names:
        value = getenv(name)
        if value is None:
            raise ToolError(
                f"sandbox pass_env names {name!r}, but it is not set in the "
                "profile or process environment"
            )
        values[name] = value
    return values


def _process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        # The constant is defined only on Windows, so getattr keeps imports and
        # static checking portable while retaining the documented flag value.
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {"start_new_session": True}


def resolve_sandbox_executable(executable: str, workspace: Path, label: str) -> Path:
    """Resolve a trusted backend binary outside the writable workspace."""
    path = Path(executable)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ToolError(f"{label} executable is unavailable at {path}: {exc}") from None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ToolError(f"{label} executable is not an executable file: {resolved}")
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        pass
    else:
        raise ToolError(f"{label} executable may not be inside the writable workspace")
    return resolved


async def _launch_host(
    command: str, workspace: Path, extra_env: Mapping[str, str] | None = None
) -> ShellExecution:
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace),
            env={**os.environ, **extra_env} if extra_env else None,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **_process_group_kwargs(),
        )
    except OSError as exc:
        raise ToolError(f"failed to launch command: {exc}") from None

    async def exit_code() -> int:
        return (
            process.returncode
            if process.returncode is not None
            else await process.wait()
        )

    async def cleanup(abort: bool) -> None:
        if abort:
            await _kill_and_reap(process)

    return ShellExecution(process, "host (unsandboxed)", exit_code, cleanup)


# The guest's HOME lives on the bounded /tmp tmpfs, so the dotfiles and caches
# that npm, pnpm, pip, playwright and friends write under ~ (.npm, .cache,
# .config, .local) never land in the user's project and vanish with the command.


async def _launch_bubblewrap(
    command: str,
    workspace: Path,
    config: BubblewrapSandbox,
    passed: Mapping[str, str],
    *,
    pass_fds: tuple[int, ...] = (),
    extra_env: Mapping[str, str] | None = None,
) -> ShellExecution:
    if not sys.platform.startswith("linux"):
        raise ToolError("bubblewrap sandboxing is supported only on Linux hosts")
    executable = resolve_sandbox_executable(config.executable, workspace, "bubblewrap")
    _check_mount_sources(config.read_only_mounts)
    aliases, aliased_environment = _alias_guest_environment(passed)
    environment = {
        "HOME": BUBBLEWRAP_HOME,
        "PATH": config.path,
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **(extra_env or {}),
        **aliased_environment,
    }
    try:
        process = await asyncio.create_subprocess_exec(
            *_bubblewrap_argv(executable, command, workspace, config, aliases=aliases),
            cwd=str(workspace),
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            pass_fds=pass_fds,
            **_process_group_kwargs(),
        )
    except OSError as exc:
        raise ToolError(f"failed to launch bubblewrap sandbox: {exc}") from None

    async def exit_code() -> int:
        return (
            process.returncode
            if process.returncode is not None
            else await process.wait()
        )

    async def cleanup(abort: bool) -> None:
        if abort:
            await _kill_and_reap(process)

    return ShellExecution(process, "bubblewrap", exit_code, cleanup)


def _check_mount_sources(mounts: list[ReadOnlyMount]) -> None:
    for mount in mounts:
        if not Path(mount.source).exists():
            raise ToolError(
                f"sandbox read-only mount source does not exist: {mount.source}"
            )


async def _launch_oci(
    command: str,
    workspace: Path,
    config: OciSandbox,
    passed: Mapping[str, str],
    *,
    extra_env: Mapping[str, str] | None = None,
) -> ShellExecution:
    executable = resolve_sandbox_executable(
        config.executable, workspace, config.runtime
    )
    _check_mount_sources(config.read_only_mounts)
    if "," in str(workspace.resolve()):
        raise ToolError("OCI workspace paths may not contain commas")
    control_env = dict(os.environ)
    await _ensure_oci_image(executable, config, control_env)

    name = f"lingcore-{uuid.uuid4().hex}"
    if not _CONTAINER_NAME.fullmatch(name):  # defensive invariant for watchdog
        raise ToolError("could not construct a safe container name")
    environment_file = _create_oci_environment_file(passed)
    try:
        watchdog = _start_watchdog(
            executable, name, control_env, cleanup_path=environment_file
        )
    except ToolError:
        _discard_environment_file(environment_file)
        raise
    try:
        created = await _run_control(
            _oci_create_argv(
                executable,
                name,
                command,
                workspace,
                config,
                environment_file=environment_file,
                extra_env=extra_env,
            ),
            timeout=60,
            environment=control_env,
        )
    except BaseException:
        removed = await _best_effort_remove_container(executable, name, control_env)
        if removed:
            watchdog.disarm()
        else:
            watchdog.trigger()
        raise
    finally:
        _discard_environment_file(environment_file)
    if created.returncode != 0:
        removed = await _best_effort_remove_container(executable, name, control_env)
        if removed:
            watchdog.disarm()
        else:
            watchdog.trigger()
        raise ToolError(
            f"{config.runtime} failed to create the sandbox: "
            f"{_control_message(created.output)}"
        )
    try:
        process = await asyncio.create_subprocess_exec(
            str(executable),
            "start",
            "--attach",
            name,
            env=control_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **_process_group_kwargs(),
        )
    except BaseException as exc:
        removed = await _best_effort_remove_container(executable, name, control_env)
        if removed:
            watchdog.disarm()
        else:
            watchdog.trigger()
        if isinstance(exc, asyncio.CancelledError):
            raise
        if not isinstance(exc, OSError):
            raise
        raise ToolError(f"failed to start {config.runtime} sandbox: {exc}") from None

    cleaned = False

    async def cleanup(abort: bool) -> None:
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        if abort:
            try:
                await _run_control(
                    [str(executable), "kill", name],
                    timeout=15,
                    environment=control_env,
                )
            except ToolError:
                # Still kill the attached client and attempt forced removal.
                pass
            await _kill_and_reap(process)
        try:
            removed = await _remove_container(executable, name, control_env)
        except ToolError:
            watchdog.trigger()
            if abort:
                # The watchdog now owns eventual removal. Cleanup is secondary
                # to the cancellation/timeout that selected the abort path.
                return
            raise
        except BaseException:
            watchdog.trigger()
            raise
        if removed:
            watchdog.disarm()
        else:
            watchdog.trigger()
            if not abort:
                raise ToolError(f"{config.runtime} failed to remove sandbox {name}")

    async def exit_code() -> int:
        inspected = await _run_control(
            [
                str(executable),
                "inspect",
                "--format",
                "{{.State.ExitCode}}",
                name,
            ],
            timeout=15,
            environment=control_env,
        )
        text = inspected.output.decode("utf-8", errors="replace").strip()
        if inspected.returncode != 0 or not text.isdigit():
            raise ToolError(
                f"could not inspect sandbox exit code: {_control_message(inspected.output)}"
            )
        return int(text)

    return ShellExecution(
        process, f"oci/{config.runtime}:{config.container_os}", exit_code, cleanup
    )


async def _ensure_oci_image(
    executable: Path,
    config: OciSandbox,
    environment: Mapping[str, str],
) -> None:
    inspect = await _run_control(
        [
            str(executable),
            "image",
            "inspect",
            "--format",
            OCI_IMAGE_INSPECT_FORMAT,
            config.image,
        ],
        timeout=30,
        environment=environment,
    )
    present = inspect.returncode == 0
    if config.pull == "always" or (config.pull == "missing" and not present):
        pulled = await _run_control(
            [str(executable), "pull", config.image],
            timeout=config.pull_timeout,
            environment=environment,
        )
        if pulled.returncode != 0:
            raise ToolError(
                f"{config.runtime} could not pull image {config.image!r}: "
                f"{_control_message(pulled.output)}"
            )
        inspect = await _run_control(
            [
                str(executable),
                "image",
                "inspect",
                "--format",
                OCI_IMAGE_INSPECT_FORMAT,
                config.image,
            ],
            timeout=30,
            environment=environment,
        )
        present = inspect.returncode == 0
    if not present:
        raise ToolError(
            f"OCI image {config.image!r} is unavailable and pull policy is "
            f"{config.pull!r}: {_control_message(inspect.output)}"
        )
    try:
        metadata = parse_oci_image_metadata(inspect.output)
    except ValueError as exc:
        raise ToolError(f"could not validate OCI image metadata: {exc}") from None
    if metadata.volumes:
        raise ToolError(
            f"OCI image {config.image!r} declares writable volumes, which are "
            "not allowed by the sandbox: " + ", ".join(metadata.volumes)
        )
    if metadata.os_name != config.container_os:
        raise ToolError(
            f"OCI image {config.image!r} reports OS {metadata.os_name!r}, expected "
            f"{config.container_os!r}"
        )


def _start_watchdog(
    executable: Path,
    name: str,
    environment: Mapping[str, str],
    *,
    cleanup_path: Path | None = None,
) -> _Watchdog:
    watchdog_script = Path(__file__).with_name("_sandbox_watchdog.py").resolve()
    argv = [
        sys.executable,
        "-I",
        str(watchdog_script),
        str(executable),
        name,
    ]
    if cleanup_path is not None:
        argv.append(str(cleanup_path))
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            **_process_group_kwargs(),
        )
    except OSError as exc:
        raise ToolError(f"could not start container cleanup watchdog: {exc}") from None
    return _Watchdog(process)


async def _remove_container(
    executable: Path, name: str, environment: Mapping[str, str]
) -> bool:
    removed = await _run_control(
        [str(executable), "rm", "--force", "--volumes", name],
        timeout=30,
        environment=environment,
    )
    return removed.returncode == 0


async def _best_effort_remove_container(
    executable: Path, name: str, environment: Mapping[str, str]
) -> bool:
    try:
        return await asyncio.shield(_remove_container(executable, name, environment))
    except BaseException:
        return False


async def _run_control(
    argv: list[str], *, timeout: float, environment: Mapping[str, str]
) -> _ControlResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **_process_group_kwargs(),
        )
    except OSError as exc:
        raise ToolError(f"failed to run sandbox control command: {exc}") from None
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await asyncio.shield(_kill_and_reap(process))
        raise ToolError(
            f"sandbox control command timed out after {timeout:g}s"
        ) from None
    except asyncio.CancelledError:
        await asyncio.shield(_kill_and_reap(process))
        raise
    except BaseException:
        await asyncio.shield(_kill_and_reap(process))
        raise
    return _ControlResult(process.returncode or 0, output[:_CONTROL_OUTPUT_BYTES])


def _control_message(output: bytes) -> str:
    text = output.decode("utf-8", errors="replace").strip()
    return text or "no diagnostic output"


def _kill_tree(process: asyncio.subprocess.Process) -> None:
    # ``start_new_session=True`` makes the leader's PID the process-group id,
    # and that id remains allocated while any group member is alive. Signal it
    # directly (not through ``os.getpgid(leader)``) even when the leader has
    # already been reaped: its descendants can outlive it and still hold the
    # stdout pipe or keep mutating the workspace.
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    _kill_tree(process)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
