"""Sandbox backends for :mod:`lingcore.tools.builtin.shell`.

The shell tool deliberately keeps its confirmation and output policy separate
from process isolation.  This module owns the typed backend configuration and
turns one command into a supervised process.  A missing sandbox configuration
means the legacy host runner; a configured backend never falls back to it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lingcore.errors import ToolError

DEFAULT_SHELL_TIMEOUT = 60.0
DEFAULT_MAX_CAPTURE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_OUTPUT_CHARS = 16_000
_CONTROL_OUTPUT_BYTES = 256 * 1024
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]+$")
OCI_IMAGE_INSPECT_FORMAT = "{{.Os}}\n{{json .Config.Volumes}}"


class ReadOnlyMount(BaseModel):
    """One explicit host path made visible read-only in a sandbox."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str

    @field_validator("source", "target")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("mount paths must not be blank")
        return value


class BubblewrapSandbox(BaseModel):
    """Linux bubblewrap isolation rooted in an empty mount namespace."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["bubblewrap"]
    executable: str = "/usr/bin/bwrap"
    network: bool = False
    tmpfs_mb: int = Field(default=512, ge=16)
    path: str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    pass_env: list[str] = Field(default_factory=list)
    read_only_mounts: list[ReadOnlyMount] = Field(default_factory=list)

    @field_validator("pass_env")
    @classmethod
    def _valid_environment_names(cls, values: list[str]) -> list[str]:
        return _validate_environment_names(values)

    @model_validator(mode="after")
    def _validate_paths(self) -> "BubblewrapSandbox":
        _require_absolute(self.executable, "sandbox.executable")
        for mount in self.read_only_mounts:
            _require_absolute(mount.source, "read_only_mounts.source")
            target = PurePosixPath(mount.target)
            if not target.is_absolute():
                raise ValueError(
                    "bubblewrap mount targets must be absolute POSIX paths"
                )
            if ".." in target.parts:
                raise ValueError("bubblewrap mount targets may not contain '..'")
            mount.target = str(target)
            if mount.target in {"/", "/workspace", "/proc", "/dev", "/tmp"}:
                raise ValueError(
                    f"bubblewrap mount target {mount.target!r} is reserved"
                )
        return self


class OciResources(BaseModel):
    """Mandatory resource budget for an OCI-backed shell."""

    model_config = ConfigDict(extra="forbid")

    cpus: float = Field(gt=0)
    memory_mb: int = Field(ge=64)
    pids: int | None = Field(default=None, ge=16)
    ephemeral_storage_mb: int = Field(ge=16)


class OciSandbox(BaseModel):
    """Docker or Podman container isolation."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["oci"]
    runtime: Literal["docker", "podman"]
    executable: str
    image: str
    pull: Literal["missing", "always", "never"] = "missing"
    pull_timeout: float = Field(default=300.0, gt=0)
    container_os: Literal["linux", "windows"] = "linux"
    command_prefix: list[str] | None = None
    workspace_target: str | None = None
    user: str | None = None
    network: bool = False
    pass_env: list[str] = Field(default_factory=list)
    read_only_mounts: list[ReadOnlyMount] = Field(default_factory=list)
    resources: OciResources

    @field_validator("pass_env")
    @classmethod
    def _valid_environment_names(cls, values: list[str]) -> list[str]:
        return _validate_environment_names(values)

    @field_validator("image", "executable")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _platform_contract(self) -> "OciSandbox":
        _require_absolute(self.executable, "sandbox.executable")
        workspace_target: PurePosixPath | PureWindowsPath
        if self.container_os == "linux":
            self.command_prefix = self.command_prefix or ["/bin/sh", "-c"]
            self.workspace_target = self.workspace_target or "/workspace"
            if self.user is None or not self.user.strip():
                raise ValueError(
                    "Linux OCI sandboxes require an explicit non-root user"
                )
            self.user = self.user.strip()
            identity = self.user.split(":")
            if len(identity) > 2 or any(not part for part in identity):
                raise ValueError("Linux OCI user must be USER[:GROUP] or UID[:GID]")
            if _is_root_identity(identity[0]):
                raise ValueError("Linux OCI sandboxes may not run as root")
            if len(identity) == 2 and _is_root_identity(identity[1]):
                raise ValueError("Linux OCI sandboxes may not use the root group")
            if self.resources.pids is None:
                raise ValueError("Linux OCI sandboxes require resources.pids")
            workspace_target = PurePosixPath(self.workspace_target)
        else:
            if self.runtime != "docker":
                raise ValueError("native Windows containers require the Docker runtime")
            self.command_prefix = self.command_prefix or ["cmd.exe", "/S", "/C"]
            self.workspace_target = self.workspace_target or "C:\\workspace"
            self.user = self.user or "ContainerUser"
            if self.user.lower() in {"containeradministrator", "administrator"}:
                raise ValueError(
                    "native Windows sandboxes may not run as administrator"
                )
            if self.resources.pids is not None:
                raise ValueError(
                    "native Windows sandboxes do not accept resources.pids; "
                    "Hyper-V isolation supplies the process boundary"
                )
            workspace_target = PureWindowsPath(self.workspace_target)

        if not self.command_prefix or any(not part for part in self.command_prefix):
            raise ValueError("command_prefix must contain non-empty arguments")
        if not workspace_target.is_absolute():
            raise ValueError("workspace_target must be absolute for the container OS")
        if ".." in workspace_target.parts:
            raise ValueError("workspace_target may not contain '..'")
        self.workspace_target = str(workspace_target)
        for mount in self.read_only_mounts:
            _require_absolute(mount.source, "read_only_mounts.source")
            target = (
                PurePosixPath(mount.target)
                if self.container_os == "linux"
                else PureWindowsPath(mount.target)
            )
            if not target.is_absolute():
                raise ValueError(
                    "OCI mount targets must be absolute for the container OS"
                )
            if ".." in target.parts:
                raise ValueError("OCI mount targets may not contain '..'")
            mount.target = str(target)
            if mount.target == self.workspace_target:
                raise ValueError(
                    "read-only mount target conflicts with workspace_target"
                )
            if "," in mount.source or "," in mount.target:
                raise ValueError("OCI mount paths may not contain commas")
        return self


SandboxConfig = Annotated[
    BubblewrapSandbox | OciSandbox, Field(discriminator="backend")
]


class ShellOptions(BaseModel):
    """Validated ``tool_options.run_shell`` contract."""

    model_config = ConfigDict(extra="forbid")

    timeout: float = Field(default=DEFAULT_SHELL_TIMEOUT, gt=0)
    # Ceiling for a per-call ``timeout`` the model may request for a slow build
    # or test run. Unset keeps ``timeout`` as the ceiling, so a call can only
    # shorten it.
    max_timeout: float | None = Field(default=None, gt=0)
    require_confirmation: bool = True
    allow_patterns: list[str] = Field(default_factory=list)
    max_capture_bytes: int = Field(default=DEFAULT_MAX_CAPTURE_BYTES, ge=1)
    offload_over_chars: int = Field(default=8_000, ge=0)
    max_output_chars: int = Field(default=DEFAULT_MAX_OUTPUT_CHARS, ge=1)
    sandbox: SandboxConfig | None = None

    @model_validator(mode="after")
    def _check_timeout_ceiling(self) -> ShellOptions:
        if self.max_timeout is not None and self.max_timeout < self.timeout:
            raise ValueError("run_shell.max_timeout must be at least run_shell.timeout")
        return self

    def effective_timeout(self, requested: float | None) -> float:
        """Clamp a per-call timeout request to the configured ceiling."""
        ceiling = self.max_timeout if self.max_timeout is not None else self.timeout
        return min(requested if requested is not None else self.timeout, ceiling)


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
) -> ShellExecution:
    """Launch a command with the selected runner, without fallback."""
    sandbox = options.sandbox
    if sandbox is None:
        return await _launch_host(command, workspace)
    passed = _passed_environment(sandbox.pass_env, getenv)
    if isinstance(sandbox, BubblewrapSandbox):
        return await _launch_bubblewrap(command, workspace, sandbox, passed)
    return await _launch_oci(command, workspace, sandbox, passed)


def _validate_environment_names(values: list[str]) -> list[str]:
    deduplicated = list(dict.fromkeys(values))
    for name in deduplicated:
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid environment variable name: {name!r}")
    return deduplicated


def _is_root_identity(value: str) -> bool:
    """Recognize Docker/Podman root identities in name and numeric forms."""
    if value.lower() == "root":
        return True
    try:
        return int(value, 10) == 0
    except ValueError:
        return False


def _require_absolute(value: str, field: str) -> None:
    if not Path(value).is_absolute():
        raise ValueError(f"{field} must be an absolute host path")


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


def _alias_guest_environment(
    passed: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Hide guest variable names from the host-side backend process.

    A guest variable such as ``LD_PRELOAD`` must not affect the dynamic loader
    that starts Bubblewrap. Random, inert aliases carry the values across the
    namespace boundary; a shell inside the sandbox restores the requested names
    immediately before it launches the user's command.
    """
    token = uuid.uuid4().hex.upper()
    aliases: dict[str, str] = {}
    environment: dict[str, str] = {}
    for index, (name, value) in enumerate(passed.items()):
        alias = f"LINGCORE_GUEST_ENV_{token}_{index}"
        while alias in passed:
            alias += "_"
        aliases[name] = alias
        environment[alias] = value
    return aliases, environment


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


async def _launch_host(command: str, workspace: Path) -> ShellExecution:
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace),
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


def bubblewrap_system_mount_args() -> list[str]:
    """Mirror the host's executable/library aliases without assuming usrmerge."""
    argv = ["--ro-bind", "/usr", "/usr"]
    for path_text in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(path_text)
        if path.is_symlink():
            argv.extend(["--symlink", os.readlink(path), path_text])
        elif path.exists():
            argv.extend(["--ro-bind", path_text, path_text])
    return argv


def _bubblewrap_guest_command(command: str, aliases: Mapping[str, str]) -> list[str]:
    if not aliases:
        return ["/bin/sh", "-c", command]
    lines = ["set -e"]
    lines.extend(f'export {name}="${{{alias}}}"' for name, alias in aliases.items())
    lines.extend(f"unset {alias}" for alias in aliases.values())
    lines.append('exec /bin/sh -c "$1"')
    return ["/bin/sh", "-c", "\n".join(lines), "lingcore-env", command]


def _bubblewrap_argv(
    executable: Path,
    command: str,
    workspace: Path,
    config: BubblewrapSandbox,
    aliases: Mapping[str, str] | None = None,
) -> list[str]:
    argv = [
        str(executable),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--hostname",
        "lingcore",
        "--disable-userns",
        "--cap-drop",
        "ALL",
    ]
    if not config.network:
        argv.append("--unshare-net")
    argv.extend(
        [
            "--dir",
            "/etc",
            "--ro-bind-try",
            "/etc/passwd",
            "/etc/passwd",
            "--ro-bind-try",
            "/etc/group",
            "/etc/group",
            "--ro-bind-try",
            "/etc/nsswitch.conf",
            "/etc/nsswitch.conf",
            "--ro-bind-try",
            "/etc/hosts",
            "/etc/hosts",
            "--ro-bind-try",
            "/etc/resolv.conf",
            "/etc/resolv.conf",
            "--ro-bind-try",
            "/etc/ssl",
            "/etc/ssl",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--dir",
            "/run",
            "--dir",
            "/var",
            "--dir",
            "/workspace",
            "--bind",
            str(workspace.resolve()),
            "/workspace",
            "--size",
            str(config.tmpfs_mb * 1024 * 1024),
            "--tmpfs",
            "/tmp",
        ]
    )
    # /bin, /sbin, /lib, and /lib64 vary across usrmerged distributions.
    # Apply these before the private writable directories and custom mounts.
    root_insert = argv.index("--dir")
    argv[root_insert:root_insert] = bubblewrap_system_mount_args()
    for mount in config.read_only_mounts:
        argv.extend(["--ro-bind", str(Path(mount.source).resolve()), mount.target])
    argv.extend(["--chdir", "/workspace"])
    argv.extend(_bubblewrap_guest_command(command, aliases or {}))
    return argv


async def _launch_bubblewrap(
    command: str,
    workspace: Path,
    config: BubblewrapSandbox,
    passed: Mapping[str, str],
) -> ShellExecution:
    if not sys.platform.startswith("linux"):
        raise ToolError("bubblewrap sandboxing is supported only on Linux hosts")
    executable = resolve_sandbox_executable(config.executable, workspace, "bubblewrap")
    _check_mount_sources(config.read_only_mounts)
    aliases, aliased_environment = _alias_guest_environment(passed)
    environment = {
        "HOME": "/workspace",
        "PATH": config.path,
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **aliased_environment,
    }
    try:
        process = await asyncio.create_subprocess_exec(
            *_bubblewrap_argv(executable, command, workspace, config, aliases=aliases),
            cwd=str(workspace),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
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


def _oci_mount(source: str, target: str, *, readonly: bool) -> str:
    value = f"type=bind,source={source},target={target}"
    return value + (",readonly" if readonly else "")


@dataclass(frozen=True, slots=True)
class OciImageMetadata:
    os_name: str
    volumes: tuple[str, ...]


def parse_oci_image_metadata(output: bytes) -> OciImageMetadata:
    """Parse the bounded output of :data:`OCI_IMAGE_INSPECT_FORMAT`."""
    text = output.decode("utf-8", errors="replace").strip()
    if "\n" not in text:
        raise ValueError("OCI image inspection returned incomplete metadata")
    raw_os, raw_volumes = text.split("\n", 1)
    os_name = raw_os.strip().lower()
    if not os_name:
        raise ValueError("OCI image inspection did not report an operating system")
    try:
        decoded_volumes = json.loads(raw_volumes)
    except json.JSONDecodeError:
        raise ValueError(
            "OCI image inspection returned invalid volume metadata"
        ) from None
    if decoded_volumes is None:
        volumes: tuple[str, ...] = ()
    elif isinstance(decoded_volumes, dict) and all(
        isinstance(name, str) for name in decoded_volumes
    ):
        volumes = tuple(sorted(decoded_volumes))
    else:
        raise ValueError("OCI image inspection returned invalid volume metadata")
    return OciImageMetadata(os_name=os_name, volumes=volumes)


def _create_oci_environment_file(passed: Mapping[str, str]) -> Path | None:
    """Stage guest-only variables for the runtime CLI with restrictive access."""
    if not passed:
        return None
    for value in passed.values():
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ToolError(
                "OCI sandbox environment values may not contain NUL or newlines"
            )
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix="lingcore-sandbox-env-")
    except OSError as exc:
        raise ToolError(f"could not stage OCI sandbox environment: {exc}") from None
    path = Path(raw_path)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for name, value in passed.items():
                stream.write(f"{name}={value}\n")
    except (OSError, UnicodeError) as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass
        raise ToolError(f"could not stage OCI sandbox environment: {exc}") from None
    return path


def _discard_environment_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        # The watchdog receives the same path and retries unlinking before it
        # disarms or performs crash cleanup.
        pass


def _oci_create_argv(
    executable: Path,
    name: str,
    command: str,
    workspace: Path,
    config: OciSandbox,
    *,
    environment_file: Path | None = None,
) -> list[str]:
    assert config.workspace_target is not None
    assert config.command_prefix is not None
    assert config.user is not None
    resources = config.resources
    argv = [
        str(executable),
        "create",
        "--name",
        name,
        "--label",
        "io.lingcore.sandbox=true",
        "--no-healthcheck",
        "--cpus",
        str(resources.cpus),
        "--memory",
        f"{resources.memory_mb}m",
        "--user",
        config.user,
        "--workdir",
        config.workspace_target,
        "--mount",
        _oci_mount(str(workspace.resolve()), config.workspace_target, readonly=False),
    ]
    if not config.network:
        argv.extend(["--network", "none"])
    if config.pass_env:
        if environment_file is None:
            raise ValueError("OCI pass_env requires a staged environment file")
        argv.extend(["--env-file", str(environment_file)])
    for mount in config.read_only_mounts:
        argv.extend(
            [
                "--mount",
                _oci_mount(
                    str(Path(mount.source).resolve()), mount.target, readonly=True
                ),
            ]
        )

    if config.container_os == "linux":
        assert resources.pids is not None
        argv.extend(
            [
                "--read-only",
                "--tmpfs",
                f"/tmp:rw,nosuid,nodev,size={resources.ephemeral_storage_mb}m",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges=true",
                "--pids-limit",
                str(resources.pids),
            ]
        )
    else:
        argv.extend(
            [
                "--isolation",
                "hyperv",
                "--storage-opt",
                f"size={resources.ephemeral_storage_mb}m",
            ]
        )

    argv.extend(["--entrypoint", config.command_prefix[0], config.image])
    argv.extend([*config.command_prefix[1:], command])
    return argv


async def _launch_oci(
    command: str,
    workspace: Path,
    config: OciSandbox,
    passed: Mapping[str, str],
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
    if process.returncode is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
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
