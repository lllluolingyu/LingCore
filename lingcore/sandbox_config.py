"""Validated shell policy and sandbox configuration, independent of process launch."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_SHELL_TIMEOUT = 60.0

DEFAULT_MAX_CAPTURE_BYTES = 10 * 1024 * 1024

DEFAULT_MAX_OUTPUT_CHARS = 16_000

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]+$")


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
