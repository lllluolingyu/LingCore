"""OCI image metadata, environment staging, and container argument construction."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lingcore.errors import ToolError
from lingcore.sandbox_config import OciSandbox

OCI_IMAGE_INSPECT_FORMAT = "{{.Os}}\n{{json .Config.Volumes}}"


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
    extra_env: Mapping[str, str] | None = None,
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
    for name, value in (extra_env or {}).items():
        argv.extend(["--env", f"{name}={value}"])
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
