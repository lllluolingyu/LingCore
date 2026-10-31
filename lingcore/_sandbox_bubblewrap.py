"""Bubblewrap mount arguments and guest environment isolation."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from pathlib import Path

from lingcore.sandbox_config import BubblewrapSandbox


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


BUBBLEWRAP_HOME = "/tmp/home"


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
            "--dir",
            BUBBLEWRAP_HOME,
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
