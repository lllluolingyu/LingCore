"""Opt-in real-backend tests used by trusted self-hosted CI runners."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from lingcore.errors import ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.shell import ShellArgs, run_shell

MODE = os.environ.get("LINGCORE_SANDBOX_E2E", "")
pytestmark = pytest.mark.skipif(not MODE, reason="real sandbox E2E is opt-in")


def _executable(name: str) -> str:
    configured = os.environ.get("LINGCORE_SANDBOX_EXECUTABLE")
    found = configured or shutil.which(name)
    if not found:
        pytest.fail(f"{name} is required by LINGCORE_SANDBOX_E2E={MODE}")
    return str(Path(found).resolve())


def _bubblewrap_context(
    workspace: Path,
    *,
    timeout: float = 30,
    pass_env: list[str] | None = None,
    read_only_mounts: list[dict[str, str]] | None = None,
) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        options={
            "run_shell": {
                "timeout": timeout,
                "require_confirmation": False,
                "sandbox": {
                    "backend": "bubblewrap",
                    "executable": _executable("bwrap"),
                    "network": False,
                    "tmpfs_mb": 64,
                    "pass_env": pass_env or [],
                    "read_only_mounts": read_only_mounts or [],
                },
            }
        },
    )


async def _host_process_exists(pattern: str) -> bool:
    pgrep = _executable("pgrep")
    process = await asyncio.create_subprocess_exec(
        pgrep,
        "-f",
        pattern,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return await process.wait() == 0


async def _assert_process_disappears(pattern: str) -> None:
    for _ in range(30):
        if not await _host_process_exists(pattern):
            return
        await asyncio.sleep(0.1)
    pytest.fail(f"sandbox process survived cleanup: {pattern}")


async def test_real_sandbox_contract(tmp_path):
    if MODE == "bubblewrap":
        sandbox: dict[str, object] = {
            "backend": "bubblewrap",
            "executable": _executable("bwrap"),
            "network": False,
            "tmpfs_mb": 64,
        }
        command = (
            "test ! -e /etc/shadow && printf isolated > /workspace/sandbox-marker.txt"
        )
        expected_runner = "bubblewrap"
    elif MODE in {"docker-linux", "podman-linux"}:
        runtime = MODE.removesuffix("-linux")
        if os.name == "nt":
            default_user = "1000:1000"
        else:
            default_user = f"{os.getuid()}:{os.getgid()}"
        sandbox = {
            "backend": "oci",
            "runtime": runtime,
            "executable": _executable(runtime),
            "image": os.environ.get("LINGCORE_SANDBOX_IMAGE", "python:3.13-slim"),
            "pull": "missing",
            "container_os": "linux",
            "user": os.environ.get("LINGCORE_SANDBOX_USER", default_user),
            "network": False,
            "resources": {
                "cpus": 1,
                "memory_mb": 256,
                "pids": 64,
                "ephemeral_storage_mb": 64,
            },
        }
        command = (
            "if touch /lingcore-root-write 2>/dev/null; then exit 91; fi; "
            "printf isolated > /workspace/sandbox-marker.txt"
        )
        expected_runner = f"oci/{runtime}:linux"
    elif MODE == "docker-windows":
        image = os.environ.get("LINGCORE_WINDOWS_IMAGE")
        if not image:
            pytest.fail("LINGCORE_WINDOWS_IMAGE must name a host-compatible image")
        sandbox = {
            "backend": "oci",
            "runtime": "docker",
            "executable": _executable("docker"),
            "image": image,
            "pull": "missing",
            "container_os": "windows",
            "network": False,
            "resources": {
                "cpus": 1,
                "memory_mb": 512,
                "ephemeral_storage_mb": 256,
            },
        }
        command = "echo isolated>C:\\workspace\\sandbox-marker.txt"
        expected_runner = "oci/docker:windows"
    else:
        pytest.fail(f"unknown LINGCORE_SANDBOX_E2E mode: {MODE}")

    context = ToolContext(
        workspace=tmp_path,
        options={
            "run_shell": {
                "timeout": 180,
                "require_confirmation": False,
                "sandbox": sandbox,
            }
        },
    )
    result = await run_shell(ShellArgs(command=command), context)
    assert f"(runner: {expected_runner})" in result
    assert "(exit code: 0)" in result
    assert (tmp_path / "sandbox-marker.txt").read_text(encoding="utf-8").strip() == (
        "isolated"
    )


@pytest.mark.skipif(MODE != "bubblewrap", reason="Bubblewrap-specific boundary")
async def test_bubblewrap_hides_host_and_ambient_environment(tmp_path, monkeypatch):
    host_secret = tmp_path.parent / f"host-secret-{uuid.uuid4().hex}"
    host_secret.write_text("must stay hidden", encoding="utf-8")
    toolchain = tmp_path.parent / f"toolchain-{uuid.uuid4().hex}"
    toolchain.mkdir()
    (toolchain / "version").write_text("trusted-toolchain", encoding="utf-8")
    monkeypatch.setenv("LINGCORE_AMBIENT_SECRET_E2E", "must-not-enter-sandbox")

    command = " && ".join(
        [
            f"test ! -e {shlex.quote(str(host_secret))}",
            "test ! -e /etc/shadow",
            "test ! -e /sys",
            "! touch /usr/lingcore-must-stay-read-only",
            'test "$HOME" = /workspace',
            'test -z "${LINGCORE_AMBIENT_SECRET_E2E+x}"',
            'test "$(cat /opt/lingcore-toolchain/version)" = trusted-toolchain',
            "! touch /opt/lingcore-toolchain/mutation",
            "test \"$(df -k /tmp | awk 'NR == 2 { print $2 }')\" -le 65536",
            "printf boundary-ok > /workspace/boundary-marker.txt",
        ]
    )
    context = _bubblewrap_context(
        tmp_path,
        read_only_mounts=[
            {"source": str(toolchain), "target": "/opt/lingcore-toolchain"}
        ],
    )
    result = await run_shell(ShellArgs(command=command), context)
    assert "(exit code: 0)" in result
    assert not (toolchain / "mutation").exists()
    assert (tmp_path / "boundary-marker.txt").read_text(encoding="utf-8") == (
        "boundary-ok"
    )


@pytest.mark.skipif(MODE != "bubblewrap", reason="Bubblewrap-specific boundary")
async def test_bubblewrap_blocks_network_caps_and_nested_namespaces(tmp_path):
    command = " && ".join(
        [
            "if python3 -c 'import socket; "
            'socket.create_connection(("1.1.1.1", 53), .2)'
            "'; then exit 90; fi",
            'awk \'/CapEff:/ { found=1; if ($2 != "0000000000000000") '
            "exit 91 } END { if (!found) exit 92 }' /proc/self/status",
            "if unshare --user --map-root-user true 2>/dev/null; then exit 93; fi",
            'test "$(ps -e --no-headers | wc -l)" -le 6',
        ]
    )
    result = await run_shell(ShellArgs(command=command), _bubblewrap_context(tmp_path))
    assert "(exit code: 0)" in result


@pytest.mark.skipif(MODE != "bubblewrap", reason="Bubblewrap-specific boundary")
async def test_bubblewrap_passes_only_named_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGCORE_ALLOWED_SECRET_E2E", "explicit-value")
    monkeypatch.setenv("LINGCORE_BLOCKED_SECRET_E2E", "ambient-value")
    command = (
        'test "$LINGCORE_ALLOWED_SECRET_E2E" = explicit-value && '
        'test -z "${LINGCORE_BLOCKED_SECRET_E2E+x}"'
    )
    result = await run_shell(
        ShellArgs(command=command),
        _bubblewrap_context(tmp_path, pass_env=["LINGCORE_ALLOWED_SECRET_E2E"]),
    )
    assert "(exit code: 0)" in result


@pytest.mark.skipif(MODE != "bubblewrap", reason="Bubblewrap-specific supervision")
async def test_bubblewrap_timeout_leaves_no_process(tmp_path):
    marker = f"lingcore-bwrap-timeout-{uuid.uuid4().hex}"
    pattern = f"^{marker} 30$"
    command = "/bin/bash -c " + shlex.quote(f"exec -a {marker} sleep 30")
    with pytest.raises(ToolError, match="timed out"):
        await run_shell(
            ShellArgs(command=command),
            _bubblewrap_context(tmp_path, timeout=0.25),
        )
    await _assert_process_disappears(pattern)


@pytest.mark.skipif(MODE != "bubblewrap", reason="Bubblewrap-specific supervision")
async def test_bubblewrap_cancellation_leaves_no_process(tmp_path):
    marker = f"lingcore-bwrap-cancel-{uuid.uuid4().hex}"
    pattern = f"^{marker} 30$"
    command = "/bin/bash -c " + shlex.quote(f"exec -a {marker} sleep 30")
    task = asyncio.create_task(
        run_shell(
            ShellArgs(command=command),
            _bubblewrap_context(tmp_path, timeout=30),
        )
    )
    for _ in range(30):
        if await _host_process_exists(pattern):
            break
        await asyncio.sleep(0.1)
    else:
        task.cancel()
        pytest.fail("sandbox child never started")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_process_disappears(pattern)


@pytest.mark.skipif(MODE != "bubblewrap", reason="Bubblewrap-specific supervision")
async def test_bubblewrap_parent_crash_leaves_no_process(tmp_path):
    marker = f"lingcore-bwrap-parent-death-{uuid.uuid4().hex}"
    pattern = f"^{marker} 30$"
    command = "/bin/bash -c " + shlex.quote(f"exec -a {marker} sleep 30")
    driver = f"""
import asyncio
from pathlib import Path
from lingcore.tools import ToolContext
from lingcore.tools.builtin.shell import ShellArgs, run_shell
context = ToolContext(
    workspace=Path({str(tmp_path)!r}),
    options={{"run_shell": {{
        "timeout": 30,
        "require_confirmation": False,
        "sandbox": {{
            "backend": "bubblewrap",
            "executable": {_executable("bwrap")!r},
            "network": False,
            "tmpfs_mb": 64,
        }},
    }}}},
)
asyncio.run(run_shell(ShellArgs(command={command!r}), context))
"""
    parent = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        driver,
        cwd=str(Path(__file__).parent.parent),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        for _ in range(30):
            if await _host_process_exists(pattern):
                break
            if parent.returncode is not None:
                pytest.fail(f"sandbox driver exited early: {parent.returncode}")
            await asyncio.sleep(0.1)
        else:
            pytest.fail("sandbox child never started")
        parent.kill()
        await parent.wait()
        await _assert_process_disappears(pattern)
    finally:
        if parent.returncode is None:
            parent.kill()
            await parent.wait()
