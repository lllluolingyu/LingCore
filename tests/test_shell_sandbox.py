"""Typed configuration and backend contracts for sandboxed run_shell."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, ToolError
from lingcore.sandbox import (
    BubblewrapSandbox,
    OciSandbox,
    ShellOptions,
    _alias_guest_environment,
    _bubblewrap_argv,
    _ControlResult,
    _ensure_oci_image,
    _oci_create_argv,
    _run_control,
    bubblewrap_system_mount_args,
)
from lingcore.tools import ToolContext
from lingcore.tools.builtin.shell import ShellArgs, run_shell


def _linux_oci(**overrides: object) -> OciSandbox:
    raw: dict[str, object] = {
        "backend": "oci",
        "runtime": "docker",
        "executable": "/usr/bin/docker",
        "image": "python:3.13-slim",
        "container_os": "linux",
        "user": "1000:1000",
        "resources": {
            "cpus": 2,
            "memory_mb": 2048,
            "pids": 256,
            "ephemeral_storage_mb": 512,
        },
    }
    raw.update(overrides)
    return OciSandbox.model_validate(raw)


def test_shell_options_forbid_misspelled_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ShellOptions.model_validate({"sandox": {"backend": "bubblewrap"}})


def test_profile_load_validates_nested_shell_options(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
name: invalid-shell
llm:
  model: test
tools: [run_shell]
tool_options:
  run_shell:
    timeout_seconds: 20
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="timeout_seconds"):
        AgentProfile.load(tmp_path)


def test_linux_oci_requires_nonroot_user_pid_and_all_budgets():
    for root_identity in (
        "root",
        "root:1000",
        "0",
        "0:1000",
        "00",
        "+0",
        "-0",
    ):
        with pytest.raises(ValidationError, match="run as root"):
            _linux_oci(user=root_identity)
    for root_group in ("1000:root", "1000:0", "1000:00", "1000:+0"):
        with pytest.raises(ValidationError, match="root group"):
            _linux_oci(user=root_group)
    with pytest.raises(ValidationError, match="resources.pids"):
        _linux_oci(
            resources={
                "cpus": 1,
                "memory_mb": 128,
                "ephemeral_storage_mb": 64,
            }
        )
    with pytest.raises(ValidationError, match="ephemeral_storage_mb"):
        _linux_oci(resources={"cpus": 1, "memory_mb": 128, "pids": 32})


def test_native_windows_contract_requires_docker_hyperv_without_pid_limit():
    base = {
        "backend": "oci",
        "runtime": "docker",
        "executable": "C:\\Program Files\\Docker\\docker.exe",
        "image": "mcr.microsoft.com/windows/nanoserver:ltsc2025",
        "container_os": "windows",
        "resources": {
            "cpus": 2,
            "memory_mb": 2048,
            "ephemeral_storage_mb": 512,
        },
    }
    # Host-path validation follows the current host. Pure Windows contracts are
    # exercised on Windows E2E runners; use a POSIX executable for this pure
    # cross-platform validation when the unit suite runs elsewhere.
    if os.name != "nt":
        base["executable"] = "/usr/bin/docker"
    config = OciSandbox.model_validate(base)
    assert config.user == "ContainerUser"
    assert config.command_prefix == ["cmd.exe", "/S", "/C"]

    with pytest.raises(ValidationError, match="Docker runtime"):
        OciSandbox.model_validate({**base, "runtime": "podman"})
    resources = dict(base["resources"])
    resources["pids"] = 64
    with pytest.raises(ValidationError, match="do not accept resources.pids"):
        OciSandbox.model_validate({**base, "resources": resources})


def test_bubblewrap_argv_has_empty_root_security_and_workspace_mount(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = BubblewrapSandbox(
        backend="bubblewrap",
        executable="/usr/bin/bwrap",
        network=False,
    )
    argv = _bubblewrap_argv(Path(config.executable), "echo ok", workspace, config)
    joined = " ".join(argv)
    assert "--unshare-user" in argv
    assert "--unshare-pid" in argv
    assert "--unshare-net" in argv
    assert "--cap-drop ALL" in joined
    assert ["--bind", str(workspace.resolve()), "/workspace"] == argv[
        argv.index("--bind") : argv.index("--bind") + 3
    ]
    assert "--bind / /" not in joined
    assert argv[-3:] == ["/bin/sh", "-c", "echo ok"]


def test_bubblewrap_mirrors_host_system_aliases():
    argv = bubblewrap_system_mount_args()
    for path_text in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(path_text)
        if path.is_symlink():
            expected = ["--symlink", os.readlink(path), path_text]
        elif path.exists():
            expected = ["--ro-bind", path_text, path_text]
        else:
            continue
        assert any(argv[index : index + 3] == expected for index in range(len(argv)))


def test_bubblewrap_guest_environment_is_aliased_before_host_launch(tmp_path):
    aliases, host_environment = _alias_guest_environment(
        {"LD_PRELOAD": "./workspace-escape.so", "BUILD_TOKEN": "secret"}
    )
    assert "LD_PRELOAD" not in host_environment
    assert "BUILD_TOKEN" not in host_environment
    assert set(host_environment.values()) == {"./workspace-escape.so", "secret"}

    config = BubblewrapSandbox(backend="bubblewrap")
    argv = _bubblewrap_argv(
        Path(config.executable), "echo ok", tmp_path, config, aliases=aliases
    )
    joined = "\n".join(argv)
    assert "workspace-escape.so" not in joined
    assert "secret" not in joined
    assert f'export LD_PRELOAD="${{{aliases["LD_PRELOAD"]}}}"' in joined
    assert f"unset {aliases['LD_PRELOAD']}" in joined


async def test_bubblewrap_launcher_never_receives_guest_variable_names(
    tmp_path, monkeypatch
):
    import lingcore.sandbox as sandbox_module

    captured_argv: tuple[str, ...] = ()
    captured_environment: Mapping[str, str] = {}

    class FakeProcess:
        returncode = 0

        async def wait(self) -> int:
            return 0

    async def fake_subprocess(*argv: str, **kwargs: object) -> FakeProcess:
        nonlocal captured_argv, captured_environment
        captured_argv = argv
        captured_environment = kwargs["env"]  # type: ignore[assignment]
        return FakeProcess()

    monkeypatch.setattr(
        sandbox_module.asyncio, "create_subprocess_exec", fake_subprocess
    )
    config = BubblewrapSandbox(
        backend="bubblewrap", executable=str(Path(sys.executable).resolve())
    )
    execution = await sandbox_module._launch_bubblewrap(
        "printf ok",
        tmp_path,
        config,
        {"LD_PRELOAD": "/workspace/host-loader-escape.so"},
    )

    assert execution.runner == "bubblewrap"
    assert "LD_PRELOAD" not in captured_environment
    aliases = [name for name in captured_environment if name.startswith("LINGCORE_")]
    assert len(aliases) == 1
    assert captured_environment[aliases[0]] == "/workspace/host-loader-escape.so"
    assert "/workspace/host-loader-escape.so" not in captured_argv


def test_watchdog_uses_isolated_known_script(monkeypatch):
    import lingcore.sandbox as sandbox_module

    captured: list[str] = []

    class FakeProcess:
        stdin = None

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        captured.extend(argv)
        return FakeProcess()

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", fake_popen)
    watchdog = sandbox_module._start_watchdog(
        Path("/usr/bin/docker"), "lingcore-" + "a" * 32, {}
    )

    assert watchdog.process is not None
    assert captured[0:2] == [sys.executable, "-I"]
    assert (
        Path(captured[2])
        == Path(sandbox_module.__file__).with_name("_sandbox_watchdog.py").resolve()
    )
    assert captured[3:] == ["/usr/bin/docker", "lingcore-" + "a" * 32]


def test_linux_oci_argv_applies_security_and_resource_budget(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = _linux_oci()
    argv = _oci_create_argv(
        Path(config.executable), "lingcore-" + "a" * 32, "pytest -q", workspace, config
    )
    joined = " ".join(argv)
    for expected in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges=true",
        "--pids-limit 256",
        "--memory 2048m",
        "--cpus 2.0",
        "--user 1000:1000",
    ):
        assert expected in joined
    assert argv[-5:] == [
        "--entrypoint",
        "/bin/sh",
        "python:3.13-slim",
        "-c",
        "pytest -q",
    ]


async def test_missing_pass_env_fails_before_backend_launch(tmp_path):
    context = ToolContext(
        workspace=tmp_path,
        options={
            "run_shell": {
                "require_confirmation": False,
                "sandbox": {
                    "backend": "bubblewrap",
                    "executable": "/definitely/missing/bwrap",
                    "pass_env": ["LINGCORE_MISSING_SANDBOX_TEST"],
                },
            }
        },
    )
    with pytest.raises(ToolError, match="LINGCORE_MISSING_SANDBOX_TEST"):
        await run_shell(ShellArgs(command="echo unsafe"), context)


async def test_oci_runner_pulls_runs_inspects_and_removes(tmp_path, monkeypatch):
    import lingcore.sandbox as sandbox_module

    monkeypatch.delenv("SANDBOX_BUILD_TOKEN", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[list[str]] = []
    image_present = False
    environment_file: Path | None = None

    async def fake_control(
        argv: list[str], *, timeout: float, environment: Mapping[str, str]
    ) -> _ControlResult:
        nonlocal environment_file, image_present
        calls.append(argv)
        assert "SANDBOX_BUILD_TOKEN" not in environment
        if argv[1:3] == ["image", "inspect"]:
            return _ControlResult(
                0 if image_present else 1,
                b"linux\nnull\n" if image_present else b"",
            )
        if argv[1] == "pull":
            image_present = True
            return _ControlResult(0, b"")
        if argv[1] == "create":
            environment_file = Path(argv[argv.index("--env-file") + 1])
            assert environment_file.read_text(encoding="utf-8") == (
                "SANDBOX_BUILD_TOKEN=guest-secret\n"
            )
        if argv[1] == "inspect":
            return _ControlResult(0, b"7\n")
        return _ControlResult(0, b"")

    class FakeReader:
        def __init__(self) -> None:
            self._output = b"sandbox output\n"

        async def read(self, limit: int) -> bytes:
            output, self._output = self._output, b""
            return output

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeReader()
            self.returncode = 0

        async def wait(self) -> int:
            return 0

    class FakeWatchdog:
        disarmed = False

        def disarm(self) -> None:
            self.disarmed = True

        def trigger(self) -> None:
            pytest.fail("successful cleanup must not trigger the watchdog")

    watchdog = FakeWatchdog()

    def fake_watchdog(
        executable: Path,
        name: str,
        environment: Mapping[str, str],
        *,
        cleanup_path: Path | None = None,
    ) -> FakeWatchdog:
        assert cleanup_path is not None
        return watchdog

    async def fake_subprocess(*argv: str, **kwargs: object) -> FakeProcess:
        assert argv[1:3] == ("start", "--attach")
        return FakeProcess()

    monkeypatch.setattr(sandbox_module, "_run_control", fake_control)
    monkeypatch.setattr(sandbox_module, "_start_watchdog", fake_watchdog)
    monkeypatch.setattr(
        sandbox_module.asyncio, "create_subprocess_exec", fake_subprocess
    )

    context = ToolContext(
        workspace=workspace,
        options={
            "run_shell": {
                "require_confirmation": False,
                "sandbox": {
                    "backend": "oci",
                    "runtime": "docker",
                    "executable": str(Path(os.sys.executable).resolve()),
                    "image": "example.test/build:fixed",
                    "pull": "missing",
                    "container_os": "linux",
                    "user": "1000:1000",
                    "pass_env": ["SANDBOX_BUILD_TOKEN"],
                    "resources": {
                        "cpus": 1,
                        "memory_mb": 128,
                        "pids": 32,
                        "ephemeral_storage_mb": 64,
                    },
                },
            }
        },
        environment={"SANDBOX_BUILD_TOKEN": "guest-secret"},
    )
    result = await run_shell(ShellArgs(command="echo from container"), context)
    assert "(runner: oci/docker:linux)" in result
    assert "(exit code: 7)" in result
    assert "sandbox output" in result
    rendered = [" ".join(call) for call in calls]
    assert any("pull example.test/build:fixed" in call for call in rendered)
    assert any("create --name lingcore-" in call for call in rendered)
    assert any(
        "inspect --format {{.State.ExitCode}} lingcore-" in call for call in rendered
    )
    assert any("rm --force --volumes lingcore-" in call for call in rendered)
    assert watchdog.disarmed
    assert environment_file is not None and not environment_file.exists()


async def test_oci_rejects_image_declared_writable_volumes(monkeypatch):
    import lingcore.sandbox as sandbox_module

    async def fake_control(
        argv: list[str], *, timeout: float, environment: Mapping[str, str]
    ) -> _ControlResult:
        assert argv[1:3] == ["image", "inspect"]
        return _ControlResult(0, b'linux\n{"/var/lib/data": {}}\n')

    monkeypatch.setattr(sandbox_module, "_run_control", fake_control)
    with pytest.raises(ToolError, match="declares writable volumes"):
        await _ensure_oci_image(Path("/usr/bin/docker"), _linux_oci(), {})


async def test_control_command_cancellation_kills_and_reaps(monkeypatch):
    import lingcore.sandbox as sandbox_module

    communicating = asyncio.Event()
    killed = False

    class FakeProcess:
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes | None]:
            communicating.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    process = FakeProcess()

    async def fake_subprocess(*argv: str, **kwargs: object) -> FakeProcess:
        return process

    async def fake_kill(candidate: object) -> None:
        nonlocal killed
        assert candidate is process
        killed = True

    monkeypatch.setattr(
        sandbox_module.asyncio, "create_subprocess_exec", fake_subprocess
    )
    monkeypatch.setattr(sandbox_module, "_kill_and_reap", fake_kill)

    task = asyncio.create_task(
        _run_control(["/runtime", "info"], timeout=60, environment={})
    )
    await communicating.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed


async def test_oci_create_cancellation_removes_partial_container(tmp_path, monkeypatch):
    import lingcore.sandbox as sandbox_module

    monkeypatch.delenv("SANDBOX_BUILD_TOKEN", raising=False)
    create_started = asyncio.Event()
    calls: list[list[str]] = []
    staged_path: Path | None = None

    async def fake_control(
        argv: list[str], *, timeout: float, environment: Mapping[str, str]
    ) -> _ControlResult:
        nonlocal staged_path
        calls.append(argv)
        assert "SANDBOX_BUILD_TOKEN" not in environment
        if argv[1:3] == ["image", "inspect"]:
            return _ControlResult(0, b"linux\nnull\n")
        if argv[1] == "create":
            staged_path = Path(argv[argv.index("--env-file") + 1])
            create_started.set()
            await asyncio.Event().wait()
        if argv[1] == "rm":
            return _ControlResult(0, b"")
        raise AssertionError(argv)

    class FakeWatchdog:
        disarmed = False
        triggered = False

        def disarm(self) -> None:
            self.disarmed = True

        def trigger(self) -> None:
            self.triggered = True

    watchdog = FakeWatchdog()

    def fake_watchdog(
        executable: Path,
        name: str,
        environment: Mapping[str, str],
        *,
        cleanup_path: Path | None = None,
    ) -> FakeWatchdog:
        assert cleanup_path is not None
        return watchdog

    monkeypatch.setattr(sandbox_module, "_run_control", fake_control)
    monkeypatch.setattr(sandbox_module, "_start_watchdog", fake_watchdog)

    context = ToolContext(
        workspace=tmp_path,
        options={
            "run_shell": {
                "require_confirmation": False,
                "sandbox": {
                    "backend": "oci",
                    "runtime": "docker",
                    "executable": str(Path(os.sys.executable).resolve()),
                    "image": "example.test/build:fixed",
                    "pull": "never",
                    "container_os": "linux",
                    "user": "1000:1000",
                    "pass_env": ["SANDBOX_BUILD_TOKEN"],
                    "resources": {
                        "cpus": 1,
                        "memory_mb": 128,
                        "pids": 32,
                        "ephemeral_storage_mb": 64,
                    },
                },
            }
        },
        environment={"SANDBOX_BUILD_TOKEN": "guest-secret"},
    )
    task = asyncio.create_task(
        run_shell(ShellArgs(command="echo from container"), context)
    )
    await create_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert any(call[1:4] == ["rm", "--force", "--volumes"] for call in calls)
    assert watchdog.disarmed and not watchdog.triggered
    assert staged_path is not None and not staged_path.exists()


@pytest.mark.parametrize(
    "scenario", ["abort", "exit_success", "exit_error", "exit_cancel"]
)
async def test_oci_cleanup_arms_watchdog_when_forced_removal_fails(
    tmp_path, monkeypatch, scenario
):
    import lingcore.sandbox as sandbox_module

    calls: list[list[str]] = []
    inspect_started = asyncio.Event()

    async def fake_control(
        argv: list[str], *, timeout: float, environment: Mapping[str, str]
    ) -> _ControlResult:
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return _ControlResult(0, b"linux\nnull\n")
        if argv[1] in {"create", "kill"}:
            return _ControlResult(0, b"")
        if argv[1] == "inspect":
            if scenario == "exit_error":
                return _ControlResult(1, b"inspect failed")
            if scenario == "exit_cancel":
                inspect_started.set()
                await asyncio.Event().wait()
            return _ControlResult(0, b"0\n")
        if argv[1] == "rm":
            return _ControlResult(1, b"daemon unavailable")
        raise AssertionError(argv)

    class FakeProcess:
        returncode = None
        stdout = None

    class FakeWatchdog:
        disarmed = False
        triggered = False

        def disarm(self) -> None:
            self.disarmed = True

        def trigger(self) -> None:
            self.triggered = True

    process = FakeProcess()
    watchdog = FakeWatchdog()

    async def fake_subprocess(*argv: str, **kwargs: object) -> FakeProcess:
        return process

    async def fake_kill(candidate: object) -> None:
        assert candidate is process

    monkeypatch.setattr(sandbox_module, "_run_control", fake_control)
    monkeypatch.setattr(sandbox_module, "_start_watchdog", lambda *a, **kw: watchdog)
    monkeypatch.setattr(
        sandbox_module.asyncio, "create_subprocess_exec", fake_subprocess
    )
    monkeypatch.setattr(sandbox_module, "_kill_and_reap", fake_kill)

    execution = await sandbox_module._launch_oci(
        "echo from container",
        tmp_path,
        _linux_oci(executable=str(Path(sys.executable).resolve()), pull="never"),
        {},
    )
    if scenario == "abort":
        await execution.abort()
    elif scenario == "exit_success":
        with pytest.raises(ToolError, match="failed to remove sandbox"):
            await execution.finish()
    elif scenario == "exit_error":
        with pytest.raises(ToolError, match="could not inspect sandbox exit code"):
            await execution.finish()
    else:
        task = asyncio.create_task(execution.finish())
        await inspect_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    assert any(call[1:4] == ["rm", "--force", "--volumes"] for call in calls)
    assert watchdog.triggered and not watchdog.disarmed
