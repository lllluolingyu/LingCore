from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import SourceTrialConfig
from harbor.models.trial.result import TrialResult
from harbor.publisher.packager import Packager

IMAGE = "python@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f"
TESTS = Path(__file__).parent


def environment(tmp_path):
    values = {
        key: os.environ[key]
        for key in (
            "PATH",
            "USER",
            "LANG",
            "LC_ALL",
            "DOCKER_HOST",
            "DOCKER_CONFIG",
            "DOCKER_CONTEXT",
        )
        if key in os.environ
    }
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    values.update(HOME=str(home), HF_HUB_OFFLINE="1", HARBOR_TELEMETRY="off")
    return values


def command(args):
    # The exact in-process Typer entry point that the bridge's trial command
    # will call. The outer process supplies signal isolation for this test.
    code = "import sys; sys.path.insert(0, sys.argv[1]); from harbor.cli.main import app; app(args=sys.argv[2:], standalone_mode=False)"
    return [sys.executable, "-c", code, str(TESTS), *args]


def task(tmp_path):
    root = tmp_path / ("probe-" + uuid4().hex[:10])
    root.mkdir()
    (root / "tests").mkdir()
    (root / "instruction.md").write_text("Synthetic compatibility probe.\n")
    (root / "task.toml").write_text(f'''schema_version = "2.0"
[agent]
timeout_sec = 180
[verifier]
timeout_sec = 30
[environment]
docker_image = "{IMAGE}"
cpus = 1
memory_mb = 256
''')
    (root / "tests/test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier\nprintf '0' > /logs/verifier/reward.txt\n"
    )
    return root


def test_cli_entrypoint_and_exact_single_trial_flags(tmp_path):
    root = task(tmp_path)
    result = subprocess.run(
        command(
            [
                "run",
                "-p",
                str(root),
                "-a",
                "lingcore_harbor_bridge.agent:BridgeAgent",
                "-k",
                "1",
                "-n",
                "1",
                "--max-retries",
                "0",
                "--print-config",
            ]
        ),
        env=environment(tmp_path),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    config = JobConfig.model_validate_json(result.stdout)
    assert config.n_attempts == 1
    assert config.n_concurrent_trials == 1
    assert config.retry.max_retries == 0
    assert len(config.tasks) == len(config.agents) == 1


def test_local_source_trial_regrade_configuration(tmp_path):
    config = SourceTrialConfig(action="regrade", type="local", path=tmp_path)
    assert config.model_dump(mode="json")["action"] == "regrade"
    assert config.path == tmp_path


docker_test = pytest.mark.skipif(
    os.environ.get("LINGCORE_HARBOR_E2E") != "1",
    reason="opt-in Docker compatibility check",
)


@docker_test
def test_one_trial_and_lock_digest_with_real_docker(tmp_path):
    root = task(tmp_path)
    output = tmp_path / "jobs"
    result = subprocess.run(
        command(
            [
                "run",
                "-p",
                str(root),
                "-a",
                "nop",
                "-k",
                "1",
                "-n",
                "1",
                "--max-retries",
                "0",
                "--jobs-dir",
                str(output),
                "--job-name",
                "single",
                "--quiet",
            ]
        ),
        env=environment(tmp_path),
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode()[-6000:]
    trials = list((output / "single").glob("*/result.json"))
    assert len(trials) == 1
    trial = TrialResult.model_validate_json(trials[0].read_bytes())
    assert trial.reward_valid
    assert trial.verifier_result.rewards == {"reward": 0.0}
    lock = json.loads((trials[0].parent / "lock.json").read_bytes())
    assert lock["task"]["digest"] == "sha256:" + Packager.compute_content_hash(root)[0]


def containers(name):
    return (
        subprocess.check_output(
            ["docker", "ps", "-q", "--filter", "name=" + name], timeout=10
        )
        .decode()
        .split()
    )


@docker_test
def test_sigterm_removes_the_task_container(tmp_path):
    root = task(tmp_path)
    log = tmp_path / "harbor.log"
    with log.open("wb") as capture:
        process = subprocess.Popen(
            command(
                [
                    "run",
                    "-p",
                    str(root),
                    "-a",
                    "probe_agent:HoldingAgent",
                    "-k",
                    "1",
                    "-n",
                    "1",
                    "--max-retries",
                    "0",
                    "--jobs-dir",
                    str(tmp_path / "jobs"),
                    "--job-name",
                    "cancel",
                    "--quiet",
                ]
            ),
            env=environment(tmp_path),
            stdout=capture,
            stderr=capture,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 90
            while not containers(root.name):
                assert process.poll() is None, log.read_text()[-6000:]
                assert time.monotonic() < deadline
                time.sleep(0.1)
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=120)
            assert not containers(root.name), log.read_text()[-6000:]
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            for container in containers(root.name):
                subprocess.run(
                    ["docker", "rm", "-f", container],
                    check=True,
                    capture_output=True,
                    timeout=20,
                )


def test_cli_normalizes_harbor_system_exit(monkeypatch, capfd):
    from lingcore_harbor_bridge import cli, commands

    def stop(_):
        raise SystemExit(2)

    monkeypatch.setattr(commands, "summarize", stop)
    assert cli.main(["summarize", "--output", "/unused"]) == 2
    import json

    assert json.loads(capfd.readouterr().out) == {
        "error": {"code": "harbor_exit", "exit_code": 2}
    }
