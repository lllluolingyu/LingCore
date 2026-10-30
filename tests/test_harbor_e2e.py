"""Opt-in real Docker integration; LingCore itself never imports Harbor."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from lingcore.evaluation.harbor.process import bridge_environment
from lingcore.evaluation.harbor.runner import check_harness_stopped
from lingcore.evaluation.lock import resolve, validate, write_lock
from lingcore.evaluation.regrade import regrade
from lingcore.evaluation.runner import resume, run_locked
from lingcore.message import ToolCall
from tests.fakes import FakeLLMClient, ScriptedTurn

pytestmark = pytest.mark.skipif(
    os.environ.get("LINGCORE_HARBOR_E2E") != "1",
    reason="opt-in Docker and separately installed Harbor bridge",
)


@pytest.fixture
def synthetic(tmp_path):
    bridge = Path(os.environ["LINGCORE_HARBOR_BRIDGE"])
    task = tmp_path / "dataset" / ("probe-" + uuid4().hex[:10])
    shutil.copytree(
        Path(__file__).parents[1]
        / "docs/examples/evaluation/harbor-synthetic/write-file",
        task,
    )
    (
        tmp_path / "agent.yaml"
    ).write_text("""llm: {model: fake, api_key_env: FIXTURE_API_KEY, modalities: []}
tools: [terminal_exec, terminal_read_output]
sessions: {enabled: false}
loop: {parallel_tools: false}
persona: {project_instructions: []}
""")
    suite = dict(
        schema_version="lingcore.eval.suite.v1",
        name="docker-synthetic",
        source={
            "kind": "harbor",
            "dataset": "synthetic",
            "path": "dataset",
            "selection": {"policy": "sorted_capability_subset", "count": 1},
        },
        agent={"profile": "agent.yaml", "tool_binding": "harbor-terminal-v1"},
        execution={
            "runner": "harbor",
            "approval_policy": "benchmark_environment",
            "limits": {"wall_seconds": 30},
        },
        grading={"authority": "harbor"},
        trace={
            "content": "redacted",
            "exports": ["atif"],
            "artifact_max_bytes": 4194304,
        },
    )
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite))
    return tmp_path, task, bridge


def client(command):
    return lambda _: FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="command",
                        name="terminal_exec",
                        arguments={"command": command},
                    )
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )


async def freeze(root, bridge):
    path = root / "lock.json"
    lock = await resolve(root / "suite.yaml", path, bridge=bridge)
    write_lock(lock, path)
    assert not await validate(lock, path, bridge=bridge)
    return path


async def test_pass_fail_regrade_and_original_immutability(synthetic, monkeypatch):
    root, task, bridge = synthetic
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret-never-forwarded")
    path = await freeze(root, bridge)
    job, passed = await run_locked(
        path,
        root / "jobs",
        bridge=bridge,
        client_factory=client("mkdir -p /app; printf 'harbor-ok\\n' > /app/answer.txt"),
    )
    assert passed["complete"] and passed["mean_reward"] == 1
    execution = next(job.glob("executions/*/result.json"))
    result = json.loads(execution.read_bytes())
    assert result["atif"] == "written" and result["trace"]["complete"]
    from lingcore.tracing.inspect import inspect_trace

    assert inspect_trace(execution.parent / "trace").complete
    failed_job, failed = await run_locked(
        path,
        root / "jobs",
        bridge=bridge,
        client_factory=client("mkdir -p /app; echo wrong > /app/answer.txt"),
    )
    assert failed["complete"] and failed["mean_reward"] == 0
    originals = {p: p.read_bytes() for p in failed_job.rglob("*") if p.is_file()}
    test = task / "tests/test.sh"
    test.write_text(test.read_text().replace("= harbor-ok", "= wrong"))
    graded = await regrade(failed_job, bridge=bridge)
    assert graded["complete"] and graded["mean_reward"] == 1
    assert all(p.read_bytes() == data for p, data in originals.items())
    for file in root.rglob("*"):
        if file.is_file():
            assert b"fixture-secret-never-forwarded" not in file.read_bytes()


async def test_cancel_mid_command_removes_container_and_resume(synthetic):
    root, task, bridge = synthetic
    path = await freeze(root, bridge)
    active = asyncio.create_task(
        run_locked(
            path, root / "jobs", bridge=bridge, client_factory=client("sleep 1000")
        )
    )
    try:
        async with asyncio.timeout(90):
            while True:
                traces = list((root / "jobs").glob("*/executions/*/trace/events.jsonl"))
                if traces and "tool.execute" in traces[0].read_text():
                    break
                await asyncio.sleep(0.1)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        job = next((root / "jobs").iterdir())
        execution = next(job.glob("executions/*"))
        check_harness_stopped(execution)
        assert not subprocess.check_output(
            ["docker", "ps", "-q", "--filter", "name=" + task.name]
        ).strip()
        report = await resume(
            job,
            bridge=bridge,
            client_factory=client("mkdir -p /app; echo harbor-ok > /app/answer.txt"),
        )
        assert report["complete"] and report["mean_reward"] == 1
    finally:
        if not active.done():
            active.cancel()
            await asyncio.gather(active, return_exceptions=True)


@pytest.mark.parametrize("agent,reward", [("oracle", 1), ("nop", 0)])
def test_harbor_reference_agents(synthetic, agent, reward):
    root, task, bridge = synthetic
    from lingcore.config import AgentProfile

    env = bridge_environment(
        AgentProfile.model_validate({"llm": {"model": "unused"}}), []
    )
    env["HARBOR_TELEMETRY"] = "off"
    code = "import sys; from harbor.cli.main import app; app(args=sys.argv[1:], standalone_mode=False)"
    result = subprocess.run(
        [
            str(bridge.parent / "python"),
            "-c",
            code,
            "run",
            "-p",
            str(task),
            "-a",
            agent,
            "-k",
            "1",
            "-n",
            "1",
            "--max-retries",
            "0",
            "--jobs-dir",
            str(root / "reference"),
            "--job-name",
            "reference",
            "--quiet",
        ],
        env=env,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode()[-4000:]
    files = list((root / "reference").glob("*/*/result.json"))
    assert len(files) == 1
    assert (
        json.loads(files[0].read_bytes())["verifier_result"]["rewards"]["reward"]
        == reward
    )


async def test_process_crash_preserves_harbor_and_resumes(synthetic):
    import sys

    from lingcore.evaluation.config import EvaluationConfigError

    root, task, bridge = synthetic
    path = await freeze(root, bridge)
    code = """import asyncio,sys
from pathlib import Path
from lingcore.evaluation.runner import run_locked
from lingcore.message import ToolCall
from tests.fakes import FakeLLMClient,ScriptedTurn
asyncio.run(run_locked(Path(sys.argv[1]),Path(sys.argv[2]),bridge=Path(sys.argv[3]),client_factory=lambda _:FakeLLMClient([ScriptedTurn(tool_calls=[ToolCall(id="c",name="terminal_exec",arguments={"command":"sleep 1000"})])])))
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        str(path),
        str(root / "jobs"),
        str(bridge),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(90):
            while True:
                traces = list((root / "jobs").glob("*/executions/*/trace/events.jsonl"))
                if traces and "tool.execute" in traces[0].read_text():
                    break
                assert process.returncode is None
                await asyncio.sleep(0.1)
        process.kill()
        await process.wait()
        job = next((root / "jobs").iterdir())
        execution = next(job.glob("executions/*"))
        async with asyncio.timeout(120):
            while True:
                try:
                    check_harness_stopped(execution)
                    break
                except EvaluationConfigError:
                    await asyncio.sleep(0.1)
        assert not (execution / "result.json").exists()
        assert list((execution / "harbor").glob("*/*/result.json"))
        originals = {
            p: p.read_bytes() for p in (execution / "harbor").rglob("*") if p.is_file()
        }
        report = await resume(
            job,
            bridge=bridge,
            client_factory=client("mkdir -p /app; echo harbor-ok > /app/answer.txt"),
        )
        assert report["complete"] and report["coverage"]["replaced"] == 1
        assert all(p.read_bytes() == data for p, data in originals.items())
        assert not subprocess.check_output(
            ["docker", "ps", "-q", "--filter", "name=" + task.name]
        ).strip()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_wall_deadline_interrupts_and_finishes_for_grading(synthetic):
    root, task, bridge = synthetic
    source = root / "suite.yaml"
    suite = yaml.safe_load(source.read_text())
    suite["execution"]["limits"]["wall_seconds"] = 0.5
    source.write_text(yaml.safe_dump(suite))
    path = await freeze(root, bridge)
    job, report = await run_locked(
        path, root / "jobs", bridge=bridge, client_factory=client("sleep 1000")
    )
    result = json.loads(next(job.glob("executions/*/result.json")).read_bytes())
    assert (
        result["execution"]["status"] == "timed_out"
        and result["execution"]["limit_type"] == "wall_seconds"
    )
    assert result["grade"]["status"] == "valid" and result["rewards"] == {"reward": 0.0}
    assert result["infrastructure_failure"] is None
    assert report["complete"] and report["valid_failures"] == 1
    assert not subprocess.check_output(
        ["docker", "ps", "-q", "--filter", "name=" + task.name]
    ).strip()
