from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import yaml

from lingcore.evaluation.harbor.outcomes import map_outcome
from lingcore.evaluation.harbor.source import select
from lingcore.evaluation.identity import identity_body
from lingcore.evaluation.lock import resolve, validate, write_lock
from lingcore.evaluation.models import HarborSource, RewardPolicy
from lingcore.evaluation.runner import resume, run_locked
from lingcore.execution import ExecutionOutcome
from lingcore.message import ToolCall
from tests.fakes import FakeLLMClient, ScriptedTurn


@pytest.fixture
def harbor_plan(tmp_path):
    task = tmp_path / "dataset" / "tiny"
    task.mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment").mkdir()
    (task / "instruction.md").write_text("Write a file.\n")
    (task / "tests/test.sh").write_text("test -f /app/answer")
    (task / "environment/Dockerfile").write_text("FROM fixture")
    desc = dict(
        status="ok",
        config={
            "agent": {"timeout_sec": 180},
            "environment": {},
            "verifier": {"timeout_sec": 1},
            "artifacts": ["/app/"],
        },
        properties=dict(
            os="linux",
            gpus=0,
            containers=1,
            cpus=1,
            memory_mb=256,
            storage_mb=1024,
            task_tools=False,
            steps=0,
            verifier_mode="separate",
        ),
        environments={
            "agent": {"kind": "image", "pinned": True},
            "verifier": {"kind": "image", "pinned": True},
        },
    )
    (task / "description.json").write_text(json.dumps(desc))
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
        name="harbor-test",
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
            "harbor": {"pass_env": ["FIXTURE_SCENARIO"]},
            "limits": {"wall_seconds": 1},
        },
        grading={"authority": "harbor"},
        trace={"content": "redacted", "exports": ["atif"]},
    )
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    bridge = tmp_path / "bridge"
    bridge.write_text(
        "#!"
        + sys.executable
        + "\n"
        + (Path(__file__).parent / "fake_harbor_bridge.py").read_text()
    )
    bridge.chmod(0o700)
    return path, bridge


def client(command="write"):
    return lambda _: FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="call", name="terminal_exec", arguments={"command": command}
                    )
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )


async def frozen(harbor_plan):
    path, bridge = harbor_plan
    lock_path = path.parent / "lock.json"
    lock = await resolve(path, lock_path, bridge=bridge)
    write_lock(lock, lock_path)
    return lock, lock_path, bridge


async def test_harbor_lock_drift_split_and_binding(harbor_plan):
    lock, path, bridge = await frozen(harbor_plan)
    assert not await validate(lock, path, bridge=bridge)
    assert lock.candidate.tools.binding_id == "harbor-terminal-v1"
    assert "path" not in identity_body(lock)["source"]["dataset"]
    (path.parent / "dataset/tiny/tests/test.sh").write_text("changed test")
    drift = await validate(lock, path, bridge=bridge)
    assert any(d.field == "source.tasks.tiny.verifier_digest" for d in drift)
    assert not any("input_digest" in d.field for d in drift)


@pytest.mark.parametrize(
    "scenario", ["", "wait_interrupt", "malformed", "unsupported", "before_hello"]
)
async def test_runner_outcomes_and_real_environment_filter(
    harbor_plan, monkeypatch, scenario
):
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret-never-forwarded")
    monkeypatch.setenv("FIXTURE_SCENARIO", scenario)
    _, path, bridge = await frozen(harbor_plan)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    results = list(job.glob("executions/*/result.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_bytes())
    if scenario == "":
        assert (
            result["grade"]["passed"]
            and report["mean_reward"] == 1
            and report["harbor_accuracy"] == 1
        )
    elif scenario == "wait_interrupt":
        assert result["execution"]["status"] == "timed_out"
        assert result["grade"]["status"] == "valid"
        frames = json.loads(
            next(job.glob("executions/*/harbor/job/trial/frames.json")).read_bytes()
        )
        assert [f["type"] for f in frames if f["type"] != "usage"] == [
            "exec",
            "interrupt",
            "finish",
        ]
    elif scenario == "unsupported":
        assert result["unsupported"] and report["coverage"]["unsupported"] == 1
        assert result["infrastructure_failure"] is None
        assert report["trials"][0]["reason"] == "unsupported"
    else:
        assert result["execution"]["cause"] == "runner"
    for file in job.rglob("*"):
        if file.is_file():
            assert b"fixture-secret-never-forwarded" not in file.read_bytes()


async def test_regrade_preserves_original_records_and_never_constructs_agent(
    harbor_plan, monkeypatch
):
    _, path, bridge = await frozen(harbor_plan)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client("fail")
    )
    assert report["mean_reward"] == 0
    before = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}
    (path.parent / "dataset/tiny/tests/test.sh").write_text("corrected")
    from lingcore.agent import Agent
    from lingcore.evaluation.regrade import regrade

    monkeypatch.setattr(
        Agent,
        "from_profile",
        lambda *a, **k: pytest.fail("Agent created during regrade"),
    )
    result = await regrade(job, bridge=bridge)
    assert result["mean_reward"] == 1
    assert all(p.read_bytes() == value for p, value in before.items())


async def test_resume_refuses_running_harness(harbor_plan):
    import fcntl

    from lingcore.evaluation.config import EvaluationConfigError

    _, path, bridge = await frozen(harbor_plan)
    job, _ = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    result = next(job.glob("executions/*/result.json"))
    result.unlink()
    with (result.parent / "harbor.lock").open("w") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        with pytest.raises(EvaluationConfigError, match="still running"):
            await resume(job, bridge=bridge, client_factory=client())
    report = await resume(job, bridge=bridge, client_factory=client())
    assert report["complete"] and report["coverage"]["replaced"] == 1


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("gpus", 1, "gpus"),
        ("containers", 2, "containers"),
        ("steps", 1, "steps"),
        ("task_tools", True, "task_tools"),
        ("cpus", 100, "cpus"),
    ],
)
def test_selection_uses_metadata(field, value, reason):
    source = HarborSource.model_validate(
        {
            "kind": "harbor",
            "dataset": "synthetic",
            "path": ".",
            "selection": {
                "policy": "sorted_capability_subset",
                "count": 1,
                "required": {"max_cpus": 4},
            },
        }
    )
    properties = dict(
        os="linux", gpus=0, containers=1, steps=0, task_tools=False, cpus=1
    )
    selected, excluded = select(
        source,
        {
            "tasks": [
                {"id": "a", "status": "ok", "properties": properties},
                {"id": "b", "status": "ok", "properties": properties | {field: value}},
            ]
        },
    )
    assert [t["id"] for t in selected] == ["a"]
    assert reason in excluded[0]["reasons"]


@pytest.mark.parametrize(
    "cause,hello,outcome,kwargs,expected",
    [
        (
            "sandbox_failed",
            False,
            "completed",
            {},
            ("failed", "environment", "not_run", "environment_failed"),
        ),
        (
            None,
            True,
            "completed",
            {"unsupported": True},
            ("failed", "environment", "not_run", "unsupported"),
        ),
        (
            "harbor_failed",
            True,
            "completed",
            {},
            ("failed", "runner", "not_run", "harbor_failed"),
        ),
        (
            None,
            True,
            "completed",
            {"cleanup_failed": True},
            ("completed", None, "invalid", "cleanup_failed"),
        ),
        (
            None,
            True,
            "completed",
            {"protocol_failed": True},
            ("failed", "runner", "invalid", "protocol_failed"),
        ),
        (
            "verifier_failed",
            True,
            "completed",
            {},
            ("completed", None, "invalid", "verifier_failed"),
        ),
        (
            "sandbox_failed",
            True,
            "completed",
            {},
            ("completed", None, "invalid", "reward_invalid"),
        ),
    ],
)
def test_ordered_outcome_table(cause, hello, outcome, kwargs, expected):
    summary = {
        "failure_cause": cause,
        "reward_valid": cause is None,
        "rewards": {"reward": 1},
    }
    execution, grade = map_outcome(
        summary, ExecutionOutcome(status=outcome), RewardPolicy(), hello=hello, **kwargs
    )
    assert (execution.status, execution.cause, grade.status, grade.reason) == expected


async def test_unsolicited_abort_interrupts_model_between_commands(
    harbor_plan, monkeypatch
):
    from lingcore.llm import LLMChunk

    monkeypatch.setenv("FIXTURE_SCENARIO", "abort_idle")
    _, path, bridge = await frozen(harbor_plan)

    class WaitingClient:
        async def stream(self, messages, tools=None):
            await asyncio.sleep(1000)
            yield LLMChunk(text_delta="never")

    job, report = await run_locked(
        path,
        path.parent / "jobs",
        bridge=bridge,
        client_factory=lambda _: WaitingClient(),
    )
    result = json.loads(next(job.glob("executions/*/result.json")).read_bytes())
    assert result["execution"]["limit_type"] == "harness_agent"
    assert result["grade"]["status"] == "valid" and report["coverage"]["retried"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"sessions": {"enabled": True}},
        {"loop": {"parallel_tools": True}},
        {"persona": {"project_instructions": ["AGENTS.md"]}},
        {"tools": ["run_shell"]},
        {"tool_options": {"terminal_exec": {"capture_bytes": -1}}},
    ],
)
async def test_harbor_profile_refusals(harbor_plan, bad):
    from lingcore.evaluation.config import EvaluationConfigError

    path, bridge = harbor_plan
    profile = path.parent / "agent.yaml"
    value = yaml.safe_load(profile.read_text()) | bad
    profile.write_text(yaml.safe_dump(value))
    with pytest.raises(EvaluationConfigError):
        await resolve(path, path.parent / "lock.json", bridge=bridge)


def test_profile_credentials_and_local_env_never_enter_bridge_env(monkeypatch):
    from lingcore.config import AgentProfile
    from lingcore.evaluation.config import EvaluationConfigError
    from lingcore.evaluation.harbor.process import bridge_environment

    profile = AgentProfile.model_validate(
        {"llm": {"model": "fake", "api_key_env": "SPECIAL_CREDENTIAL"}}
    )
    object.__setattr__(
        profile, "_profile_env", {"LOCAL_SETTING": "private-local-value"}
    )
    monkeypatch.setenv("SPECIAL_CREDENTIAL", "secret")
    with pytest.raises(EvaluationConfigError):
        bridge_environment(profile, ["SPECIAL_CREDENTIAL"])
    env = bridge_environment(profile, ["LOCAL_SETTING"])
    assert "SPECIAL_CREDENTIAL" not in env and "LOCAL_SETTING" not in env


@pytest.mark.parametrize("value", [None, 2, float("nan"), True])
def test_invalid_named_reward(value):
    summary = {"reward_valid": True, "rewards": {"named": value}}
    _, grade = map_outcome(
        summary,
        ExecutionOutcome(status="completed"),
        RewardPolicy(name="named"),
        hello=True,
    )
    assert grade.status == "invalid"


async def test_regrade_subprocess_cancellation_joins_teardown(tmp_path):
    import sys

    from lingcore.evaluation.harbor.process import invoke_cancellable

    bridge = tmp_path / "bridge"
    marker = tmp_path / "terminated"
    ready = tmp_path / "ready"
    bridge.write_text(
        "#!"
        + sys.executable
        + "\n"
        + f"""import signal,time
from pathlib import Path
def stop(*args):
    Path({str(marker)!r}).write_text('done')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path({str(ready)!r}).write_text('ready')
time.sleep(1000)
"""
    )
    bridge.chmod(0o700)
    task = asyncio.create_task(invoke_cancellable(bridge, [], {}, timeout=1000))
    async with asyncio.timeout(5):
        while not ready.exists():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert marker.read_text() == "done"


async def test_private_credential_in_model_command_is_not_forwarded(
    harbor_plan, monkeypatch
):
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret-never-forwarded")
    _, path, bridge = await frozen(harbor_plan)
    job, report = await run_locked(
        path,
        path.parent / "jobs",
        bridge=bridge,
        client_factory=client("echo fixture-secret-never-forwarded"),
    )
    assert report["complete"] and report["mean_reward"] == 0
    frames = json.loads(
        next(job.glob("executions/*/harbor/job/trial/frames.json")).read_bytes()
    )
    assert not any(f["type"] == "exec" for f in frames)
    assert all(
        b"fixture-secret-never-forwarded" not in p.read_bytes()
        for p in job.rglob("*")
        if p.is_file()
    )


@pytest.mark.parametrize("scenario", ["unsupported", "malformed", "environment_once"])
async def test_regrade_only_selected_gradable_attempts(
    harbor_plan, monkeypatch, scenario
):
    from lingcore.evaluation.regrade import regrade

    suite_path, _ = harbor_plan
    suite = yaml.safe_load(suite_path.read_text())
    suite["execution"]["infrastructure_retries"] = 1
    suite_path.write_text(yaml.safe_dump(suite))
    monkeypatch.setenv("FIXTURE_SCENARIO", scenario)
    _, path, bridge = await frozen(harbor_plan)
    job, original = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client("fail")
    )
    before = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}
    results = {
        p.parent.name: json.loads(p.read_bytes())
        for p in job.glob("executions/*/result.json")
    }
    report = await regrade(job, bridge=bridge)
    assert report["complete"] == original["complete"]
    assert all(p.read_bytes() == value for p, value in before.items())
    grades = list(job.glob("executions/*/grades/regrade-*.json"))
    assert len(grades) == len(results)
    for path in grades:
        original_result = results[path.parent.parent.name]
        grade = json.loads(path.read_bytes())["grade"]
        if original_result["grade"]["status"] != "valid":
            assert grade == original_result["grade"]
            assert not (path.parent.parent / "regrades").exists()
    if scenario == "environment_once":
        assert len(results) == 2 and report["harbor_accuracy"] == 1
        assert len(list(job.glob("executions/*/regrades/*/harbor-summary.json"))) == 1
    else:
        assert report["harbor_accuracy"] == original["harbor_accuracy"] == 0
        assert report["coverage"]["unsupported"] == original["coverage"]["unsupported"]


async def test_regrade_forwards_locked_environment(harbor_plan, monkeypatch):
    from lingcore.evaluation.regrade import regrade

    monkeypatch.setenv("FIXTURE_SCENARIO", "forwarded")
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret-never-forwarded")
    _, path, bridge = await frozen(harbor_plan)
    job, _ = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    await regrade(job, bridge=bridge)
    env = json.loads(
        next(job.glob("executions/*/regrades/*/harbor/environment.json")).read_bytes()
    )
    assert env == {"FIXTURE_SCENARIO": "forwarded"}


@pytest.mark.parametrize("name", ["agent", "verifier", "setup", "build"])
async def test_regrade_refuses_unpreservable_timeouts(harbor_plan, name):
    from lingcore.evaluation.config import EvaluationConfigError
    from lingcore.evaluation.regrade import regrade

    suite_path, _ = harbor_plan
    suite = yaml.safe_load(suite_path.read_text())
    suite["execution"]["harbor"]["timeout_multipliers"] = {name: 2}
    suite_path.write_text(yaml.safe_dump(suite))
    _, path, bridge = await frozen(harbor_plan)
    job, _ = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    with pytest.raises(EvaluationConfigError, match="cannot preserve"):
        await regrade(job, bridge=bridge)
    assert not list(job.glob("executions/*/regrades"))


@pytest.mark.parametrize(
    "scenario", ["timeout_finish", "timeout_finish_eof", "digest_mismatch"]
)
async def test_finish_timeout_and_input_drift_never_retry(
    harbor_plan, monkeypatch, scenario
):
    monkeypatch.setenv("FIXTURE_SCENARIO", scenario)
    suite_path, _ = harbor_plan
    suite = yaml.safe_load(suite_path.read_text())
    suite["execution"]["infrastructure_retries"] = 1
    suite_path.write_text(yaml.safe_dump(suite))
    _, path, bridge = await frozen(harbor_plan)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    results = list(job.glob("executions/*/result.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_bytes())
    assert result["infrastructure_failure"] is None
    if scenario == "digest_mismatch":
        assert result["execution"]["cause"] == "configuration"
        assert result["grade"]["reason"] == "task_digest_mismatch"
    else:
        assert result["execution"]["status"] == "timed_out"
        assert result["execution"]["limit_type"] == "harness_agent"
        assert result["grade"]["status"] == "valid"
        assert report["complete"]


@pytest.mark.parametrize("crash", [False, True])
async def test_export_flushes_and_runs_off_loop_without_affecting_grade(
    harbor_plan, monkeypatch, crash
):
    import threading

    from lingcore.evaluation.harbor import trajectory
    from lingcore.tracing.local import LocalTraceSink

    _, path, bridge = await frozen(harbor_plan)
    original_export = trajectory.export_trajectory
    original_init = LocalTraceSink.__init__
    owner = threading.get_ident()
    exported = []

    def slow_flush(self, *args, **kwargs):
        kwargs["flush_interval_seconds"] = 3600
        original_init(self, *args, **kwargs)

    def export(*args):
        assert threading.get_ident() != owner
        if crash:
            raise RuntimeError("export failed")
        value = original_export(*args)
        exported.append(value)
        return value

    monkeypatch.setattr(LocalTraceSink, "__init__", slow_flush)
    monkeypatch.setattr(trajectory, "export_trajectory", export)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    result = json.loads(next(job.glob("executions/*/result.json")).read_bytes())
    assert report["harbor_accuracy"] == 1
    if crash:
        assert result["atif"] == "absent" and result["atif_reason"] == "incomplete"
    else:
        assert exported[0]["steps"][-1]["message"] == "done"
        assert len(exported[0]["extra"]["lingcore"]["requests"]) == 2
        assert result["atif"] == "written"


async def test_margin_is_not_a_reduced_limit_and_platform_comes_from_execution(
    harbor_plan, monkeypatch
):
    import platform

    suite_path, _ = harbor_plan
    suite = yaml.safe_load(suite_path.read_text())
    suite["execution"]["limits"]["wall_seconds"] = 30  # 180 - 150 reserved
    suite_path.write_text(yaml.safe_dump(suite))
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    lock, path, bridge = await frozen(harbor_plan)
    assert "reduced_limits" not in lock.limitations
    assert "emulated_platform" not in lock.limitations
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    _, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    assert "emulated_platform" not in report["limitations"]
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    _, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    assert "emulated_platform" in report["limitations"]


@pytest.mark.parametrize("race", [False, True])
async def test_process_cleanup_checks_group_after_leader_exit(monkeypatch, race):
    import signal
    from types import SimpleNamespace

    from lingcore.evaluation.harbor.runner import stop_process

    calls, waits = [], []

    async def wait():
        waits.append(True)
        if race and len(waits) == 1:
            raise TimeoutError
        return 0

    def killpg(pid, sig):
        calls.append(sig)
        if sig in {0, signal.SIGKILL}:
            raise ProcessLookupError

    monkeypatch.setattr("lingcore.evaluation.harbor.runner.os.killpg", killpg)
    assert (
        await stop_process(SimpleNamespace(pid=123, returncode=0, wait=wait))
        is not race
    )
    assert calls == [signal.SIGTERM, signal.SIGKILL if race else 0]


@pytest.mark.parametrize(
    "content,reason", [("metadata", "metadata_only"), ("redacted", "disabled")]
)
async def test_disabled_export_reason(harbor_plan, content, reason):
    suite_path, _ = harbor_plan
    suite = yaml.safe_load(suite_path.read_text())
    suite["trace"] = {"content": content}
    suite_path.write_text(yaml.safe_dump(suite))
    _, path, bridge = await frozen(harbor_plan)
    job, _ = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    result = json.loads(next(job.glob("executions/*/result.json")).read_bytes())
    assert result["atif_reason"] == reason


async def test_profile_terminal_options_match_identity_and_wire(harbor_plan):
    suite_path, _ = harbor_plan
    profile_path = suite_path.parent / "agent.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["tool_options"] = {"terminal_exec": {"capture_bytes": 23, "max_timeout": 5}}
    profile_path.write_text(yaml.safe_dump(profile))
    lock, path, bridge = await frozen(harbor_plan)
    assert lock.candidate.tools.options["capture_bytes"] == 23
    job, _ = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    frames = json.loads(
        next(job.glob("executions/*/harbor/job/trial/frames.json")).read_bytes()
    )
    command = next(f for f in frames if f["type"] == "exec")
    assert command["capture_bytes"] == 23 and command["timeout"] == 5


async def test_deadline_reserves_probe_interrupt_sweep_and_export(harbor_plan):
    from lingcore.evaluation.config import EvaluationConfigError

    suite_path, _ = harbor_plan
    suite = yaml.safe_load(suite_path.read_text())
    suite["execution"]["limits"]["wall_seconds"] = 31
    suite_path.write_text(yaml.safe_dump(suite))
    with pytest.raises(EvaluationConfigError, match="150 seconds"):
        await frozen(harbor_plan)
