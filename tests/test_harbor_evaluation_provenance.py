"""Harbor evaluation tests: provenance."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest
import yaml

from lingcore.evaluation.harbor.outcomes import map_outcome
from lingcore.evaluation.lock import resolve
from lingcore.evaluation.models import RewardPolicy
from lingcore.evaluation.runner import run_locked
from lingcore.execution import ExecutionOutcome
from tests.support.harbor_evaluation import client, frozen
from tests.support.harbor_evaluation import harbor_plan as harbor_plan
from tests.support.harbor_evaluation import trusted_release as trusted_release


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
    from lingcore.evaluation.compare import compare_jobs
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
    comparison = compare_jobs(
        job, job, baseline_grades="latest", candidate_grades="latest"
    )
    assert comparison["coverage"]["planned_pairs"] == 1
    assert comparison["complete"] == report["complete"]
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


@pytest.mark.parametrize("name", ["verifier", "setup", "build"])
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
