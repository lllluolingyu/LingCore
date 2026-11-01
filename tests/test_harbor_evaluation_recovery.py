"""Harbor evaluation tests: recovery."""

from __future__ import annotations

import json

import pytest
import yaml

from lingcore.evaluation.runner import run_locked
from tests.support.harbor_evaluation import client, frozen
from tests.support.harbor_evaluation import harbor_plan as harbor_plan
from tests.support.harbor_evaluation import trusted_release as trusted_release


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


@pytest.mark.parametrize("reason", ["verifier_failed", "reward_invalid"])
async def test_regrade_repairs_verifier_failures(harbor_plan, monkeypatch, reason):
    from lingcore.evaluation.regrade import regrade

    monkeypatch.setenv("FIXTURE_SCENARIO", reason)
    _, path, bridge = await frozen(harbor_plan)
    job, original = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    result_path = next(job.glob("executions/*/result.json"))
    result = json.loads(result_path.read_bytes())
    assert result["grade"]["reason"] == reason and original["harbor_accuracy"] == 0
    before = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}
    (path.parent / "dataset/tiny/tests/test.sh").write_text("corrected verifier")
    report = await regrade(job, bridge=bridge)
    assert report["complete"] and report["harbor_accuracy"] == 1

    assert report["regrade_skipped"] == []
    grade = json.loads(
        next(job.glob("executions/*/grades/regrade-*.json")).read_bytes()
    )
    assert grade["grade"]["status"] == "valid" and grade["grade"]["passed"]
    assert grade["reward_valid"] and grade["rewards"] == {"reward": 1.0}
    assert all(p.read_bytes() == value for p, value in before.items())


async def test_regrade_compares_mixed_gradable_and_environment_failed_cases(
    harbor_plan, monkeypatch
):
    import shutil

    from lingcore.evaluation.compare import compare_jobs
    from lingcore.evaluation.regrade import regrade

    suite_path, _ = harbor_plan
    task = suite_path.parent / "dataset/tiny"
    shutil.copytree(task, task.with_name("z-healthy"))
    suite = yaml.safe_load(suite_path.read_text())
    suite["source"]["selection"]["count"] = 2
    suite["execution"]["infrastructure_retries"] = 0
    suite_path.write_text(yaml.safe_dump(suite))
    monkeypatch.setenv("FIXTURE_SCENARIO", "environment_once")
    lock, path, bridge = await frozen(harbor_plan)
    job, _ = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client("fail")
    )
    report = await regrade(job, bridge=bridge)
    header = json.loads(
        (job / "grade-sets" / (report["grade_set"] + ".json")).read_bytes()
    )
    assert {c["id"] for c in header["cases"]} == {t.id for t in lock.source.tasks}
    compared = compare_jobs(job, job, candidate_grades="latest")
    assert not compared["complete"]
    assert compared["coverage"]["valid_pairs"] == 1
    assert compared["coverage"]["unpaired"] == 1


@pytest.mark.parametrize("previous", [False, True])
async def test_failed_harbor_regrade_keeps_latest_completed_report(
    harbor_plan, monkeypatch, previous
):
    import importlib

    from lingcore.evaluation.config import EvaluationConfigError
    from lingcore.evaluation.regrade import read_report, regrade

    module = importlib.import_module("lingcore.evaluation.harbor.regrade")
    _, path, bridge = await frozen(harbor_plan)
    job, original = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    completed = await regrade(job, bridge=bridge) if previous else original

    async def fail(*args, **kwargs):
        raise EvaluationConfigError("bridge failed", "bridge_failed")

    monkeypatch.setattr(module, "invoke_cancellable", fail)
    with pytest.raises(EvaluationConfigError, match="bridge failed"):
        await regrade(job, bridge=bridge)
    assert read_report(job, "latest")["grade_set"] == completed["grade_set"]
    failed = [
        json.loads(p.read_bytes())
        for p in job.glob("grade-sets/regrade-*.json")
        if not json.loads(p.read_bytes())["complete"]
    ]
    assert len(failed) == 1
    assert not read_report(job, failed[0]["id"])["complete"]


@pytest.mark.parametrize("missing", ["regrade_error", "trial_record"])
async def test_regrade_missing_evidence_skips_only_affected_attempt(
    harbor_plan, monkeypatch, missing
):
    import shutil

    from lingcore.evaluation.regrade import regrade

    suite_path, _ = harbor_plan
    task = suite_path.parent / "dataset/tiny"
    shutil.copytree(task, task.with_name("z-healthy"))
    suite = yaml.safe_load(suite_path.read_text())
    suite["source"]["selection"]["count"] = 2
    suite_path.write_text(yaml.safe_dump(suite))
    monkeypatch.setenv("FIXTURE_SCENARIO", "verifier_failed")
    _, path, bridge = await frozen(harbor_plan)
    job, original = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    records = {
        json.loads(p.read_bytes())["case_id"]: p.parent
        for p in job.glob("executions/*/result.json")
    }
    assert len(records) == 2 and original["harbor_accuracy"] == 0
    if missing == "trial_record":
        (records["tiny"] / "harbor/job/trial/result.json").unlink()
    else:
        monkeypatch.setenv("FIXTURE_SCENARIO", "regrade_missing_first")
    before = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}
    report = await regrade(job, bridge=bridge)
    assert report["harbor_accuracy"] == 0.5 and report["mean_reward"] == 1
    skipped = [
        {
            "execution_id": records["tiny"].name,
            "case_id": "tiny",
            "reason": "missing_evidence",
        }
    ]
    assert report["regrade_skipped"] == skipped
    header = json.loads(next(job.glob("grade-sets/regrade-*.json")).read_bytes())
    assert header["complete"] and header["skipped"] == skipped
    copied = json.loads(
        next((records["tiny"] / "grades").glob("regrade-*.json")).read_bytes()
    )
    old = json.loads(before[records["tiny"] / "result.json"])
    assert {k: copied[k] for k in ("grade", "rewards", "reward_valid")} == {
        k: old[k] for k in ("grade", "rewards", "reward_valid")
    }
    assert copied["regrade_skip_reason"] == "missing_evidence"
    healthy = json.loads(
        next((records["z-healthy"] / "grades").glob("regrade-*.json")).read_bytes()
    )
    assert healthy["grade"]["status"] == "valid" and healthy["grade"]["passed"]
    assert all(p.read_bytes() == value for p, value in before.items())


async def test_regrade_allows_agent_timeout_multiplier(harbor_plan, monkeypatch):
    from lingcore.agent import Agent
    from lingcore.evaluation.regrade import regrade

    suite_path, _ = harbor_plan
    suite = yaml.safe_load(suite_path.read_text())
    suite["execution"]["harbor"]["timeout_multipliers"] = {"agent": 2}
    suite["execution"]["limits"]["wall_seconds"] = 200  # Above the task's 180s.
    suite_path.write_text(yaml.safe_dump(suite))
    _, path, bridge = await frozen(harbor_plan)
    job, _ = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client("fail")
    )
    monkeypatch.setattr(
        Agent,
        "from_profile",
        lambda *a, **k: pytest.fail("Agent created during regrade"),
    )
    report = await regrade(job, bridge=bridge)
    assert report["complete"] and report["harbor_accuracy"] == 1
