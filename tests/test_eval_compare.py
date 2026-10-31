from __future__ import annotations

import json

import pytest
import yaml

from lingcore.agent import Agent
from lingcore.evaluation.cli import main
from lingcore.evaluation.compare import compare_jobs
from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.lock import resolve, write_lock
from lingcore.evaluation.regrade import regrade
from lingcore.evaluation.runner import run_locked, run_suite
from lingcore.evaluation.storage import job_lock
from tests.fakes import FakeLLMClient
from tests.test_evaluation import copied_suite, example_env, greeting_client
from tests.test_regression_suite import regression_client, script, suite_at


async def simple_job(suite, output):
    job, report = await run_suite(suite, output, client_factory=greeting_client)
    assert report["complete"]
    return job


async def test_compare_pairs_new_suite_and_bootstraps_cases(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = suite_at(tmp_path / "suite.yaml")
    lock_path = tmp_path / "lock.json"
    write_lock(await resolve(suite, lock_path), lock_path)
    baseline, _ = await run_locked(
        lock_path, tmp_path / "out", client_factory=regression_client
    )

    def candidate_client(spec):
        if spec.case_id != "calendar-dst":
            return regression_client(spec)
        turns = script(spec.case_id)
        turns[2].tool_calls[0].arguments["content"] = "wrong timezone\n"
        return FakeLLMClient(turns)

    candidate, _ = await run_locked(
        lock_path, tmp_path / "out", client_factory=candidate_client
    )

    def refuse(*args, **kwargs):
        raise AssertionError("comparison must not assemble an agent")

    monkeypatch.setattr(Agent, "from_profile", refuse)
    report = compare_jobs(baseline, candidate)
    assert report["complete"] and report["exit_code"] == 1
    assert report["regressions"] == ["calendar-dst"]
    assert report["mean_delta"] == pytest.approx(-0.2)
    assert report["coverage"] == {
        "planned_pairs": 15,
        "valid_pairs": 15,
        "unpaired": 0,
        "cases": 5,
    }
    assert report["uncertainty"]["unit"] == "case"
    low, high = report["uncertainty"]["interval"]
    assert low < -0.2 and high == 0
    assert compare_jobs(baseline, candidate)["uncertainty"] == report["uncertainty"]
    assert report["usage"]["all"]["delta"]["requests"] == 0
    assert report["paired_mean_latency_delta_ms"] is not None
    assert compare_jobs(candidate, baseline)["improvements"] == ["calendar-dst"]


async def test_self_comparison_and_offline_sources(tmp_path, monkeypatch, capsys):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    job = await simple_job(suite, tmp_path / "out")
    for path in (suite, tmp_path / "agent.yaml", tmp_path / "case.yaml"):
        path.unlink()
    result = compare_jobs(job, job)
    assert result["exit_code"] == 0 and result["mean_delta"] == 0
    # Repetitions of one case are not independent tasks for uncertainty.
    assert result["uncertainty"]["interval"] is None
    assert result["uncertainty"]["reason"] == "fewer_than_two_cases"
    target = tmp_path / "comparison.json"
    assert main(["compare", str(job), str(job), "--output", str(target)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(target.read_text())
    assert main(["compare", str(job), str(job), "--output", str(target)]) == 2
    with job_lock(job):
        assert main(["compare", str(job), str(job)]) == 2


@pytest.mark.parametrize(
    "change,section",
    [
        ("input", "source.inputs"),
        ("grader", "grading.cases"),
        ("repetitions", "trials"),
        ("limit", "execution"),
        ("trace", "trace"),
    ],
)
async def test_incompatible_comparisons_are_refused(
    tmp_path, monkeypatch, change, section
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    baseline = await simple_job(suite, tmp_path / "out")
    path = tmp_path / "case.yaml" if change in {"input", "grader"} else suite
    value = yaml.safe_load(path.read_text())
    if change == "input":
        value["input"]["turns"][0]["user"] += " Verify twice."
    elif change == "grader":
        value["graders"][0]["equals"] = "wrong\n"
    elif change == "repetitions":
        value["execution"]["repetitions"] = 1
    elif change == "limit":
        value["execution"]["limits"]["wall_seconds"] = 59
    else:
        value["trace"]["max_bytes"] *= 2
    path.write_text(yaml.safe_dump(value))
    candidate = await simple_job(suite, tmp_path / "out")
    with pytest.raises(EvaluationConfigError, match=section):
        compare_jobs(baseline, candidate)


async def test_candidate_model_change_is_allowed(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    baseline = await simple_job(suite, tmp_path / "out")
    monkeypatch.setenv("LINGCORE_MODEL", "different-candidate")
    candidate = await simple_job(suite, tmp_path / "out")
    result = compare_jobs(baseline, candidate)
    assert result["complete"]
    assert result["candidate"]["model"]["id"] == "different-candidate"
    assert (
        result["candidate"]["candidate_digest"]
        != result["baseline"]["candidate_digest"]
    )


async def test_missing_execution_does_not_reuse_stale_report_success(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    baseline = await simple_job(suite, tmp_path / "out")
    candidate = await simple_job(suite, tmp_path / "out")
    next(candidate.glob("executions/*/result.json")).unlink()
    original_report = (candidate / "report.json").read_bytes()
    result = compare_jobs(baseline, candidate)
    assert result["exit_code"] == 2 and not result["complete"]
    assert result["mean_delta"] is None and result["uncertainty"]["interval"] is None
    assert result["descriptive_paired_delta"] == 0
    assert result["coverage"]["valid_pairs"] == 1
    assert result["coverage"]["unpaired"] == 1
    assert result["usage"]["all"]["coverage"] == "incomplete"
    assert result["usage"]["all"]["delta"]["requests"] is None
    assert (candidate / "report.json").read_bytes() == original_report


async def test_selected_regrades_must_match_and_preserve_originals(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    baseline = await simple_job(suite, tmp_path / "out")
    candidate = await simple_job(suite, tmp_path / "out")
    original = {
        p: p.read_bytes() for p in tmp_path.glob("out/*/executions/*/result.json")
    }
    case = tmp_path / "case.yaml"
    value = yaml.safe_load(case.read_text())
    value["graders"][0]["equals"] = "corrected expectation\n"
    case.write_text(yaml.safe_dump(value))
    await regrade(baseline)
    await regrade(candidate)
    with pytest.raises(EvaluationConfigError, match="grading.cases"):
        compare_jobs(baseline, candidate, candidate_grades="latest")
    result = compare_jobs(
        baseline, candidate, baseline_grades="latest", candidate_grades="latest"
    )
    assert result["exit_code"] == 0
    assert (
        result["cases"][0]["baseline_mean"] == result["cases"][0]["candidate_mean"] == 0
    )
    assert result["baseline"]["grade_set"].startswith("regrade-")
    assert all(p.read_bytes() == data for p, data in original.items())
    grade_id = result["candidate"]["grade_set"]
    next(candidate.glob(f"executions/*/grades/{grade_id}.json")).unlink()
    result = compare_jobs(
        baseline, candidate, baseline_grades="latest", candidate_grades="latest"
    )
    assert result["exit_code"] == 2


def test_bootstrap_bounds_rejected_before_reading_jobs(tmp_path):
    with pytest.raises(EvaluationConfigError, match="bootstrap_samples"):
        compare_jobs(tmp_path, tmp_path, bootstrap_samples=1)
