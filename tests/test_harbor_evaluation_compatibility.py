"""Harbor evaluation tests: compatibility."""

from __future__ import annotations

import json

import pytest
import yaml

from lingcore.evaluation.lock import resolve, validate
from lingcore.evaluation.runner import resume, run_locked
from tests.support.harbor_evaluation import benchmark_suite, client, frozen
from tests.support.harbor_evaluation import harbor_plan as harbor_plan
from tests.support.harbor_evaluation import trusted_release as trusted_release


def test_bundled_terminal_bench_release():
    from lingcore.evaluation.harbor.provenance import release_manifest

    manifest = release_manifest("terminal-bench/terminal-bench@4.0.0")
    assert manifest is not None
    assert len(manifest["tasks"]) == 66
    assert (
        manifest["hub_version"]
        == "sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732"
    )
    assert (
        manifest["release_source"]["commit"]
        == "452bf305c6daa62fc59061d22133a7cbc7c1572e"
    )
    assert manifest["policy"]["repetitions"] == 5
    assert release_manifest("terminal-bench/terminal-bench@latest") is None


async def test_verified_release_and_subset_provenance(harbor_plan, trusted_release):
    lock, path, bridge = await frozen(harbor_plan)
    assert lock.source.dataset["provenance"]["status"] == "verified"
    assert "release_unverified" not in lock.limitations
    assert not await validate(lock, path, bridge=bridge)
    trusted_release["tasks"]["another"] = "sha256:" + "d" * 64
    subset = await resolve(harbor_plan[0], path, bridge=bridge)
    assert subset.source.dataset["provenance"]["status"] == "verified_subset"
    assert subset.source.dataset["provenance"]["published_tasks"] == 2
    assert await validate(lock, path, bridge=bridge)


@pytest.mark.parametrize(
    "change,code",
    [
        ("task", "release_task_mismatch"),
        ("release", "release_source_mismatch"),
        ("extra", "release_task_mismatch"),
        ("ignored", "release_task_mismatch"),
    ],
)
async def test_false_release_claims_refused(harbor_plan, trusted_release, change, code):
    import shutil

    from lingcore.evaluation.config import EvaluationConfigError

    path, _ = harbor_plan
    if change == "task":
        (path.parent / "dataset/tiny/instruction.md").write_text("Modified task")
    elif change == "extra":
        shutil.copytree(path.parent / "dataset/tiny", path.parent / "dataset/extra")
    elif change == "ignored":
        description_path = path.parent / "dataset/tiny/description.json"
        description = json.loads(description_path.read_bytes())
        description["package_complete"] = False
        description_path.write_text(json.dumps(description))
        from lingcore.evaluation.harbor.process import invoke

        described = invoke(
            harbor_plan[1], ["describe", "--dataset", str(path.parent / "dataset")], {}
        )
        trusted_release["tasks"]["tiny"] = described["tasks"][0]["harbor_digest"]
    else:
        suite = yaml.safe_load(path.read_text())
        suite["source"]["release_source"] = {"commit": "b" * 40}
        path.write_text(yaml.safe_dump(suite))
    with pytest.raises(EvaluationConfigError) as exc:
        await frozen(harbor_plan)
    assert exc.value.diagnostic.code == code


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "subset",
        "repetitions",
        "wall_seconds",
        "model_requests",
        "tool_calls",
        "reported_tokens",
        "usage_missing",
        "automatic_turns",
        "multiplier",
        "reward",
        "trace",
        "training",
        "images",
    ],
)
async def test_benchmark_refuses_modified_policy(harbor_plan, trusted_release, change):
    from lingcore.evaluation.config import EvaluationConfigError

    path, _ = harbor_plan
    suite = benchmark_suite(harbor_plan)
    if change == "unknown":
        suite["source"]["dataset"] = "untrusted"
    elif change == "subset":
        trusted_release["tasks"]["missing"] = "sha256:" + "d" * 64
    elif change == "repetitions":
        suite["execution"]["repetitions"] = 1
    elif change in suite["execution"]["limits"]:
        suite["execution"]["limits"][change] = (
            "stop" if change == "usage_missing" else 1
        )
    elif change == "multiplier":
        suite["execution"]["harbor"]["timeout_multipliers"] = {"agent": 2}
    elif change == "reward":
        suite["grading"]["reward"] = {"pass_at": 0}
    elif change == "trace":
        suite["trace"]["required"] = False
    elif change == "training":
        suite["source"]["training_use"] = "unspecified"
    elif change == "images":
        description_path = path.parent / "dataset/tiny/description.json"
        description = json.loads(description_path.read_bytes())
        description["environments"]["agent"]["pinned"] = False
        description_path.write_text(json.dumps(description))
        from lingcore.evaluation.harbor.process import invoke

        description = invoke(
            harbor_plan[1], ["describe", "--dataset", str(path.parent / "dataset")], {}
        )
        trusted_release["tasks"]["tiny"] = description["tasks"][0]["harbor_digest"]
    path.write_text(yaml.safe_dump(suite))
    with pytest.raises(EvaluationConfigError) as exc:
        await frozen(harbor_plan)
    assert exc.value.diagnostic.code == "benchmark_policy"


async def test_complete_benchmark_and_regrade_eligibility(harbor_plan, trusted_release):
    from lingcore.evaluation.regrade import regrade

    benchmark_suite(harbor_plan)
    lock, path, bridge = await frozen(harbor_plan)
    assert len(lock.trials) == 5 and not lock.limitations
    assert not await validate(lock, path, bridge=bridge)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    assert report["track"] == "benchmark"
    assert report["benchmark_compatible"] and report["benchmark_score"] == 1
    assert report["score_scope"] == "complete_benchmark"
    assert report["benchmark_eligibility"]["reasons"] == []
    assert report["coverage"]["valid"] == 5
    assert report["resources"]["tasks"]["tiny"] == lock.source.tasks[0].resources
    assert report["cases"][0]["resources"] == lock.source.tasks[0].resources
    assert all(
        r["comparison"]["track"] == "benchmark" for r in report["cases"][0]["results"]
    )
    graded = await regrade(job, bridge=bridge)
    assert graded["complete"] and not graded["benchmark_compatible"]
    assert graded["benchmark_score"] is None
    assert graded["benchmark_eligibility"]["reasons"] == ["regraded"]


async def test_benchmark_credentials_can_be_supplied_after_resolve(
    harbor_plan, trusted_release, monkeypatch
):
    benchmark_suite(harbor_plan)
    monkeypatch.delenv("FIXTURE_API_KEY")
    lock, path, bridge = await frozen(harbor_plan)
    original_lock = path.read_bytes()
    assert lock.limitations == ["credential_unset"]
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret-never-forwarded")
    assert not await validate(lock, path, bridge=bridge)

    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    assert report["complete"] and report["harbor_accuracy"] == 1
    assert report["benchmark_compatible"] and report["benchmark_score"] == 1
    assert report["benchmark_eligibility"]["reasons"] == []
    assert report["score_scope"] == "complete_benchmark"
    # Lock-time availability stays visible, without changing the frozen plan.
    assert report["limitations"] == ["credential_unset"]
    assert path.read_bytes() == (job / "lock.json").read_bytes() == original_lock
    assert json.loads((job / "report.json").read_bytes()) == report


async def test_runner_diagnostic_withholds_benchmark_until_resume(
    harbor_plan, trusted_release, monkeypatch
):
    from lingcore.evaluation import jobs

    benchmark_suite(harbor_plan)
    lock, path, bridge = await frozen(harbor_plan)
    original_scan = jobs.scan
    failed = False

    def intermittent_scan(job, plan=None):
        nonlocal failed
        state = original_scan(job, plan)
        if len(state.executions) == len(lock.trials) and not failed:
            failed = True
            raise OSError("injected transient ledger read failure")
        return state

    monkeypatch.setattr(jobs, "scan", intermittent_scan)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    assert failed and not report["complete"]
    assert report["diagnostics"][0]["code"] == "runner_failed"
    assert report["coverage"]["valid"] == len(lock.trials)
    assert not report["benchmark_compatible"] and report["benchmark_score"] is None
    assert report["benchmark_eligibility"]["reasons"] == ["incomplete"]
    assert report["score_scope"] == "benchmark_ineligible_descriptive"
    assert all(
        not result["benchmark_compatible"]
        and not result["comparison"]["benchmark_compatible"]
        for row in report["cases"]
        for result in row["results"]
    )
    assert json.loads((job / "report.json").read_bytes()) == report

    def forbidden(spec):
        pytest.fail("completed trials must not run again")

    recovered = await resume(job, bridge=bridge, client_factory=forbidden)
    assert recovered["complete"] and "diagnostics" not in recovered
    assert recovered["coverage"]["executions"] == len(lock.trials)
    assert recovered["benchmark_compatible"] and recovered["benchmark_score"] == 1
    assert recovered["benchmark_eligibility"]["reasons"] == []


async def test_invalid_benchmark_withholds_score(
    harbor_plan, trusted_release, monkeypatch
):
    benchmark_suite(harbor_plan)
    monkeypatch.setenv("FIXTURE_SCENARIO", "unsupported")
    _, path, bridge = await frozen(harbor_plan)
    _, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    assert not report["benchmark_compatible"] and report["benchmark_score"] is None
    assert "incomplete" in report["benchmark_eligibility"]["reasons"]
    assert report["resources"]["tasks"]["tiny"]


async def test_development_never_claims_benchmark(harbor_plan, trusted_release):
    _, path, bridge = await frozen(harbor_plan)
    _, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    assert report["complete"] and not report["benchmark_compatible"]
    assert report["benchmark_score"] is None
    assert "development_track" in report["benchmark_eligibility"]["reasons"]


async def test_benchmark_refuses_partial_selection_of_complete_export(
    harbor_plan, trusted_release
):
    import shutil

    from lingcore.evaluation.config import EvaluationConfigError
    from lingcore.evaluation.harbor.process import invoke

    path, bridge = harbor_plan
    shutil.copytree(path.parent / "dataset/tiny", path.parent / "dataset/another")
    description = invoke(
        bridge, ["describe", "--dataset", str(path.parent / "dataset")], {}
    )
    trusted_release["tasks"] = {
        t["id"]: t["harbor_digest"] for t in description["tasks"]
    }
    benchmark_suite(harbor_plan)  # Still selects only one of the two tasks.
    with pytest.raises(EvaluationConfigError, match="full_task_selection"):
        await frozen(harbor_plan)


async def test_unstarted_report_has_resources_and_no_benchmark_score(
    harbor_plan, trusted_release
):
    from lingcore.evaluation.jobs import JobState
    from lingcore.evaluation.report import build_job_report

    benchmark_suite(harbor_plan)
    lock, _, _ = await frozen(harbor_plan)
    report = build_job_report(
        lock,
        JobState(()),
        locked=True,
        lock_digest=lock.digest,
        plan_digest=lock.plan_digest,
    )
    assert report["resources"]["tasks"]["tiny"]["verifier"]["memory_mb"] == 512
    assert report["resources"]["enforcement"]["storage_mb"] == "not_enforced"
    assert report["cases"][0]["resources"]["agent"]["cpus"] == 1
    assert not report["cases"][0]["results"]
    assert report["benchmark_score"] is None and not report["benchmark_compatible"]
    assert "incomplete" in report["benchmark_eligibility"]["reasons"]


def test_native_cannot_opt_into_benchmark():
    from pydantic import ValidationError

    from lingcore.evaluation.models import Suite

    with pytest.raises(ValidationError, match="benchmark track requires"):
        Suite.model_validate(
            {
                "schema_version": "lingcore.eval.suite.v1",
                "name": "native",
                "track": "benchmark",
                "source": {"cases": ["case.yaml"]},
                "agent": {"profile": "agent.yaml"},
            }
        )
