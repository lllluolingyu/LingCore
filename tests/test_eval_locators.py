"""User-supplied parent paths remain usable through resolve/run/regrade/report."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lingcore.evaluation.compare import compare_jobs
from lingcore.evaluation.lock import resolve, validate, write_lock
from lingcore.evaluation.regrade import read_report, regrade
from lingcore.evaluation.runner import run_locked, run_suite
from lingcore.tracing.inspect import inspect_trace
from tests.support.harbor_evaluation import client
from tests.support.harbor_evaluation import harbor_plan as harbor_plan
from tests.test_evaluation import copied_suite, example_env, greeting_client


@pytest.mark.parametrize("absolute_dataset", [False, True])
async def test_harbor_external_dataset_and_nested_lock(
    harbor_plan, monkeypatch, absolute_dataset
):
    original, bridge = harbor_plan
    root = original.parent
    suite = yaml.safe_load(original.read_text())
    suite["source"]["path"] = (
        str(root / "dataset") if absolute_dataset else "../dataset"
    )
    suite["agent"]["profile"] = "../agent.yaml"
    suites = root / "suites"
    suites.mkdir()
    (suites / "harbor.yaml").write_text(yaml.safe_dump(suite))
    monkeypatch.chdir(suites)
    path = Path("../locks/harbor.json")
    lock = await resolve(Path("../suites/harbor.yaml"), path, bridge=bridge)
    write_lock(lock, path)
    assert ".." in lock.source.dataset["path"]
    assert not await validate(lock, path, bridge=bridge)
    job, original_report = await run_locked(
        path, Path("../jobs"), bridge=bridge, client_factory=client("fail")
    )
    assert original_report["mean_reward"] == 0
    report = await regrade(job, bridge=bridge)
    assert report["mean_reward"] == 1
    relative_job = Path("../jobs") / job.name
    assert read_report(relative_job, "latest") == report
    assert compare_jobs(
        job.resolve(), relative_job, baseline_grades="latest", candidate_grades="latest"
    )["complete"]
    trace = next((relative_job / "executions").glob("*/trace"))
    assert inspect_trace(trace).complete


async def test_native_direct_run_accepts_parent_paths(tmp_path, monkeypatch):
    example_env(monkeypatch)
    copied_suite(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    job, report = await run_suite(
        Path("../suite.yaml"), Path("../jobs"), client_factory=greeting_client
    )
    assert report["complete"]
    assert read_report(Path("../jobs") / job.name) == report
