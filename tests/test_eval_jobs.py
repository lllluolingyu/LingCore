from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.jobs import scan
from lingcore.evaluation.lock import resolve, write_lock
from lingcore.evaluation.report import exit_code
from lingcore.evaluation.runner import resume, run_locked, run_suite
from lingcore.evaluation.storage import job_lock
from tests.test_evaluation import copied_suite, example_env, greeting_client


async def locked_example(tmp_path, monkeypatch, *, execution=None):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    if execution:
        values = yaml.safe_load(suite.read_text())
        values["execution"].update(execution)
        suite.write_text(yaml.safe_dump(values))
    path = tmp_path / "local.lock.json"
    lock = await resolve(suite, path)
    write_lock(lock, path)
    return suite, path, lock


async def test_locked_jobs_share_identity_and_direct_traces(tmp_path, monkeypatch):
    _, path, lock = await locked_example(tmp_path, monkeypatch)
    for _ in range(2):
        job, report = await run_locked(
            path, tmp_path / "out", client_factory=greeting_client
        )
        assert exit_code(report) == 0 and report["plan_digest"] == lock.plan_digest
        assert (
            report["schema_version"] == "lingcore.eval-report.v2" and report["locked"]
        )
        assert (job / "lock.json").read_bytes() == path.read_bytes()
        for record in scan(job).executions:
            assert record.result.locked and record.result.lock_digest == lock.digest
            assert (record.directory / "trace/events.jsonl").exists()
            assert not Path(record.trial.scratch_dir).exists()
            assert (record.directory / "evidence.json").exists()
            assert all(
                p.suffix == ".bin" for p in (record.directory / "artifacts").iterdir()
            )


async def test_locked_drift_creates_no_job(tmp_path, monkeypatch):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    (tmp_path / "world.md").write_text("changed prompt")

    def forbidden(spec):
        raise AssertionError("factory must not be called on drift")

    with pytest.raises(EvaluationConfigError, match="candidate"):
        await run_locked(path, tmp_path / "out", client_factory=forbidden)
    assert not (tmp_path / "out").exists()


async def test_resume_replaces_only_orphaned_execution_and_preserves_records(
    tmp_path, monkeypatch
):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    orphan = scan(job).executions[0]
    (orphan.directory / "result.json").unlink()
    before = {p: p.read_bytes() for p in (job / "executions").rglob("*") if p.is_file()}
    called = []

    def client(spec):
        called.append(spec.trial_id)
        return greeting_client(spec)

    report = await resume(job, client_factory=client)
    assert called == [orphan.trial.trial_id] and exit_code(report) == 0
    assert (
        report["coverage"]["replaced"] == 1 and report["coverage"]["interrupted"] == 1
    )
    assert all(p.read_bytes() == data for p, data in before.items())
    assert (orphan.directory / "interrupted.json").exists()
    attempts = scan(job).for_trial(orphan.trial.trial_id)
    assert [e.trial.attempt for e in attempts] == [1, 2]
    assert (
        report["usage"]["all"]["requests"] == 9
        and report["usage"]["selected"]["requests"] == 6
    )
    called.clear()
    assert exit_code(await resume(job, client_factory=client)) == 0 and called == []


async def test_held_job_and_unlocked_resume_are_refused(tmp_path, monkeypatch):
    suite, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    with job_lock(job):
        with pytest.raises(EvaluationConfigError, match="locked"):
            await resume(job, client_factory=greeting_client)
    unlocked, report = await run_suite(
        suite, tmp_path / "out", client_factory=greeting_client
    )
    assert not report["locked"] and (unlocked / "plan.json").exists()
    with pytest.raises(EvaluationConfigError, match="unlocked"):
        await resume(unlocked, client_factory=greeting_client)


async def test_cancelled_trial_has_readable_trace_and_resumes(tmp_path, monkeypatch):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    from tests.test_evaluation_regressions import HangingClient

    raw = HangingClient()
    task = asyncio.create_task(
        run_locked(path, tmp_path / "out", client_factory=lambda spec: raw)
    )
    await raw.entered.wait()
    job = next((tmp_path / "out").iterdir())
    trace = next(job.glob("executions/*/trace/events.jsonl"))
    assert trace.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    report = await resume(job, client_factory=greeting_client)
    assert exit_code(report) == 0 and report["coverage"]["replaced"] == 1


async def test_killed_worker_leaves_partial_journal_for_resume(tmp_path, monkeypatch):
    import sys

    _, path, _ = await locked_example(tmp_path, monkeypatch)
    output = tmp_path / "out"
    code = """
import asyncio, sys
from pathlib import Path
from lingcore.evaluation.runner import run_locked
from tests.test_evaluation_regressions import HangingClient
asyncio.run(run_locked(Path(sys.argv[1]), Path(sys.argv[2]), client_factory=lambda spec: HangingClient()))
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, str(path), str(output)
    )
    try:
        async with asyncio.timeout(10):
            while True:
                files = list(output.glob("job-*/executions/*/trace/events.jsonl"))
                if files and b'"llm.request"' in files[0].read_bytes():
                    break
                await asyncio.sleep(0.03)
        process.kill()
        await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    job = next(output.iterdir())
    orphan = scan(job).executions[0]
    assert orphan.state == "orphaned" and Path(orphan.trial.scratch_dir).exists()
    report = await resume(job, client_factory=greeting_client)
    assert exit_code(report) == 0 and report["coverage"]["replaced"] == 1
    assert not Path(orphan.trial.scratch_dir).exists()
    marker = json.loads((orphan.directory / "interrupted.json").read_text())
    assert marker["model_requests"] == 1


async def test_resume_accepts_relocated_lock_but_rejects_a_different_lock(
    tmp_path, monkeypatch
):
    import shutil

    from lingcore.evaluation.lock import load_lock

    source = tmp_path / "source"
    source.mkdir()
    _, path, _ = await locked_example(source, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    moved = tmp_path / "moved"
    shutil.move(source, moved)
    location = moved / path.name
    assert (
        exit_code(await resume(job, lock_path=location, client_factory=greeting_client))
        == 0
    )
    # Same semantic plan, different full lock digest (new resolution timestamp).
    write_lock(await resolve(moved / "suite.yaml", location), location, replace=True)
    assert (
        load_lock(location).digest
        != json.loads((job / "lock.json").read_text())["digest"]
    )
    with pytest.raises(EvaluationConfigError, match="does not match"):
        await resume(job, lock_path=location, client_factory=greeting_client)


async def test_resume_refuses_executions_outside_the_frozen_plan(tmp_path, monkeypatch):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    record = scan(job).executions[0]
    for name in ("trial.json", "result.json"):
        file = record.directory / name
        value = json.loads(file.read_text())
        value["trial_id"] = "unknown-trial"
        file.write_text(json.dumps(value))
    with pytest.raises(EvaluationConfigError, match="frozen plan"):
        await resume(job, client_factory=greeting_client)


async def test_cancellation_joins_collection_before_removing_workspace(
    tmp_path, monkeypatch
):
    import threading

    from lingcore.evaluation import evidence

    _, path, _ = await locked_example(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = evidence.collect

    def blocked(workspace, *args):
        entered.set()
        assert release.wait(5)
        assert workspace.exists()
        return original(workspace, *args)

    monkeypatch.setattr(evidence, "collect", blocked)
    task = asyncio.create_task(
        run_locked(path, tmp_path / "out", client_factory=greeting_client)
    )
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    job = next((tmp_path / "out").iterdir())
    record = scan(job).executions[0]
    assert (record.directory / "evidence.json").exists()
    assert not Path(record.trial.scratch_dir).exists()
    assert record.result.grade.reason == "collect_cancelled"


async def test_failed_fixture_preparation_removes_partial_scratch(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = copied_suite(
        tmp_path, case_changes={"fixture": {"files": {"a": "file", "a/b": "conflict"}}}
    )
    job, report = await run_suite(
        suite, tmp_path / "out", client_factory=greeting_client
    )
    assert exit_code(report) == 2
    for record in scan(job).executions:
        assert record.result.execution.cause == "environment"
        assert not Path(record.trial.scratch_dir).exists()
        assert not record.result.cleanup_failed
