from __future__ import annotations

import json

import pytest
import yaml

from lingcore.errors import ConfigError, LLMStreamError
from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.jobs import scan
from lingcore.evaluation.lock import load_lock, validate
from lingcore.evaluation.regrade import regrade
from lingcore.evaluation.report import exit_code
from lingcore.evaluation.runner import resume, run_locked
from tests.test_eval_jobs import locked_example
from tests.test_evaluation import greeting_client


class AuthenticationFailure:
    async def stream(self, messages, tools=None):
        raise LLMStreamError("fixture auth failure", retryable=False, category="auth")
        yield


@pytest.mark.parametrize("reason", ["auth", "configuration"])
async def test_resume_restarts_a_stopped_trial_once_and_keeps_history(
    tmp_path, monkeypatch, reason
):
    _, path, _ = await locked_example(
        tmp_path, monkeypatch, execution={"repetitions": 3}
    )
    calls = []

    def broken(spec):
        calls.append(spec.trial_id)
        if len(calls) == 1:
            return greeting_client(spec)
        if reason == "configuration":
            raise ConfigError("fixture configuration failure")
        return AuthenticationFailure()

    job, report = await run_locked(path, tmp_path / "out", client_factory=broken)
    assert exit_code(report) == 2 and len(calls) == 2
    before = {p: p.read_bytes() for p in (job / "executions").rglob("*") if p.is_file()}
    calls.clear()

    def still_broken(spec):
        calls.append(spec.trial_id)
        if reason == "configuration":
            raise ConfigError("fixture configuration failure")
        return AuthenticationFailure()

    stopped = await resume(job, client_factory=still_broken)
    assert len(calls) == 1 and exit_code(stopped) == 2
    assert stopped["stopped_reason"] == reason
    calls.clear()
    monkeypatch.setenv("LINGCORE_API_KEY", "replacement-fixture-key")

    def fixed(spec):
        calls.append(spec.trial_id)
        return greeting_client(spec)

    recovered = await resume(job, client_factory=fixed)
    assert calls == ["write-greeting--candidate--2", "write-greeting--candidate--3"]
    assert exit_code(recovered) == 0 and "stopped_reason" not in recovered
    assert recovered["coverage"]["executions"] == 5
    assert recovered["coverage"]["retried"] == 0
    assert recovered["usage"]["selected"]["requests"] == 9
    assert recovered["usage"]["all"]["requests"] == (11 if reason == "auth" else 9)
    attempts = recovered["trials"][1]["attempts"]
    assert [a["status"] for a in attempts] == ["failed", "failed", "completed"]
    assert [a["selected"] for a in attempts] == [False, False, True]
    assert all(p.read_bytes() == value for p, value in before.items())
    calls.clear()
    assert exit_code(await resume(job, client_factory=fixed)) == 0 and calls == []


@pytest.mark.parametrize("payload", [b"", b'{"truncated":', b"[]", b"{}"])
async def test_damaged_result_is_replaced_without_rewriting_it(
    tmp_path, monkeypatch, payload
):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    damaged = scan(job).executions[0]
    file = damaged.directory / "result.json"
    file.write_bytes(payload)
    calls = []

    def client(spec):
        calls.append(spec.trial_id)
        return greeting_client(spec)

    report = await resume(job, client_factory=client)
    assert exit_code(report) == 0 and calls == [damaged.trial.trial_id]
    assert file.read_bytes() == payload
    assert (damaged.directory / "interrupted.json").exists()
    assert report["coverage"]["replaced"] == 1
    assert report["usage"]["all"]["requests"] == 9
    assert any(
        i["execution_id"] == damaged.trial.execution_id
        and i["code"] == "unreadable_result"
        for i in report["ledger_issues"]
    )
    assert exit_code(await resume(job, client_factory=client)) == 0 and len(calls) == 1


@pytest.mark.parametrize("payload", [b"", b"{", None])
async def test_unreadable_trial_is_set_aside_and_reported(
    tmp_path, monkeypatch, payload
):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    damaged = scan(job).executions[0]
    file = damaged.directory / "trial.json"
    if payload is None:
        file.unlink()
    else:
        file.write_bytes(payload)
    before = {p: p.read_bytes() for p in damaged.directory.rglob("*") if p.is_file()}
    partial = await regrade(job)
    assert exit_code(partial) == 2
    report = await resume(job, client_factory=greeting_client)
    assert exit_code(report) == 0
    assert report["coverage"]["unassigned_executions"] == 1
    assert report["usage"]["all"]["coverage"] == "incomplete"
    assert any(
        i["execution_id"] == damaged.trial.execution_id
        and i["code"] == "unreadable_trial"
        for i in report["ledger_issues"]
    )
    assert all(p.read_bytes() == value for p, value in before.items())


async def test_stray_regular_files_do_not_block_resume_or_regrade(
    tmp_path, monkeypatch
):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    (job / "executions/.DS_Store").write_bytes(b"Finder metadata")
    assert exit_code(await resume(job, client_factory=greeting_client)) == 0
    assert exit_code(await regrade(job)) == 0


async def test_execution_symlinks_are_still_refused(tmp_path, monkeypatch):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    (job / "executions/link").symlink_to(tmp_path)
    with pytest.raises(EvaluationConfigError, match="unsafe"):
        await resume(job, client_factory=greeting_client)
    with pytest.raises(EvaluationConfigError, match="unsafe"):
        await regrade(job)


async def test_lock_preflight_drift_keeps_safe_diagnostics(tmp_path, monkeypatch):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    file = tmp_path / "case.yaml"
    value = yaml.safe_load(file.read_text())
    value["requires"]["tools"].append("unavailable_tool")
    file.write_text(yaml.safe_dump(value))
    drift = await validate(load_lock(path), path)
    assert drift[0].kind == "unreadable"
    assert drift[0].diagnostic.code == "required_tools"
    assert "unavailable_tool" in drift[0].diagnostic.message
    with pytest.raises(EvaluationConfigError, match="required_tools.*unavailable_tool"):
        await run_locked(path, tmp_path / "drifted", client_factory=greeting_client)
    assert not (tmp_path / "drifted").exists()
    with pytest.raises(EvaluationConfigError, match="required_tools.*unavailable_tool"):
        await resume(job, client_factory=greeting_client)


async def test_lock_parser_diagnostics_do_not_include_raw_values(tmp_path, monkeypatch):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    (tmp_path / "case.yaml").write_text("[broken: fixture-secret-only")
    drift = await validate(load_lock(path), path)
    assert drift[0].diagnostic.code == "configuration_read"
    assert "fixture-secret-only" not in json.dumps(drift[0].model_dump())


@pytest.mark.parametrize("marker", ["interrupted", "scratch-cleanup"])
async def test_damaged_recovery_marker_is_preserved_and_replaced(
    tmp_path, monkeypatch, marker
):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    damaged = scan(job).executions[0]
    (damaged.directory / "result.json").unlink()
    file = damaged.directory / (marker + ".json")
    file.write_bytes(b"")
    assert exit_code(await resume(job, client_factory=greeting_client)) == 0
    assert file.read_bytes() == b""
    assert list(damaged.directory.glob(marker + "-*.json"))
    assert exit_code(await resume(job, client_factory=greeting_client)) == 0


async def test_unreadable_result_stays_interrupted_after_it_becomes_readable(
    tmp_path, monkeypatch
):
    from lingcore.evaluation import jobs

    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    damaged = scan(job).executions[0]
    original = jobs.read_json

    def unreadable(file):
        if file == damaged.directory / "result.json":
            raise PermissionError("fixture read failure")
        return original(file)

    with monkeypatch.context() as patch:
        patch.setattr(jobs, "read_json", unreadable)
        assert exit_code(await resume(job, client_factory=greeting_client)) == 0
    attempts = scan(job).for_trial(damaged.trial.trial_id)
    assert attempts[0].state == "interrupted" and attempts[1].result is not None
    report = await resume(job, client_factory=greeting_client)
    selected = next(
        t for t in report["trials"] if t["trial_id"] == damaged.trial.trial_id
    )
    assert selected["selected_execution"] == attempts[1].trial.execution_id


async def test_unreadable_trial_stays_set_aside_after_it_becomes_readable(
    tmp_path, monkeypatch
):
    from lingcore.evaluation import jobs

    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    damaged = scan(job).executions[0]
    original = jobs.read_json

    def unreadable(file):
        if file == damaged.directory / "trial.json":
            raise PermissionError("fixture read failure")
        return original(file)

    with monkeypatch.context() as patch:
        patch.setattr(jobs, "read_json", unreadable)
        assert exit_code(await resume(job, client_factory=greeting_client)) == 0
    report = await resume(job, client_factory=greeting_client)
    assert exit_code(report) == 0
    assert report["coverage"]["unassigned_executions"] == 1
    assert all(
        e.trial.execution_id != damaged.trial.execution_id for e in scan(job).executions
    )
