from __future__ import annotations

import asyncio
import json

import pytest
import yaml

from lingcore.__main__ import main
from lingcore.evaluation import runner
from lingcore.evaluation.storage import job_lock
from tests.test_evaluation import copied_suite, example_env, greeting_client


def fake_runs(monkeypatch):
    locked, unlocked, continuation = runner.run_locked, runner.run_suite, runner.resume

    async def run_locked(path, output, **kwargs):
        return await locked(path, output, client_factory=greeting_client, **kwargs)

    async def run_suite(path, output, **kwargs):
        return await unlocked(path, output, client_factory=greeting_client, **kwargs)

    async def resume(path, **kwargs):
        return await continuation(path, client_factory=greeting_client, **kwargs)

    monkeypatch.setattr(runner, "run_locked", run_locked)
    monkeypatch.setattr(runner, "run_suite", run_suite)
    monkeypatch.setattr(runner, "resume", resume)


def test_cli_locked_lifecycle_and_exit_codes(tmp_path, monkeypatch, capsys):
    example_env(monkeypatch)
    fake_runs(monkeypatch)
    suite = copied_suite(tmp_path)
    lock = tmp_path / "local.lock.json"
    assert main(["eval", "resolve", str(suite), "--lock", str(lock)]) == 0
    assert main(["eval", "validate", "--lock", str(lock)]) == 0
    assert main(["eval", "resolve", str(suite), "--lock", str(lock)]) == 2
    assert main(["eval", "resolve", str(suite), "--lock", str(lock), "--replace"]) == 0
    output = tmp_path / "out"
    assert main(["eval", "run", "--lock", str(lock), "--output", str(output)]) == 0
    job = next(output.iterdir())
    assert main(["eval", "resume", str(job)]) == 0
    assert main(["eval", "report", str(job)]) == 0
    assert main(["eval", "regrade", str(job)]) == 0
    assert main(["eval", "report", str(job), "--grades", "latest"]) == 0
    with job_lock(job):
        assert main(["eval", "resume", str(job)]) == 2
        assert main(["eval", "regrade", str(job)]) == 2
    case = tmp_path / "case.yaml"
    value = yaml.safe_load(case.read_text())
    value["graders"][0]["equals"] = "wrong"
    case.write_text(yaml.safe_dump(value))
    assert main(["eval", "validate", "--lock", str(lock)]) == 2
    assert (
        main(
            ["eval", "run", "--lock", str(lock), "--output", str(tmp_path / "drifted")]
        )
        == 2
    )
    assert not (tmp_path / "drifted").exists()
    assert main(["eval", "regrade", str(job)]) == 1
    assert main(["eval", "report", str(job), "--grades", "latest"]) == 1
    assert main(["eval", "report", str(job), "--grades", "original"]) == 0
    assert "fixture-secret-only" not in capsys.readouterr().out


@pytest.mark.parametrize("command", ["resolve", "validate", "run", "resume", "regrade"])
def test_cli_interruptions_are_130_and_print_known_job(
    tmp_path, monkeypatch, capsys, command
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    lock = tmp_path / "lock.json"
    assert main(["eval", "resolve", str(suite), "--lock", str(lock)]) == 0
    capsys.readouterr()

    def interrupt(coro):
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio, "run", interrupt)
    args = {
        "resolve": [str(suite), "--lock", str(tmp_path / "new.json")],
        "validate": ["--lock", str(lock)],
        "run": [str(suite), "--output", str(tmp_path / "out")],
        "resume": [str(tmp_path / "job")],
        "regrade": [str(tmp_path / "job")],
    }
    assert main(["eval", command, *args[command]]) == 130
    if command in {"resume", "regrade"}:
        assert str(tmp_path / "job") in capsys.readouterr().out
    assert not (tmp_path / "new.json").exists()


def test_v1_report_stays_readable_but_cannot_resume_or_regrade(tmp_path):
    from lingcore.evaluation.report import build_report

    (tmp_path / "job.json").write_text(json.dumps({"mode": "unlocked_native"}))
    (tmp_path / "plan.json").write_text(json.dumps({"planned": 0}))
    (tmp_path / "report.json").write_text(json.dumps(build_report([], 0)))
    assert main(["eval", "report", str(tmp_path)]) == 0
    assert main(["eval", "report", str(tmp_path), "--grades", "latest"]) == 0
    assert main(["eval", "resume", str(tmp_path)]) == 2
    assert main(["eval", "regrade", str(tmp_path)]) == 2


def test_unreadable_lock_and_report_are_2(tmp_path):
    assert main(["eval", "validate", "--lock", str(tmp_path / "missing")]) == 2
    assert main(["eval", "report", str(tmp_path / "missing")]) == 2


@pytest.mark.parametrize("command", ["resume", "regrade", "report"])
def test_corrupt_record_shapes_are_refused_without_a_traceback(tmp_path, command):
    from lingcore.evaluation.report import build_report

    (tmp_path / "job.json").write_text("[]")
    (tmp_path / "plan.json").write_text("[]")
    report = build_report([], 0)
    report["cases"] = None
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert main(["eval", command, str(tmp_path)]) == 2


def test_locked_cli_commands_print_safe_preflight_diagnostics(
    tmp_path, monkeypatch, capsys
):
    example_env(monkeypatch)
    fake_runs(monkeypatch)
    suite = copied_suite(tmp_path)
    lock = tmp_path / "lock.json"
    output = tmp_path / "out"
    assert main(["eval", "resolve", str(suite), "--lock", str(lock)]) == 0
    assert main(["eval", "run", "--lock", str(lock), "--output", str(output)]) == 0
    job = next(output.iterdir())
    case = tmp_path / "case.yaml"
    value = yaml.safe_load(case.read_text())
    value["requires"]["tools"].append("unavailable_tool")
    case.write_text(yaml.safe_dump(value))
    for args in (
        ["validate", "--lock", str(lock)],
        ["run", "--lock", str(lock)],
        ["resume", str(job)],
    ):
        capsys.readouterr()
        assert main(["eval", *args]) == 2
        captured = capsys.readouterr()
        assert "required_tools" in captured.out + captured.err
        assert "unavailable_tool" in captured.out + captured.err
        assert "fixture-secret-only" not in captured.out + captured.err
