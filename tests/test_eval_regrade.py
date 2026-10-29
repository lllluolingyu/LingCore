from __future__ import annotations

import json

import pytest
import yaml

from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.regrade import read_report, regrade
from lingcore.evaluation.report import exit_code
from lingcore.evaluation.runner import run_locked, run_suite
from tests.test_eval_jobs import locked_example
from tests.test_evaluation import copied_suite, example_env, greeting_client


async def test_regrade_corrects_grader_without_agent_or_record_changes(
    tmp_path, monkeypatch
):
    suite, path, _ = await locked_example(tmp_path, monkeypatch)
    case_path = tmp_path / "case.yaml"
    case = yaml.safe_load(case_path.read_text())
    case["graders"][0]["equals"] = "wrong expected answer"
    case_path.write_text(yaml.safe_dump(case))
    # Resolve again before running the intentionally failing original grading.
    from lingcore.evaluation.lock import resolve, write_lock

    write_lock(await resolve(suite, path), path, replace=True)
    job, original = await run_locked(
        path, tmp_path / "out", client_factory=greeting_client
    )
    assert exit_code(original) == 1
    before = {p: p.read_bytes() for p in (job / "executions").rglob("*") if p.is_file()}
    original_report = (job / "report.json").read_bytes()
    case["graders"][0]["equals"] = "Hello, LingCore!\n"
    case_path.write_text(yaml.safe_dump(case))
    from lingcore.agent import Agent

    def forbidden(*args, **kwargs):
        raise AssertionError("regrade cannot construct an Agent")

    monkeypatch.setattr(Agent, "from_profile", forbidden)
    corrected = await regrade(job)
    assert exit_code(corrected) == 0 and corrected["grade_set"] != "original"
    assert (job / "report.json").read_bytes() == original_report
    assert all(p.read_bytes() == data for p, data in before.items())
    assert read_report(job, "latest")["grade_set"] == corrected["grade_set"]
    assert exit_code(read_report(job)) == 1


@pytest.mark.parametrize("change", ["input", "new_artifact"])
async def test_regrade_refuses_input_changes_and_unpreserved_evidence(
    tmp_path, monkeypatch, change
):
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    case_path = tmp_path / "case.yaml"
    case = yaml.safe_load(case_path.read_text())
    if change == "input":
        case["input"]["turns"][0]["user"] = "changed agent input"
    else:
        case["graders"][0]["path"] = "never-collected.txt"
    case_path.write_text(yaml.safe_dump(case))
    with pytest.raises(EvaluationConfigError, match="write-greeting"):
        await regrade(job)
    assert not list((job / "grade-sets").glob("regrade-*"))


async def test_missing_preserved_file_remains_valid_failure_on_regrade(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    from lingcore.usage import TokenUsage
    from tests.fakes import FakeLLMClient, ScriptedTurn

    job, report = await run_suite(
        suite,
        tmp_path / "out",
        client_factory=lambda spec: FakeLLMClient(
            [ScriptedTurn(text="done", usage=TokenUsage("m", 1, 1))]
        ),
    )
    assert exit_code(report) == 1
    regraded = await regrade(job)
    assert exit_code(regraded) == 1 and regraded["coverage"]["valid"] == 2


async def test_extra_declared_evidence_can_support_a_new_grader(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = copied_suite(
        tmp_path, case_changes={"evidence": {"artifacts": ["absent.txt"]}}
    )
    job, _ = await run_suite(suite, tmp_path / "out", client_factory=greeting_client)
    file = tmp_path / "case.yaml"
    value = yaml.safe_load(file.read_text())
    value["graders"].append(
        {
            "id": "extra",
            "kind": "artifact_text",
            "path": "absent.txt",
            "equals": "anything",
        }
    )
    file.write_text(yaml.safe_dump(value))
    report = await regrade(job)
    assert exit_code(report) == 1
    assert report["cases"][0]["results"][0]["grade"]["scores"]["extra"] == 0


async def test_interrupted_regrade_preserves_partial_grade_set(tmp_path, monkeypatch):
    import asyncio
    import importlib

    module = importlib.import_module("lingcore.evaluation.regrade")
    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    entered = asyncio.Event()

    async def wait(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(module, "grade", wait)
    task = asyncio.create_task(regrade(job))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    header = json.loads(next((job / "grade-sets").glob("regrade-*.json")).read_text())
    assert header["complete"] is False
    assert read_report(job, "latest")["interrupted"] is True


async def test_regrade_normalizes_artifact_paths_and_preserves_incomplete_trace(
    tmp_path, monkeypatch
):
    from lingcore.evaluation.jobs import scan

    _, path, _ = await locked_example(tmp_path, monkeypatch)
    job, _ = await run_locked(path, tmp_path / "out", client_factory=greeting_client)
    file = tmp_path / "case.yaml"
    value = yaml.safe_load(file.read_text())
    value["graders"][0]["path"] = "./hello.txt"
    file.write_text(yaml.safe_dump(value))
    # Simulate an incomplete captured trace; the grader must not infer success.
    for record in scan(job).executions:
        manifest = record.directory / "evidence.json"
        saved = json.loads(manifest.read_text())
        saved["trace_complete"] = False
        manifest.write_text(json.dumps(saved))
    report = await regrade(job)
    assert exit_code(report) == 2
    grade = report["cases"][0]["results"][0]["grade"]
    assert grade["scores"]["artifact"] == 1
    assert grade["graders"][1]["reason"] == "metadata_incomplete"


async def test_regrade_refuses_argument_predicates_when_content_was_not_preserved(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path, suite_changes={"trace": {"content": "metadata"}})
    file = tmp_path / "case.yaml"
    value = yaml.safe_load(file.read_text())
    policy = value["graders"][1]
    arguments = [assertion.pop("arguments") for assertion in policy["require"]]
    file.write_text(yaml.safe_dump(value))
    job, report = await run_suite(
        suite, tmp_path / "out", client_factory=greeting_client
    )
    assert exit_code(report) == 0
    policy["require"][0]["arguments"] = arguments[0]
    file.write_text(yaml.safe_dump(value))
    with pytest.raises(EvaluationConfigError, match="grader tool_policy"):
        await regrade(job)


async def test_regrade_refuses_a_larger_limit_when_artifact_bytes_were_not_preserved(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    file = tmp_path / "case.yaml"
    value = yaml.safe_load(file.read_text())
    value["graders"][0]["max_bytes"] = 4
    file.write_text(yaml.safe_dump(value))
    job, report = await run_suite(
        suite, tmp_path / "out", client_factory=greeting_client
    )
    assert exit_code(report) == 1
    value["graders"][0]["max_bytes"] = 64
    file.write_text(yaml.safe_dump(value))
    with pytest.raises(EvaluationConfigError, match="grader artifact"):
        await regrade(job)
