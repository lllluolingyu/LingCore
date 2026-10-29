"""Regressions from the milestone-one review, using no provider services."""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml

from lingcore.agent import Agent
from lingcore.evaluation.models import ExecutionConfig
from lingcore.evaluation.report import exit_code
from lingcore.evaluation.runner import run_suite
from lingcore.execution import ExecutionBudget, ExecutionLimits, run_headless
from lingcore.llm import LLMChunk
from lingcore.message import ToolCall
from lingcore.tracing.config import known_secrets
from lingcore.tracing.content import CommittedContentBuffer, sanitized_content
from lingcore.tracing.inspect import inspect_trace
from lingcore.usage import TokenUsage
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.test_evaluation import copied_suite, example_env, greeting_client
from tests.test_trace_runtime import profile, trace_directory


def test_only_credential_environment_values_are_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGCORE_MODEL", "deepseek-chat")
    monkeypatch.setenv("LINGCORE_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("TERM_KEYMAP", "ordinary-keymap")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "ordinary-setting")
    monkeypatch.setenv("SERVICE_API_KEY", "credential-value")
    monkeypatch.setenv("CUSTOM_CONNECTION", "declared-credential")
    monkeypatch.setenv("SERVICE_TOKEN", "1")
    prof = profile(
        tmp_path, llm={"model": "deepseek-chat", "api_key_env": "CUSTOM_CONNECTION"}
    )
    secrets = known_secrets(prof)
    assert {"credential-value", "declared-credential", "1"} <= set(secrets)
    assert not {
        "deepseek-chat",
        "https://example.test/v1",
        "ordinary-keymap",
        "ordinary-setting",
    } & set(secrets)
    data, _ = sanitized_content(
        {
            "model": "model-1.1",
            "seq": 1231,
            "message": "one 1 two",
            "api_key": "1",
            "password": "credential-value",
        },
        secrets,
        4096,
    )
    assert json.loads(data) == {
        "model": "model-1.1",
        "seq": 1231,
        "message": "one 1 two",
        "api_key": "[REDACTED]",
        "password": "[REDACTED]",
    }


async def test_model_attributes_and_candidate_fingerprints_survive_redaction(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    plans = []
    for model in ("deepseek-chat", "gpt-5.5-mini"):
        monkeypatch.setenv("LINGCORE_MODEL", model)
        monkeypatch.setenv("SERVICE_TOKEN", "1")
        source = tmp_path / model
        source.mkdir()
        suite = copied_suite(source)
        job, report = await run_suite(
            suite, tmp_path / "out", client_factory=greeting_client
        )
        assert exit_code(report) == 0
        plans.append(
            json.loads((job / "plan.json").read_text())["candidate_fingerprint"]
        )
        trace = inspect_trace(next(job.glob("executions/*/trace")))
        models = {
            r.attributes["model"] for r in trace.records if r.name == "llm.request"
        }
        assert models == {model}
    assert plans[0] != plans[1]


def test_content_cap_preserves_individual_items_and_counts_wrappers():
    buffer = CommittedContentBuffer(4096, ())
    for index in range(40):
        buffer.append(
            "tool.arguments",
            {"span_id": f"span-{index}", "arguments": {"index": index}},
        )
    captured = json.loads(buffer.data)
    assert isinstance(captured, list) and len(captured) > 1
    assert captured[0]["value"]["arguments"] == {"index": 0}
    assert buffer.byte_count == len(buffer.data) <= 4096
    assert buffer.truncated and buffer.omitted_items > 0


async def test_repeated_prompt_is_captured_once_and_tool_evidence_still_grades(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = copied_suite(
        tmp_path,
        suite_changes={"trace": {"content": "redacted", "artifact_max_bytes": 4096}},
    )

    def client(spec):
        raw = greeting_client(spec)
        # Both predicates are proved before later content exhausts the buffer.
        raw._turns[-1:] = [
            ScriptedTurn(
                text="large committed reply " * 1000, usage=TokenUsage("m", 1, 1)
            )
        ]
        return raw

    job, report = await run_suite(suite, tmp_path / "out", client_factory=client)
    assert exit_code(report) == 0
    for trace_dir in job.glob("executions/*/trace"):
        capture = next(trace_dir.glob("artifacts/*.json")).read_bytes()
        items = json.loads(capture)
        assert isinstance(items, list) and len(capture) <= 4096
        assert sum(item["kind"] == "prompt" for item in items) == 1
        assert (
            len(
                [
                    item
                    for item in items
                    if item["kind"] == "tool.arguments" and not item["truncated"]
                ]
            )
            == 2
        )
        trace = inspect_trace(trace_dir)
        assert any(
            r.name == "content.capture" and r.attributes["truncated"]
            for r in trace.records
        )


@pytest.mark.parametrize("limits", [{}, {"reported_tokens": 100}, {"wall_seconds": 2}])
def test_partial_native_limits_keep_default_bounds(limits):
    config = ExecutionConfig.model_validate({"limits": limits})
    assert config.limits.wall_seconds == limits.get("wall_seconds", 60)
    assert config.limits.model_requests == 20
    assert config.limits.tool_calls == 40


def test_explicit_unlimited_native_limit_is_preserved():
    config = ExecutionConfig.model_validate({"limits": {"model_requests": None}})
    assert config.limits.model_requests is None
    assert config.limits.wall_seconds == 60 and config.limits.tool_calls == 40


class HangingClient:
    def __init__(self):
        self.entered = asyncio.Event()
        self.closed = False

    async def stream(self, messages, tools=None):
        try:
            self.entered.set()
            yield LLMChunk(text_delta="partial")
            await asyncio.Event().wait()
        finally:
            self.closed = True


async def test_headless_external_cancellation_reaches_timeout_caller(tmp_path):
    raw = HangingClient()
    agent = Agent.from_profile(
        profile(tmp_path), llm=raw, execution_budget=ExecutionBudget()
    )
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.03):
            await run_headless(agent, ["go"])
    assert raw.closed and agent._closed and not agent.turn_pending_finalization
    assert agent.execution_outcome.status == "cancelled"
    assert inspect_trace(trace_directory(agent)).complete


async def test_suite_taskgroup_cancellation_preserves_results_and_propagates(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    raw = HangingClient()

    class GroupFailure(Exception):
        pass

    async def fail():
        await raw.entered.wait()
        raise GroupFailure

    task = None
    with pytest.raises(ExceptionGroup):
        async with asyncio.TaskGroup() as group:
            task = group.create_task(
                run_suite(
                    copied_suite(tmp_path),
                    tmp_path / "out",
                    client_factory=lambda spec: raw,
                )
            )
            group.create_task(fail())
    assert task.cancelled() and raw.closed
    job = next((tmp_path / "out").iterdir())
    report = json.loads((job / "report.json").read_text())
    assert exit_code(report) == 130 and report["coverage"]["started"] == 1
    result = json.loads(next(job.glob("executions/*/result.json")).read_text())
    assert result["execution"]["status"] == "cancelled"


async def test_suite_timeout_during_grading_keeps_completed_execution(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    import lingcore.evaluation.runner as runner

    grading = asyncio.Event()

    async def wait(*args):
        grading.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "grade", wait)
    task = asyncio.create_task(
        run_suite(
            copied_suite(tmp_path), tmp_path / "out", client_factory=greeting_client
        )
    )
    await grading.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    job = next((tmp_path / "out").iterdir())
    result = json.loads(next(job.glob("executions/*/result.json")).read_text())
    assert result["execution"]["status"] == "completed"
    assert result["grade"]["status"] != "valid"


async def test_grader_failure_does_not_rewrite_completed_execution(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    import lingcore.evaluation.runner as runner

    async def broken(*args):
        raise RuntimeError("exception contains credential-value")

    monkeypatch.setattr(runner, "grade", broken)
    job, report = await run_suite(
        copied_suite(tmp_path), tmp_path / "out", client_factory=greeting_client
    )
    assert exit_code(report) == 2
    results = report["cases"][0]["results"]
    assert all(
        r["execution"]["status"] == "completed" and r["grade"]["status"] == "invalid"
        for r in results
    )
    assert all(
        any(
            d["phase"] == "grade" and d["error_type"] == "RuntimeError"
            for d in r["diagnostics"]
        )
        for r in results
    )
    assert "exception contains credential-value" not in "".join(
        p.read_text() for p in job.rglob("*.json")
    )


async def test_one_grader_failure_preserves_independent_grades(tmp_path, monkeypatch):
    example_env(monkeypatch)
    from lingcore.evaluation.graders import ToolPolicyGrader

    async def broken(self, evidence):
        raise RuntimeError("private grader detail")

    monkeypatch.setattr(ToolPolicyGrader, "grade", broken)
    _, report = await run_suite(
        copied_suite(tmp_path), tmp_path / "out", client_factory=greeting_client
    )
    assert exit_code(report) == 2
    for result in report["cases"][0]["results"]:
        assert result["execution"]["status"] == "completed"
        assert result["grade"]["status"] == "invalid"
        assert result["grade"]["scores"] == {"artifact": 1.0, "tool_policy": None}
        assert result["grade"]["graders"][1]["error_type"] == "RuntimeError"


async def test_credential_rotation_does_not_change_candidate_fingerprint(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    fingerprints = []
    for credential in ("first-service-credential", "different-service-credential"):
        monkeypatch.setenv("LINGCORE_API_KEY", credential)
        job, report = await run_suite(
            suite, tmp_path / "out", client_factory=greeting_client
        )
        assert exit_code(report) == 0
        fingerprints.append(
            json.loads((job / "plan.json").read_text())["candidate_fingerprint"]
        )
    assert fingerprints[0] == fingerprints[1]


async def test_locked_suite_reports_actionable_safe_reason(
    tmp_path, monkeypatch, capsys
):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    data = yaml.safe_load(suite.read_text())
    data["source"]["lock"]["required"] = True
    suite.write_text(yaml.safe_dump(data))
    job, report = await run_suite(
        suite, tmp_path / "out", client_factory=greeting_client
    )
    assert "locked plan" in report["configuration_error"]
    assert "eval run --lock" in report["configuration_error"]
    from lingcore.__main__ import main

    assert main(["eval", "report", str(job)]) == 2
    assert "locked plan" in capsys.readouterr().out


async def test_wall_time_admission_and_driver_agree(tmp_path):
    class LateAdmission(FakeLLMClient):
        async def stream(self, messages, tools=None):
            # No await lets admission observe expiry before the driver timer.
            agent.execution_budget.deadline = 0
            async for chunk in super().stream(messages, tools):
                yield chunk

    raw = LateAdmission(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="r", name="read_file", arguments={"path": "hello.txt"})
                ]
            )
        ]
    )
    agent = Agent.from_profile(
        profile(tmp_path),
        llm=raw,
        execution_budget=ExecutionBudget(ExecutionLimits(wall_seconds=10)),
    )
    result = await run_headless(agent, ["go"])
    assert (
        result.outcome.status == "timed_out"
        and result.outcome.limit_type == "wall_seconds"
    )
