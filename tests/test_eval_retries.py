from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.errors import LLMStreamError
from lingcore.evaluation.identity import RetryPolicy
from lingcore.evaluation.jobs import (
    ExecutionRecord,
    TrialRecord,
    infrastructure_cause,
    select,
)
from lingcore.evaluation.models import Grade, TrialResult, UsageSummary
from lingcore.evaluation.report import exit_code
from lingcore.evaluation.runner import run_locked
from lingcore.execution import ExecutionOutcome
from lingcore.tracing import TraceHealth
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.test_eval_jobs import locked_example
from tests.test_evaluation import greeting_client


def record(
    attempt,
    *,
    status="failed",
    cause="runner",
    category=None,
    cleanup=False,
    passed=True,
):
    result = TrialResult(
        trial_id="one",
        execution_id=str(attempt),
        attempt=attempt,
        case_id="one",
        case_revision=1,
        repetition=1,
        execution=ExecutionOutcome(
            status=status, cause=cause, provider_error_category=category
        ),
        grade=Grade(status="valid", passed=passed),
        usage=UsageSummary(),
        trace=TraceHealth(),
        content="metadata",
        cleanup_failed=cleanup,
    )
    trial = TrialRecord(
        trial_id="one",
        case_id="one",
        repetition=1,
        execution_id=str(attempt),
        attempt=attempt,
        started_at="now",
        plan_digest="sha256:" + "0" * 64,
        scratch_dir="/tmp/unused",
    )
    return ExecutionRecord(trial, Path("/tmp/unused"), "finished", result)


@pytest.mark.parametrize(
    "kwargs,cause",
    [
        ({}, "runner"),
        ({"cause": "environment"}, "environment"),
        ({"cause": "provider", "category": "transient"}, "provider_transient"),
        ({"cause": "provider", "category": "interrupted"}, "provider_transient"),
        ({"cause": "provider", "category": "request"}, None),
        ({"cause": "provider", "category": "auth"}, None),
        ({"cause": "provider", "category": "configuration"}, None),
        ({"cause": "configuration"}, None),
        ({"cause": "agent"}, None),
        ({"status": "budget_exceeded", "cause": "agent"}, None),
        ({"status": "timed_out", "cause": "agent"}, None),
        ({"status": "completed", "cause": None, "passed": False}, None),
        ({"cleanup": True}, None),
    ],
)
def test_infrastructure_classification(kwargs, cause):
    assert infrastructure_cause(record(1, **kwargs).result) == cause


def test_selection_never_picks_a_later_pass():
    executions = [
        record(1),
        record(2, status="completed", cause=None, passed=False),
        record(3, status="completed", cause=None),
    ]
    selected = select(executions, RetryPolicy(infrastructure_retries=1))
    assert (
        selected.selected.trial.attempt == 2
        and not selected.selected.result.grade.passed
    )
    assert (
        select([record(1), record(2)], RetryPolicy(infrastructure_retries=1)).reason
        == "retries_exhausted"
    )
    assert (
        select([record(1, cleanup=True)], RetryPolicy(infrastructure_retries=5)).reason
        == "cleanup_failed"
    )
    assert (
        select(
            [record(1, status="cancelled", cleanup=True)],
            RetryPolicy(infrastructure_retries=5),
        ).reason
        == "cleanup_failed"
    )


async def test_runner_error_is_retried_in_fresh_state(tmp_path, monkeypatch):
    _, path, _ = await locked_example(
        tmp_path, monkeypatch, execution={"infrastructure_retries": 1}
    )
    calls = 0

    def client(spec):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("private runner error")
        return greeting_client(spec)

    _, report = await run_locked(path, tmp_path / "out", client_factory=client)
    assert exit_code(report) == 0 and calls == 3
    assert report["coverage"]["retried"] == 1
    assert report["trials"][0]["attempts"][1]["selected"]


async def test_valid_task_failure_is_not_retried(tmp_path, monkeypatch):
    _, path, _ = await locked_example(
        tmp_path, monkeypatch, execution={"infrastructure_retries": 5}
    )
    calls = 0

    def client(spec):
        nonlocal calls
        calls += 1
        return FakeLLMClient([ScriptedTurn(text="did nothing")])

    _, report = await run_locked(path, tmp_path / "out", client_factory=client)
    assert exit_code(report) == 1 and calls == 2
    assert report["coverage"]["retried"] == 0


async def test_retry_exhaustion_is_invalid(tmp_path, monkeypatch):
    _, path, _ = await locked_example(
        tmp_path, monkeypatch, execution={"infrastructure_retries": 1}
    )

    def broken(spec):
        raise RuntimeError("failed setup")

    _, report = await run_locked(path, tmp_path / "out", client_factory=broken)
    assert exit_code(report) == 2
    assert report["coverage"]["retries_exhausted"] == 2
    assert report["coverage"]["executions"] == 4


@pytest.mark.parametrize("category", ["auth", "configuration"])
async def test_provider_configuration_error_stops_remaining_trials(
    tmp_path, monkeypatch, category
):
    _, path, _ = await locked_example(
        tmp_path,
        monkeypatch,
        execution={
            "infrastructure_retries": 5,
            "retry_on": ["runner", "environment", "provider_transient"],
        },
    )

    class AuthenticationFailure:
        async def stream(self, messages, tools=None):
            raise LLMStreamError(
                "fixture provider failure", retryable=False, category=category
            )
            yield

    calls = 0

    def client(spec):
        nonlocal calls
        calls += 1
        return AuthenticationFailure()

    _, report = await run_locked(path, tmp_path / "out", client_factory=client)
    assert (
        exit_code(report) == 2 and calls == 1 and report["stopped_reason"] == category
    )


@pytest.mark.parametrize("opt_in", [False, True])
async def test_provider_retry_requires_opt_in_and_keeps_billed_usage(
    tmp_path, monkeypatch, opt_in
):
    from lingcore.usage import TokenUsage

    execution = {"infrastructure_retries": 1}
    if opt_in:
        execution["retry_on"] = ["runner", "environment", "provider_transient"]
    _, path, _ = await locked_example(tmp_path, monkeypatch, execution=execution)

    class FailedRequest:
        usage_sink = None

        async def stream(self, messages, tools=None):
            self.usage_sink(TokenUsage("served-failure", 7, 3))
            raise LLMStreamError(
                "fixture outage", retryable=False, category="transient"
            )
            yield

    calls = 0

    def client(spec):
        nonlocal calls
        calls += 1
        return FailedRequest() if calls == 1 else greeting_client(spec)

    _, report = await run_locked(path, tmp_path / "out", client_factory=client)
    assert calls == (3 if opt_in else 2)
    assert exit_code(report) == (0 if opt_in else 1)
    assert report["coverage"]["retried"] == int(opt_in)
    assert report["usage"]["all"]["requests"] == (7 if opt_in else 4)
    assert report["usage"]["all"]["input_tokens"] == (67 if opt_in else 37)
    assert report["usage"]["selected"]["requests"] == (6 if opt_in else 4)
    assert (
        "served-failure" in report["served_models"] and report["served_model_differs"]
    )


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "kind,category",
    [
        ("RateLimitError", "transient"),
        ("InternalServerError", "transient"),
        ("APIConnectionError", "transient"),
        ("APITimeoutError", "transient"),
        ("BadRequestError", "request"),
        ("NotFoundError", "configuration"),
        ("UnprocessableEntityError", "request"),
        ("AuthenticationError", "auth"),
        ("PermissionDeniedError", "auth"),
    ],
)
async def test_sdk_open_error_categories(monkeypatch, backend, kind, category):
    import importlib

    sdk = importlib.import_module(backend)
    http = importlib.import_module("httpx" if backend == "openai" else "httpx2")
    request = http.Request("POST", "https://example.test/v1")
    cls = getattr(sdk, kind)
    if kind in {"APIConnectionError", "APITimeoutError"}:
        failure = cls(request=request)
    else:
        status = {
            "RateLimitError": 429,
            "InternalServerError": 500,
            "AuthenticationError": 401,
            "PermissionDeniedError": 403,
            "NotFoundError": 404,
        }.get(kind, 400)
        failure = cls(
            "fixture", response=http.Response(status, request=request), body={}
        )
    if backend == "openai":
        from lingcore.llm import LLMClient

        client = LLMClient(model="fixture", api_key="fixture")
    else:
        from lingcore.llm_anthropic import AnthropicLLMClient

        client = AnthropicLLMClient(model="fixture", api_key="fixture")

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(client, "_open_stream", fail)
    try:
        with pytest.raises(LLMStreamError) as caught:
            [chunk async for chunk in client.stream([])]
        assert caught.value.category == category
    finally:
        await client._client.close()
