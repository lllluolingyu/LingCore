from dataclasses import replace
from types import SimpleNamespace

import pytest

from lingcore.evaluation.identity import RetryPolicy
from lingcore.evaluation.jobs import JobState
from lingcore.evaluation.models import TrialSpec
from lingcore.evaluation.report import build_job_report
from tests.test_eval_retries import record


def report(executions, *, model="gpt-4o", vision="vision-model"):
    plan = SimpleNamespace(
        trials=[TrialSpec(trial_id="one", case_id="one", repetition=1)],
        source=SimpleNamespace(cases=[SimpleNamespace(id="one")]),
        candidate=SimpleNamespace(
            model=SimpleNamespace(id=model),
            vision_model=SimpleNamespace(id=vision) if vision else None,
        ),
        execution=SimpleNamespace(retry=RetryPolicy(infrastructure_retries=1)),
    )
    return build_job_report(
        plan,
        JobState(tuple(executions)),
        locked=True,
        lock_digest="lock",
        plan_digest="plan",
    )


@pytest.mark.parametrize(
    "served,expected",
    [
        ("gpt-4o", False),
        ("gpt-4o-2024-08-06", False),
        ("vision-model", False),
        ("vision-model-2025-04-09", False),
        ("gpt-4o-mini", True),
        ("another-provider-model", True),
    ],
)
def test_served_models_match_all_configured_models_and_dated_aliases(served, expected):
    attempt = record(1, status="completed", cause=None)
    attempt = replace(
        attempt, result=attempt.result.model_copy(update={"served_models": [served]})
    )
    value = report([attempt])
    assert value["served_model_differs"] is expected
    assert value["served_models"] == [served]


def test_a_requested_snapshot_still_detects_a_different_snapshot():
    attempt = record(1, status="completed", cause=None)
    attempt = replace(
        attempt,
        result=attempt.result.model_copy(
            update={"served_models": ["gpt-4o-2024-09-01"]}
        ),
    )
    assert report([attempt], model="gpt-4o-2024-08-06")["served_model_differs"]


def test_resumed_auth_failure_does_not_count_as_another_infrastructure_retry():
    value = report(
        [
            record(1, cause="runner"),
            record(2, cause="provider", category="auth"),
            record(3, status="completed", cause=None),
        ]
    )
    assert value["coverage"]["retried"] == 1
    assert "stopped_reason" not in value
    assert value["trials"][0]["selected_execution"] == "3"
