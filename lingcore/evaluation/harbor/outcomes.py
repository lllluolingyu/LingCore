"""Ordered Harbor outcome mapping; rewards never decide retry selection."""

from __future__ import annotations

import math
from typing import Any

from lingcore.evaluation.models import Grade, RewardPolicy
from lingcore.execution import ExecutionOutcome


def map_outcome(
    summary: dict[str, Any] | None,
    outcome: ExecutionOutcome,
    reward: RewardPolicy,
    *,
    hello: bool,
    unsupported: bool = False,
    cleanup_failed: bool = False,
    protocol_failed: bool = False,
) -> tuple[ExecutionOutcome, Grade]:
    def invalid(reason: str, status: str = "invalid") -> Grade:
        return Grade.model_validate({"status": status, "reason": reason})

    if summary is None:
        return (
            outcome
            if outcome.status == "timed_out"
            else ExecutionOutcome(status="failed", cause="runner")
        ), invalid("harbor_result_missing", "not_run")
    cause = summary.get("failure_cause")
    if not hello and cause == "sandbox_failed":
        return ExecutionOutcome(status="failed", cause="environment"), invalid(
            "environment_failed", "not_run"
        )
    if not hello:
        return ExecutionOutcome(status="failed", cause="runner"), invalid(
            "hello_missing", "not_run"
        )
    if unsupported:
        return ExecutionOutcome(status="failed", cause="environment"), invalid(
            "unsupported", "not_run"
        )
    if cause == "harbor_failed":
        return ExecutionOutcome(status="failed", cause="runner"), invalid(
            "harbor_failed", "not_run"
        )
    if cleanup_failed or summary.get("metadata", {}).get("sweep") == "failed":
        return outcome, invalid("cleanup_failed")
    if protocol_failed:
        return ExecutionOutcome(status="failed", cause="runner"), invalid(
            "protocol_failed"
        )
    if outcome.cause == "provider":
        return outcome, invalid("provider_failed")
    if any(e["type"] == "AgentTimeoutError" for e in summary.get("exceptions", [])):
        outcome = ExecutionOutcome(
            status="timed_out", cause="agent", limit_type="harness_agent"
        )
    if cause == "verifier_failed":
        return outcome, invalid("verifier_failed")
    rewards = summary.get("rewards", {})
    value = rewards.get(reward.name) if isinstance(rewards, dict) else None
    if (
        not summary.get("reward_valid")
        or type(value) not in {int, float}
        or value is None
        or not math.isfinite(value)
        or value not in reward.values
    ):
        return outcome, invalid("reward_invalid")
    return outcome, Grade(
        status="valid", passed=value >= reward.pass_at, scores={reward.name: value}
    )
