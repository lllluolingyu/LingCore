"""Atomic admission and post-request accounting across worker budget trees."""

from __future__ import annotations

import pytest

from lingcore.errors import BudgetExceeded
from lingcore.execution import ExecutionBudget, ExecutionLimits
from lingcore.usage import RequestUsage


def usage(request_id, tokens=12, availability="reported"):
    return RequestUsage(request_id, "main", "fake", availability, tokens, 3)


@pytest.mark.parametrize(
    "method,limit,counter",
    [
        ("reserve_request", "model_requests", "requests"),
        ("reserve_tool", "tool_calls", "tools"),
        ("reserve_agent", "agents", "agents"),
        ("reserve_automatic_turn", "automatic_turns", "automatic_turns"),
    ],
)
@pytest.mark.parametrize("reject_at", [0, 1, 2])
def test_rejected_admission_spends_nothing_at_any_level(
    method, limit, counter, reject_at
):
    chain = []
    for i in range(3):
        chain.append(
            ExecutionBudget(
                ExecutionLimits(**{limit: 0 if i == reject_at else 2}),
                parent=chain[-1] if chain else None,
            )
        )
    with pytest.raises(BudgetExceeded) as error:
        getattr(chain[-1], method)()
    assert error.value.limit == limit
    assert all(getattr(b, counter) == 0 for b in chain)


@pytest.mark.parametrize(
    "method,counter",
    [
        ("reserve_request", "requests"),
        ("reserve_tool", "tools"),
        ("reserve_agent", "agents"),
        ("reserve_automatic_turn", "automatic_turns"),
    ],
)
def test_success_charges_each_ancestor_once(method, counter):
    root = ExecutionBudget(ExecutionLimits(automatic_turns=2))
    child = root.child(ExecutionLimits(automatic_turns=2))
    getattr(child, method)()
    assert getattr(root, counter) == getattr(child, counter) == 1


def test_inflight_overshoot_duplicate_and_late_usage_are_all_accounted():
    root = ExecutionBudget(ExecutionLimits(reported_tokens=20))
    first, second = root.child(), root.child()
    first.reserve_request()
    second.reserve_request()
    first.record_usage(usage("a"))
    second.record_usage(usage("b"))
    assert root.tokens == 30 and root.remaining_tokens == 0
    with pytest.raises(BudgetExceeded, match="reported_tokens"):
        first.reserve_request()
    # A duplicate reaching the root through another branch is still billed once.
    second.record_usage(usage("a"))
    root.record_usage(usage("a"))
    assert root.tokens == 30
    # Late provider completion remains billable after exhaustion/cancellation.
    first.record_usage(usage("late", 4))
    assert root.tokens == 37 and root.requests == 2


def test_one_request_can_overshoot_and_remaining_uses_narrowest_ancestor():
    root = ExecutionBudget()
    assert root.remaining_tokens is None
    run = root.child(ExecutionLimits(reported_tokens=100))
    worker = run.child(ExecutionLimits(reported_tokens=10))
    assert worker.remaining_tokens == 10
    worker.reserve_request()
    worker.record_usage(usage("a", 120))
    assert worker.remaining_tokens == run.remaining_tokens == 0
    assert root.tokens == 123 and root.remaining_tokens is None


@pytest.mark.parametrize("availability", ["missing", "partial"])
def test_missing_usage_stops_siblings_but_accepts_late_accounting(availability):
    root = ExecutionBudget(ExecutionLimits(usage_missing="stop"))
    a, b = root.child(), root.child()
    a.reserve_request()
    b.reserve_request()
    a.record_usage(usage("a", 0, availability))
    with pytest.raises(BudgetExceeded, match="usage_unavailable"):
        b.reserve_request()
    b.record_usage(usage("b"))
    assert root.requests == 2 and root.tokens == 18 and not root.usage_complete


def test_start_preserves_ancestor_deadline_and_child_cannot_extend_it(monkeypatch):
    monkeypatch.setattr("lingcore.execution.time.monotonic", lambda: 100)
    root = ExecutionBudget(ExecutionLimits(wall_seconds=10))
    root.start()
    child = root.child(ExecutionLimits(wall_seconds=100))
    monkeypatch.setattr("lingcore.execution.time.monotonic", lambda: 105)
    child.start()
    assert child.deadline == root.deadline == 110
    child.deadline = 1000
    assert child.deadline == 110
    monkeypatch.setattr("lingcore.execution.time.monotonic", lambda: 111)
    with pytest.raises(BudgetExceeded, match="wall_seconds"):
        child.reserve_agent()
    assert root.agents == child.agents == 0
