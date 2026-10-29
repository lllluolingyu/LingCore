"""Per-case results with explicit coverage and unlocked development provenance."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from lingcore.evaluation.models import TrialResult


def build_report(
    results: Sequence[TrialResult], planned: int, *, configuration_error: bool = False
) -> dict[str, Any]:
    valid = [r for r in results if r.grade.status == "valid"]
    by_case: dict[str, list[TrialResult]] = defaultdict(list)
    for result in results:
        by_case[result.case_id].append(result)
    cases = [
        {
            "case_id": key,
            "started": len(trials),
            "valid": sum(r.grade.status == "valid" for r in trials),
            "passed": sum(r.grade.passed is True for r in trials),
            "results": [r.model_dump(mode="json") for r in trials],
        }
        for key, trials in sorted(by_case.items())
    ]
    complete = (
        len(results) == planned
        and len(valid) == planned
        and not configuration_error
        and all(not r.cleanup_failed for r in results)
    )
    means = [
        sum(r.grade.passed is True for r in trials if r.grade.status == "valid") / n
        for trials in by_case.values()
        if (n := sum(r.grade.status == "valid" for r in trials))
    ]
    cancelled = sum(r.execution.status == "cancelled" for r in results)
    return {
        "schema_version": "lingcore.eval-report.v1",
        "locked": False,
        "benchmark_compatible": False,
        "track": "development",
        "complete": complete,
        "coverage": {
            "planned": planned,
            "started": len(results),
            "valid": len(valid),
            "invalid": len(results) - len(valid),
            "unsupported": 0,
            "cancelled": cancelled,
            "missing": max(0, planned - len(results)),
        },
        "valid_failures": sum(r.grade.passed is False for r in valid),
        "mean_case_success": sum(means) / len(means) if means else None,
        "score_scope": "complete_development" if complete else "valid_only_descriptive",
        "usage": {
            "input_tokens": sum(r.usage.input_tokens for r in results),
            "output_tokens": sum(r.usage.output_tokens for r in results),
            "coverage": "complete"
            if all(r.usage.coverage == "complete" for r in results)
            and len(results) == planned
            else "incomplete",
            "estimated_cost_usd": None,
        },
        "latency_ms": {
            "total": sum(r.elapsed_ms for r in results),
            "per_trial": [r.elapsed_ms for r in results],
        },
        "trace_health": {
            "complete_executions": sum(r.trace.complete for r in results),
            "dropped_records": sum(r.trace.dropped_records for r in results),
            "disk_errors": sum(r.trace.disk_errors for r in results),
        },
        "cases": cases,
    }


def exit_code(report: dict[str, Any]) -> int:
    if report["coverage"]["cancelled"] or report.get("interrupted"):
        return 130
    if not report["complete"]:
        return 2
    return 1 if report["valid_failures"] else 0
