"""Derived per-case reports with provenance, attempt coverage, and billed usage."""

from __future__ import annotations

import re
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
    if report.get("interrupted") or (
        report.get("schema_version") == "lingcore.eval-report.v1"
        and report["coverage"]["cancelled"]
    ):
        return 130
    if not report["complete"]:
        return 2
    return 1 if report["valid_failures"] else 0


def build_job_report(
    plan: Any,
    state: Any,
    *,
    locked: bool,
    lock_digest: str | None,
    plan_digest: str,
    grade_set: str = "original",
    grades: dict[str, Any] | None = None,
    interrupted: bool = False,
) -> dict[str, Any]:
    from lingcore.evaluation.jobs import fatal_reason, infrastructure_cause, select

    results = []
    trials = []
    invalid = 0
    started = 0
    retried = 0
    replaced_count = 0
    stop_reasons = []
    for trial in plan.trials:
        executions = state.for_trial(trial.trial_id)
        started += bool(executions)
        selection = select(executions, plan.execution.retry)
        selected = selection.selected
        if selection.reason:
            stop_reasons.append(selection.reason)
        if selected and selected.result:
            result = selected.result
            if grades is not None:
                from lingcore.evaluation.models import Grade

                result = result.model_copy(
                    update={
                        "grade": grades.get(
                            result.execution_id,
                            Grade(status="not_run", reason="grade_set_incomplete"),
                        )
                    }
                )
            results.append(result)
            invalid += result.grade.status != "valid"
        elif selection.resolved:
            invalid += 1
        for i, execution in enumerate(executions):
            replaced_count += execution.interrupted and i < len(executions) - 1
            previous = next(
                (e for e in reversed(executions[:i]) if not e.interrupted), None
            )
            if (
                not execution.interrupted
                and previous is not None
                and previous.result
                and infrastructure_cause(previous.result)
                in plan.execution.retry.retry_on
            ):
                retried += 1
        trials.append(
            {
                **trial.model_dump(),
                "selected_execution": selected.trial.execution_id if selected else None,
                "reason": selection.reason,
                "attempts": [
                    {
                        "execution_id": e.trial.execution_id,
                        "attempt": e.trial.attempt,
                        "status": e.result.execution.status
                        if e.result
                        else "interrupted"
                        if e.interrupted
                        else e.state,
                        "cause": e.result.execution.cause if e.result else None,
                        "stop_reason": fatal_reason(e.result) if e.result else None,
                        "scratch_cleanup": e.scratch_cleanup,
                        "selected": selected is not None
                        and selected.trial.execution_id == e.trial.execution_id,
                    }
                    for e in executions
                ],
            }
        )
    report = build_report(results, len(plan.trials))
    valid = sum(r.grade.status == "valid" for r in results)
    report.update(
        schema_version="lingcore.eval-report.v2",
        locked=locked,
        lock_digest=lock_digest,
        plan_digest=plan_digest,
        grade_set=grade_set,
        trials=trials,
        interrupted=interrupted,
    )
    report["complete"] = valid == len(plan.trials) and invalid == 0 and not interrupted
    report["score_scope"] = (
        "complete_development" if report["complete"] else "valid_only_descriptive"
    )
    report["coverage"].update(
        started=started,
        valid=valid,
        invalid=invalid,
        missing=len(plan.trials) - valid - invalid,
        cancelled=sum(
            bool(e.result and e.result.execution.status == "cancelled")
            for e in state.executions
        ),
        interrupted=sum(e.interrupted for e in state.executions),
        replaced=replaced_count,
        retried=retried,
        retries_exhausted=stop_reasons.count("retries_exhausted"),
        executions=len(state.executions)
        + sum(i.code == "unreadable_trial" for i in state.issues),
        unassigned_executions=sum(i.code == "unreadable_trial" for i in state.issues),
    )
    report["ledger_issues"] = [issue.model_dump(mode="json") for issue in state.issues]

    def usage_total(usages: Any) -> dict[str, Any]:
        usages = list(usages)
        return {
            "input_tokens": sum(u.input_tokens for u in usages),
            "output_tokens": sum(u.output_tokens for u in usages),
            "cached_input_tokens": sum(u.cached_input_tokens for u in usages),
            "reasoning_tokens": sum(u.reasoning_tokens for u in usages),
            "requests": sum(u.requests for u in usages),
            "coverage": "complete"
            if all(u.coverage == "complete" for u in usages)
            else "incomplete",
            "estimated_cost_usd": None,
        }

    all_usage = [
        e.result.usage if e.result else e.interruption.usage
        for e in state.executions
        if e.result or e.interruption
    ]
    report["usage"]["selected"] = usage_total(r.usage for r in results)
    report["usage"]["all"] = usage_total(all_usage)
    if len(results) != len(plan.trials):
        report["usage"]["selected"]["coverage"] = "incomplete"
    if any(i.code == "unreadable_trial" for i in state.issues) or any(
        not e.result and not e.interruption for e in state.executions
    ):
        report["usage"]["all"]["coverage"] = "incomplete"
    models = sorted(
        {
            model
            for e in state.executions
            for model in (
                e.result.served_models
                if e.result
                else e.interruption.served_models
                if e.interruption
                else []
            )
        }
    )
    report["served_models"] = models
    configured = {plan.candidate.model.id}
    if plan.candidate.vision_model is not None:
        configured.add(plan.candidate.vision_model.id)
    report["served_model_differs"] = any(
        not _model_matches(model, configured) for model in models
    )
    if stop_reasons:
        report["selection_reasons"] = sorted(set(stop_reasons))
    fatal = next(
        (reason for reason in stop_reasons if reason in {"auth", "configuration"}), None
    )
    if fatal:
        report["stopped_reason"] = fatal
    # Include cases with no selected execution so missing/invalid work is visible.
    for case in plan.source.cases:
        if not any(row["case_id"] == case.id for row in report["cases"]):
            report["cases"].append(
                {
                    "case_id": case.id,
                    "started": sum(
                        t["case_id"] == case.id and bool(t["attempts"]) for t in trials
                    ),
                    "valid": 0,
                    "passed": 0,
                    "results": [],
                }
            )
    for row in report["cases"]:
        row["started"] = sum(
            t["case_id"] == row["case_id"] and bool(t["attempts"]) for t in trials
        )
    return report


def _model_matches(served: str, configured: set[str]) -> bool:
    if served in configured:
        return True
    # Providers may resolve an unversioned alias to a dated snapshot. A lock
    # that already requests a specific snapshot must still detect another date.
    match = re.fullmatch(r"(.+)-\d{4}-\d{2}-\d{2}", served)
    return bool(match and match[1] in configured)
