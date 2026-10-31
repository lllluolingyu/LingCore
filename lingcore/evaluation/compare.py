"""Offline paired comparison of preserved jobs, with case-level uncertainty."""

from __future__ import annotations

import math
import random
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from lingcore.evaluation import jobs
from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.identity import PlanIdentity
from lingcore.evaluation.models import EvaluationDiagnostic, Grade, TrialResult
from lingcore.evaluation.regrade import read_report
from lingcore.evaluation.report import build_job_report
from lingcore.evaluation.runner import _plan_record
from lingcore.evaluation.storage import job_lock, read_json
from lingcore.paths import PathEscapeError


@dataclass
class ComparisonJob:
    plan: PlanIdentity
    report: dict[str, Any]
    graders: dict[str, str]
    results: dict[str, TrialResult]


def _invalid() -> EvaluationConfigError:
    return EvaluationConfigError(
        "comparison report does not match its job plan", "invalid_comparison_job"
    )


def _load(job: Path, grades: str) -> ComparisonJob:
    plan, provenance = _plan_record(job)
    report = read_report(job, grades)
    if report["schema_version"] != "lingcore.eval-report.v2":
        raise EvaluationConfigError("comparison requires v2 job reports", "legacy_job")
    if any(
        report.get(k) != provenance[k] for k in ("plan_digest", "lock_digest", "locked")
    ):
        raise _invalid()
    # read_report validates a requested grade id. Validate the stored one too
    # before using it as a locator, including for an original report.
    grade_id = report.get("grade_set")
    if not isinstance(grade_id, str):
        raise _invalid()
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", grade_id):
        raise _invalid()
    header = read_json(job / "grade-sets" / (grade_id + ".json"))
    if (
        header.get("id") != grade_id
        or header.get("schema_version") != "lingcore.eval-grades.v1"
    ):
        raise _invalid()
    graders = {c["id"]: c["graders_digest"] for c in header["cases"]}
    if len(graders) != len(header["cases"]) or set(graders) != {
        c.id for c in plan.source.cases
    }:
        raise _invalid()
    if grade_id == "original" and graders != {
        c.id: c.graders_digest for c in plan.source.cases
    }:
        raise _invalid()
    # A report can be left behind by a crash or outlive an unreadable result.
    # Rebuild from the ledger, never turn a stale success summary into evidence.
    state = jobs.scan(job, plan)
    overrides: dict[str, Grade] | None = None
    if grade_id != "original":
        overrides = {}
        for execution in state.executions:
            if execution.result is None:
                continue
            try:
                value = read_json(execution.directory / "grades" / (grade_id + ".json"))
            except (FileNotFoundError, PathEscapeError) as exc:
                if isinstance(exc, FileNotFoundError) or isinstance(
                    exc.__cause__, FileNotFoundError
                ):
                    continue
                raise
            if (
                value.get("execution_id") != execution.trial.execution_id
                or value.get("grade_set") != grade_id
                or value.get("schema_version") != "lingcore.eval-grade.v1"
            ):
                raise _invalid()
            overrides[execution.trial.execution_id] = Grade.model_validate(
                value["grade"]
            )
    report = build_job_report(
        plan,
        state,
        locked=provenance["locked"],
        lock_digest=provenance["lock_digest"],
        plan_digest=provenance["plan_digest"],
        grade_set=grade_id,
        grades=overrides,
        interrupted=bool(report.get("interrupted")),
        diagnostics=[
            EvaluationDiagnostic.model_validate(d)
            for d in report.get("diagnostics", [])
        ],
    )
    report["complete"] &= header.get("complete", True)
    planned = {t.trial_id: t for t in plan.trials}
    selections = {t["trial_id"]: t["selected_execution"] for t in report["trials"]}
    if set(selections) != set(planned) or len(selections) != len(report["trials"]):
        raise _invalid()
    results: dict[str, TrialResult] = {}
    for case in report["cases"]:
        for raw in case["results"]:
            result = TrialResult.model_validate(raw)
            spec = planned.get(result.trial_id)
            if (
                spec is None
                or result.trial_id in results
                or result.case_id != spec.case_id
                or result.repetition != spec.repetition
                or result.case_id != case["case_id"]
                or result.plan_digest != report["plan_digest"]
                or result.execution_id != selections[result.trial_id]
            ):
                raise _invalid()
            results[result.trial_id] = result
    if report["coverage"]["planned"] != len(planned):
        raise _invalid()
    return ComparisonJob(plan, report, graders, results)


def _compatible(left: ComparisonJob, right: ComparisonJob) -> None:
    def source(job: ComparisonJob) -> dict[str, Any]:
        return {c.id: (c.revision, c.input_digest) for c in job.plan.source.cases}

    def trials(job: ComparisonJob) -> list[tuple[str, int]]:
        return [(t.case_id, t.repetition) for t in job.plan.trials]

    sections = {
        "source.kind": (left.plan.source.kind, right.plan.source.kind),
        "source.inputs": (source(left), source(right)),
        "track": (left.plan.suite.track, right.plan.suite.track),
        "trials": (trials(left), trials(right)),
        "execution": (
            left.plan.execution.model_dump(exclude={"digest"}),
            right.plan.execution.model_dump(exclude={"digest"}),
        ),
        "trace": (left.plan.trace.model_dump(), right.plan.trace.model_dump()),
        "grading.policy": (
            left.plan.grading.model_dump(exclude={"digest"}),
            right.plan.grading.model_dump(exclude={"digest"}),
        ),
        "grading.cases": (left.graders, right.graders),
    }
    changed = [name for name, (a, b) in sections.items() if a != b]
    if changed:
        raise EvaluationConfigError(
            "incompatible comparison: " + ", ".join(changed), "comparison_mismatch"
        )


def _score(result: TrialResult | None, reward: str | None) -> float | None:
    if result is None or result.grade.status != "valid" or result.cleanup_failed:
        return None
    if reward is None:
        return float(result.grade.passed) if result.grade.passed is not None else None
    value = result.grade.scores.get(reward)
    if value is None or not math.isfinite(value):
        return None
    return value


def _interval(deltas: list[float], samples: int, seed: int) -> list[float] | None:
    if len(deltas) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))

    def percentile(p: float) -> float:
        position = p * (samples - 1)
        low = int(position)
        high = min(low + 1, samples - 1)
        return draws[low] + (draws[high] - draws[low]) * (position - low)

    return [percentile(0.025), percentile(0.975)]


def _usage(left: ComparisonJob, right: ComparisonJob) -> dict[str, Any]:
    result = {}
    for scope in ("selected", "all"):
        a, b = left.report["usage"][scope], right.report["usage"][scope]
        complete = a["coverage"] == b["coverage"] == "complete"
        result[scope] = {
            "baseline": a,
            "candidate": b,
            "delta": {
                key: b[key] - a[key] if complete else None
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                    "reasoning_tokens",
                    "requests",
                )
            },
            "coverage": "complete" if complete else "incomplete",
        }
    return result


def compare_jobs(
    baseline: Path,
    candidate: Path,
    *,
    baseline_grades: str = "original",
    candidate_grades: str = "original",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare immutable evidence; no source files, model, agent or tool required.

    Exit status is independent of bootstrap uncertainty: any lower per-case mean
    is a regression. Incomplete evidence has only descriptive paired differences.
    """
    if not 100 <= bootstrap_samples <= 10000:
        raise EvaluationConfigError(
            "bootstrap_samples must be between 100 and 10000", "comparison_samples"
        )
    baseline, candidate = baseline.absolute(), candidate.absolute()
    with ExitStack() as stack:
        for path in sorted({baseline, candidate}):
            stack.enter_context(job_lock(path))
        left, right = (
            _load(baseline, baseline_grades),
            _load(candidate, candidate_grades),
        )
    _compatible(left, right)
    reward = getattr(left.plan.grading, "reward", None)
    reward_name = reward.name if reward is not None else None
    right_trials = {(t.case_id, t.repetition): t for t in right.plan.trials}
    cases: dict[str, dict[str, Any]] = {}
    paired = 0
    latencies: list[float] = []
    for spec in left.plan.trials:
        other = right_trials[(spec.case_id, spec.repetition)]
        a, b = left.results.get(spec.trial_id), right.results.get(other.trial_id)
        av, bv = _score(a, reward_name), _score(b, reward_name)
        delta = bv - av if av is not None and bv is not None else None
        row = cases.setdefault(spec.case_id, {"case_id": spec.case_id, "trials": []})
        row["trials"].append(
            {
                "repetition": spec.repetition,
                "baseline": av,
                "candidate": bv,
                "delta": delta,
                "baseline_status": "missing" if a is None else a.grade.status,
                "candidate_status": "missing" if b is None else b.grade.status,
            }
        )
        if delta is not None:
            paired += 1
            assert a is not None and b is not None
            latencies.append(b.elapsed_ms - a.elapsed_ms)
    case_deltas = []
    for row in cases.values():
        pairs = [t for t in row["trials"] if t["delta"] is not None]
        row.update(
            planned=len(row["trials"]),
            paired=len(pairs),
            baseline_mean=mean(t["baseline"] for t in pairs) if pairs else None,
            candidate_mean=mean(t["candidate"] for t in pairs) if pairs else None,
            delta=mean(t["delta"] for t in pairs) if pairs else None,
        )
        row["complete"] = len(pairs) == len(row["trials"])
        row["change"] = (
            "unpaired"
            if not pairs
            else "regression"
            if row["delta"] < 0
            else "improvement"
            if row["delta"] > 0
            else "unchanged"
        )
        if pairs:
            case_deltas.append(row["delta"])
    complete = bool(
        left.report["complete"]
        and right.report["complete"]
        and paired == len(left.plan.trials)
    )
    regressions = [c["case_id"] for c in cases.values() if c["change"] == "regression"]
    improvements = [
        c["case_id"] for c in cases.values() if c["change"] == "improvement"
    ]
    interval = _interval(case_deltas, bootstrap_samples, seed) if complete else None

    def identity(job: ComparisonJob) -> dict[str, Any]:
        return {
            "plan_digest": job.report["plan_digest"],
            "lock_digest": job.report["lock_digest"],
            "locked": job.report["locked"],
            "grade_set": job.report["grade_set"],
            "candidate_digest": job.plan.candidate.digest,
            "runtime_digest": job.plan.runtime.digest,
            "model": job.plan.candidate.model.model_dump(),
            "coverage": job.report["coverage"],
            "served_models": job.report.get("served_models", []),
            "served_model_differs": job.report.get("served_model_differs", False),
        }

    return {
        "schema_version": "lingcore.eval-comparison.v1",
        "baseline": identity(left),
        "candidate": identity(right),
        "complete": complete,
        "score_scope": "complete_paired" if complete else "partial_descriptive",
        "metric": "mean_reward" if reward_name else "mean_case_success",
        "coverage": {
            "planned_pairs": len(left.plan.trials),
            "valid_pairs": paired,
            "unpaired": len(left.plan.trials) - paired,
            "cases": len(cases),
        },
        "mean_delta": mean(case_deltas) if complete and case_deltas else None,
        "descriptive_paired_delta": mean(case_deltas) if case_deltas else None,
        "uncertainty": {
            "method": "paired_case_bootstrap",
            "unit": "case",
            "confidence": 0.95,
            "samples": bootstrap_samples,
            "seed": seed,
            "interval": interval,
            "reason": None
            if interval is not None
            else "incomplete_pairs"
            if not complete
            else "fewer_than_two_cases",
        },
        "regressions": regressions,
        "improvements": improvements,
        "cases": list(cases.values()),
        "usage": _usage(left, right),
        "paired_mean_latency_delta_ms": mean(latencies) if latencies else None,
        "exit_code": 2 if not complete else 1 if regressions else 0,
    }
