"""Append-only execution ledger and grade-independent attempt selection."""

from __future__ import annotations

import errno
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.identity import (
    PlanIdentity,
    RetryPolicy,
    identity_body,
    value_digest,
)
from lingcore.evaluation.lock import utc_now
from lingcore.evaluation.models import (
    RetryCause,
    StrictModel,
    TrialResult,
    TrialSpec,
    UsageSummary,
)
from lingcore.evaluation.storage import directory, read_json, write_json
from lingcore.paths import PathEscapeError
from lingcore.tracing.inspect import inspect_trace


class TrialRecord(TrialSpec):
    execution_id: str
    attempt: int = Field(ge=1)
    started_at: str
    plan_digest: str
    scratch_dir: str


class Interruption(StrictModel):
    detected_at: str
    model_requests: int = 0
    tool_calls: int = 0
    served_models: list[str] = Field(default_factory=list)
    usage: UsageSummary = Field(
        default_factory=lambda: UsageSummary(coverage="incomplete")
    )


class ScratchCleanup(StrictModel):
    status: Literal["removed", "missing", "refused", "failed"]


IssueCode = Literal[
    "unreadable_trial",
    "unreadable_result",
    "unreadable_interruption",
    "unreadable_cleanup",
]


class LedgerIssue(StrictModel):
    execution_id: str
    record: str
    code: IssueCode
    error_type: str


@dataclass(frozen=True)
class ExecutionRecord:
    trial: TrialRecord
    directory: Path
    state: Literal["finished", "interrupted", "orphaned"]
    result: TrialResult | None = None
    interruption: Interruption | None = None
    scratch_cleanup: str | None = None

    @property
    def interrupted(self) -> bool:
        return self.state != "finished" or bool(
            self.result and self.result.execution.status in {"cancelled", "interrupted"}
        )


@dataclass(frozen=True)
class JobState:
    executions: tuple[ExecutionRecord, ...]
    issues: tuple[LedgerIssue, ...] = ()

    def for_trial(self, trial_id: str) -> list[ExecutionRecord]:
        return sorted(
            (e for e in self.executions if e.trial.trial_id == trial_id),
            key=lambda e: e.trial.attempt,
        )


@dataclass(frozen=True)
class Selection:
    selected: ExecutionRecord | None = None
    resolved: bool = False
    reason: str | None = None
    stopped: ExecutionRecord | None = None


def infrastructure_cause(result: TrialResult) -> RetryCause | None:
    if result.cleanup_failed:
        return None
    outcome = result.execution
    if outcome.status != "failed":
        return None
    if outcome.cause in {"runner", "environment"}:
        return "runner" if outcome.cause == "runner" else "environment"
    if outcome.cause == "provider" and outcome.provider_error_category in {
        "transient",
        "interrupted",
    }:
        return "provider_transient"
    return None


def fatal_reason(result: TrialResult) -> str | None:
    if result.execution.cause == "configuration":
        return "configuration"
    if (
        result.execution.cause == "provider"
        and result.execution.provider_error_category in {"auth", "configuration"}
    ):
        return result.execution.provider_error_category
    return None


def select(executions: list[ExecutionRecord], policy: RetryPolicy) -> Selection:
    failures = 0
    ordered = sorted(executions, key=lambda e: e.trial.attempt)
    for index, record in enumerate(ordered):
        if record.result and record.result.cleanup_failed:
            return Selection(resolved=True, reason="cleanup_failed")
        if record.interrupted:
            continue
        result = record.result
        assert result is not None
        fatal = fatal_reason(result)
        if fatal:
            # A later start record proves an explicit resume replaced this stop.
            # The executor never starts another attempt after a new fatal error
            # in the same invocation. Neither grades nor retry allowances enter
            # this decision.
            if index < len(ordered) - 1:
                continue
            return Selection(resolved=True, reason=fatal, stopped=record)
        if infrastructure_cause(result) in policy.retry_on:
            failures += 1
            if failures > policy.infrastructure_retries:
                return Selection(resolved=True, reason="retries_exhausted")
            continue
        return Selection(record, True)
    return Selection()


def pending(
    plan: PlanIdentity, state: JobState, policy: RetryPolicy
) -> list[TrialSpec]:
    return [
        trial
        for trial in plan.trials
        if not select(state.for_trial(trial.trial_id), policy).resolved
    ]


_Record = TypeVar("_Record", bound=BaseModel)


def _read_record(
    path: Path, model: type[_Record], issues: list[LedgerIssue], code: IssueCode
) -> _Record | None:
    try:
        return model.model_validate(read_json(path))
    except (OSError, ValueError, PathEscapeError) as exc:
        cause: BaseException | None = exc
        while cause is not None:
            if isinstance(cause, OSError) and cause.errno in {
                errno.ELOOP,
                errno.ENOTDIR,
            }:
                raise EvaluationConfigError(
                    "unsafe execution ledger entry", "invalid_job"
                ) from None
            cause = cause.__cause__
        issues.append(
            LedgerIssue(
                execution_id=path.parent.name,
                record=path.name,
                code=code,
                error_type=type(exc).__name__,
            )
        )
        return None


def _read_marker(
    path: Path,
    prefix: str,
    model: type[_Record],
    issues: list[LedgerIssue],
    code: IssueCode,
) -> _Record | None:
    with directory(path) as root:
        names = []
        for name, _, _, is_link in root.iter_entries():
            if (
                name == prefix + ".json"
                or name.startswith(prefix + "-")
                and name.endswith(".json")
            ):
                if is_link:
                    raise EvaluationConfigError(
                        "unsafe execution ledger entry", "invalid_job"
                    )
                names.append(name)
    result = None
    for name in sorted(names, key=lambda name: (name != prefix + ".json", name)):
        record = _read_record(path / name, model, issues, code)
        if result is None:
            result = record
    return result


def quarantine(job: Path, issue: LedgerIssue) -> None:
    """Keep an unassigned execution out of selection even if its file recovers."""
    if issue.code != "unreadable_trial":
        return
    try:
        write_json(
            job / "quarantined" / issue.execution_id / "issue.json",
            issue.model_dump(mode="json"),
        )
    except FileExistsError:
        pass  # The directory itself is the durable quarantine decision.


def _quarantined(job: Path) -> dict[str, LedgerIssue]:
    root = job / "quarantined"
    if root.is_symlink():
        raise EvaluationConfigError("unsafe quarantine entry", "invalid_job")
    if not root.exists():
        return {}
    with directory(root) as control:
        entries = sorted(control.iter_entries())
    result = {}
    for name, is_dir, _, is_link in entries:
        if is_link:
            raise EvaluationConfigError("unsafe quarantine entry", "invalid_job")
        if not is_dir:
            continue
        issue = LedgerIssue(
            execution_id=name,
            record="trial.json",
            code="unreadable_trial",
            error_type="QuarantinedTrial",
        )
        try:
            saved = LedgerIssue.model_validate(read_json(root / name / "issue.json"))
            if saved.execution_id == name and saved.code == "unreadable_trial":
                issue = saved
        except (OSError, ValueError, PathEscapeError):
            pass  # A damaged diagnostic cannot reverse a persisted quarantine.
        result[name] = issue
    return result


def scan(job: Path, plan: PlanIdentity | None = None) -> JobState:
    executions = job / "executions"
    quarantined = _quarantined(job)
    if not executions.exists():
        return JobState((), tuple(quarantined.values()))
    records = []
    issues = list(quarantined.values())
    with directory(executions) as root:
        names = sorted(root.iter_entries())
    for name, is_dir, _, is_link in names:
        if is_link:
            raise EvaluationConfigError("unsafe execution ledger entry", "invalid_job")
        if not is_dir:
            continue
        if name in quarantined:
            continue
        path = executions / name
        trial = _read_record(
            path / "trial.json", TrialRecord, issues, "unreadable_trial"
        )
        if trial is None:
            continue  # Keep unassigned directories untouched and visible in reports.
        if trial.execution_id != name:
            raise EvaluationConfigError("execution identity mismatch", "invalid_job")
        interruption = _read_marker(
            path, "interrupted", Interruption, issues, "unreadable_interruption"
        )
        if (path / "result.json").exists() or (path / "result.json").is_symlink():
            result = _read_record(
                path / "result.json", TrialResult, issues, "unreadable_result"
            )
            if result is not None:
                if (result.execution_id, result.trial_id, result.attempt) != (
                    name,
                    trial.trial_id,
                    trial.attempt,
                ):
                    raise EvaluationConfigError(
                        "result identity mismatch", "invalid_job"
                    )
                if interruption is None:
                    records.append(ExecutionRecord(trial, path, "finished", result))
                    continue
        cleanup = _read_marker(
            path, "scratch-cleanup", ScratchCleanup, issues, "unreadable_cleanup"
        )
        records.append(
            ExecutionRecord(
                trial,
                path,
                "interrupted" if interruption else "orphaned",
                interruption=interruption,
                scratch_cleanup=cleanup.status if cleanup else None,
            )
        )
    identities = {(r.trial.trial_id, r.trial.attempt) for r in records}
    if len(identities) != len(records):
        raise EvaluationConfigError("duplicate execution attempt", "invalid_job")
    if plan is not None:
        planned = {t.trial_id: t for t in plan.trials}
        revisions = {c.id: c.revision for c in plan.source.cases}
        plan_digest = value_digest(identity_body(plan))
        for record in records:
            trial, execution_result = record.trial, record.result
            spec = planned.get(trial.trial_id)
            if (
                spec is None
                or (trial.case_id, trial.repetition) != (spec.case_id, spec.repetition)
                or trial.plan_digest != plan_digest
                or execution_result is not None
                and (
                    execution_result.case_id != trial.case_id
                    or execution_result.repetition != trial.repetition
                    or execution_result.case_revision != revisions[trial.case_id]
                    or execution_result.plan_digest != plan_digest
                )
            ):
                raise EvaluationConfigError(
                    "execution does not belong to the frozen plan", "invalid_job"
                )
    return JobState(tuple(records), tuple(issues))


def _remove_scratch(path: Path) -> str:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "failed"
    if (
        not path.name.startswith("lingcore-eval-")
        or path.parent.resolve() != Path(tempfile.gettempdir()).resolve()
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
    ):
        return "refused"
    try:
        shutil.rmtree(path)
        return "removed"
    except OSError:
        return "failed"


def mark_interrupted(record: ExecutionRecord) -> None:
    if record.state == "finished":
        return
    if record.state == "orphaned":
        _write_interruption(record)
    if record.scratch_cleanup is None:
        cleanup = _remove_scratch(Path(record.trial.scratch_dir))
        _write_marker(record.directory, "scratch-cleanup", {"status": cleanup})


def _write_marker(directory: Path, prefix: str, value: dict[str, object]) -> None:
    # Preserve even damaged records. A new marker records the recovery instead
    # of rewriting the file that a crash left behind.
    name = prefix + ".json"
    if (directory / name).exists():
        name = prefix + "-" + uuid4().hex + ".json"
    write_json(directory / name, value)


def _write_interruption(record: ExecutionRecord) -> None:
    interruption = Interruption(detected_at=utc_now())
    try:
        trace = inspect_trace(record.directory / "trace")
        usage = {
            r.attributes["request_id"]: r.attributes
            for r in trace.records
            if r.name == "request.usage"
        }
        requests = sum(
            r.name == "llm.request" and r.kind == "span.start" for r in trace.records
        )
        interruption = Interruption(
            detected_at=utc_now(),
            model_requests=requests,
            tool_calls=sum(
                r.name == "tool.dispatch" and r.kind == "span.start"
                for r in trace.records
            ),
            usage=UsageSummary(
                input_tokens=sum(
                    int(u.get("input_tokens") or 0) for u in usage.values()
                ),
                output_tokens=sum(
                    int(u.get("output_tokens") or 0) for u in usage.values()
                ),
                cached_input_tokens=sum(
                    int(u.get("cached_input_tokens") or 0) for u in usage.values()
                ),
                reasoning_tokens=sum(
                    int(u.get("reasoning_tokens") or 0) for u in usage.values()
                ),
                requests=requests,
                coverage="incomplete",
            ),
            served_models=sorted(
                {
                    str(u["model"])
                    for u in usage.values()
                    if u.get("availability") != "missing" and u.get("model")
                }
            ),
        )
    except (ValueError, OSError, PathEscapeError, KeyError, TypeError):
        pass
    _write_marker(record.directory, "interrupted", interruption.model_dump(mode="json"))
