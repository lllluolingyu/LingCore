"""Shared locked/unlocked native execution with an append-only job ledger."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from lingcore.agent import Agent
from lingcore.errors import ConfigError
from lingcore.evaluation import evidence, jobs
from lingcore.evaluation.config import (
    EvaluationConfigError,
    ResolvedSuite,
    load_yaml,
    resolve_suite,
)
from lingcore.evaluation.graders import grade
from lingcore.evaluation.identity import (
    PlanIdentity,
    build_identity,
    identity_body,
    value_digest,
)
from lingcore.evaluation.lock import Lock, parse_lock, utc_now, validate
from lingcore.evaluation.models import (
    Comparison,
    EvaluationDiagnostic,
    EvaluationPhase,
    Grade,
    Suite,
    TrialResult,
    TrialSpec,
    UsageSummary,
)
from lingcore.evaluation.report import build_job_report, build_report
from lingcore.evaluation.storage import (
    directory,
    finish_io,
    job_lock,
    read_bytes,
    read_json,
    write_bytes,
    write_json,
)
from lingcore.execution import ExecutionBudget, ExecutionOutcome, run_headless
from lingcore.tracing import TraceHealth, TraceRecorder
from lingcore.tracing.config import known_secrets
from lingcore.tracing.content import redact_known_secrets
from lingcore.tracing.context import new_context
from lingcore.tracing.local import LocalTraceSink
from lingcore.tracing.requests import StreamingClient
from lingcore.usage import RequestUsage

ClientFactory = Callable[[TrialSpec], StreamingClient]
_LOG = logging.getLogger(__name__)


def _usage_summary(records: tuple[RequestUsage, ...]) -> UsageSummary:
    return UsageSummary(
        input_tokens=sum(u.input_tokens or 0 for u in records),
        output_tokens=sum(u.output_tokens or 0 for u in records),
        cached_input_tokens=sum(u.cached_input_tokens or 0 for u in records),
        reasoning_tokens=sum(u.reasoning_tokens or 0 for u in records),
        requests=len(records),
        coverage="complete"
        if all(u.availability == "reported" for u in records)
        else "incomplete",
    )


def _plan_record(job: Path) -> tuple[PlanIdentity, dict[str, Any]]:
    record = read_json(job / "plan.json")
    if record.get("schema_version") != "lingcore.eval-plan.v2":
        raise EvaluationConfigError(
            "v1 jobs have no frozen identity; resolve a lock and start a new job",
            "legacy_job",
        )
    plan = PlanIdentity.model_validate(record["identity"])
    if value_digest(identity_body(plan)) != record["plan_digest"]:
        raise EvaluationConfigError(
            "job plan digest does not match its identity", "invalid_job"
        )
    metadata = read_json(job / "job.json")
    if any(
        record[key] != metadata.get(key)
        for key in ("locked", "lock_digest", "plan_digest")
    ):
        raise EvaluationConfigError(
            "job provenance does not match its plan", "invalid_job"
        )
    return plan, record


def _write_plan(
    job: Path, plan: PlanIdentity, *, locked: bool, lock_digest: str | None
) -> None:
    write_json(
        job / "plan.json",
        {
            "schema_version": "lingcore.eval-plan.v2",
            "identity": {
                key: value
                for key, value in plan.model_dump(mode="json").items()
                if key in PlanIdentity.model_fields
            },
            "locked": locked,
            "lock_digest": lock_digest,
            "plan_digest": value_digest(identity_body(plan)),
            "candidate_fingerprint": plan.candidate.digest,
            "planned": len(plan.trials),
            "limits": plan.execution.limits,
            "capture": plan.trace.model_dump(mode="json"),
            "cases": [{"id": c.id, "revision": c.revision} for c in plan.source.cases],
        },
    )


def _write_original_grades(
    job: Path, resolved: ResolvedSuite, plan: PlanIdentity
) -> None:
    write_json(
        job / "grade-sets/original.json",
        {
            "schema_version": "lingcore.eval-grades.v1",
            "id": "original",
            "created_at": utc_now(),
            "cases": [
                {
                    "id": case.id,
                    "graders_digest": identity.graders_digest,
                    "graders": [g.model_dump(mode="json") for g in identity.graders],
                    "specs": [g.model_dump(mode="json") for g in case.graders],
                }
                for case, identity in zip(
                    resolved.cases, plan.source.cases, strict=True
                )
            ],
        },
    )


def _prepare_workspace(scratch: Path, resolved: ResolvedSuite, case_id: str) -> None:
    case = next(c for c in resolved.cases if c.id == case_id)
    workspace = scratch / "workspace"
    with directory(workspace, create=True):
        pass
    for name, content in case.fixture.files.items():
        write_bytes(workspace / name, content.encode("utf-8"))


async def _trial(
    resolved: ResolvedSuite,
    spec: TrialSpec,
    destination: Path,
    client_factory: ClientFactory | None,
    *,
    plan: PlanIdentity,
    attempt: int,
    locked: bool,
    lock_digest: str | None,
) -> TrialResult:
    case = next(c for c in resolved.cases if c.id == spec.case_id)
    execution_id = str(uuid4())
    result_dir = destination / execution_id
    scratch = Path(tempfile.gettempdir()) / ("lingcore-eval-" + execution_id)
    plan_digest = value_digest(identity_body(plan))
    trial = jobs.TrialRecord(
        **spec.model_dump(),
        execution_id=execution_id,
        attempt=attempt,
        started_at=utc_now(),
        plan_digest=plan_digest,
        scratch_dir=str(scratch),
    )
    await finish_io(
        write_json, result_dir / "trial.json", trial.model_dump(mode="json")
    )
    agent: Agent | None = None
    recorder: TraceRecorder | None = None
    outcome = ExecutionOutcome(status="failed", cause="environment")
    grading = Grade(status="not_run")
    diagnostics: list[EvaluationDiagnostic] = []
    phase: EvaluationPhase = "setup"
    execution_finished = False
    created_scratch = False
    cleanup_failed = False
    cancelled = False
    started = time.monotonic()
    elapsed = 0
    try:
        scratch.mkdir(mode=0o700)
        created_scratch = True
        await finish_io(_prepare_workspace, scratch, resolved, case.id)
        profile = resolved.profile.model_copy(deep=True)
        profile.workspace = str(scratch / "workspace")
        recorder = TraceRecorder(
            LocalTraceSink(
                result_dir / "trace",
                manifest={
                    "schema_version": "lingcore.trace.v1",
                    "run_id": execution_id,
                    "trial_id": spec.trial_id,
                    "plan_digest": plan_digest,
                    "capture": plan.trace.model_dump(mode="json"),
                },
                max_bytes=plan.trace.max_bytes,
            ),
            run_id=execution_id,
            content=plan.trace.content,
            artifact_max_bytes=plan.trace.artifact_max_bytes,
            secrets=known_secrets(profile),
        )
        phase = "runner"
        raw = client_factory(spec) if client_factory else None
        phase = "setup"
        agent = Agent.from_profile(
            profile,
            llm=raw,
            state_dir=scratch / "state",
            recorder=recorder,
            execution_budget=ExecutionBudget(resolved.suite.execution.limits),
        )
        phase = "execute"
        headless = await run_headless(
            agent,
            [turn.user for turn in case.input.turns],
            allow_automatic_turns=resolved.suite.execution.limits.automatic_turns > 0,
            trace_context=new_context(
                trial_id=spec.trial_id, execution_id=execution_id
            ),
        )
        outcome, elapsed = headless.outcome, headless.elapsed_ms
        cleanup_failed = headless.cleanup_failed
        execution_finished = True
        phase = "collect"
        await finish_io(
            evidence.collect,
            scratch / "workspace",
            result_dir,
            case,
            plan.trace.content,
            recorder.health,
        )
        collected = await asyncio.to_thread(
            evidence.load,
            result_dir,
            case,
            artifact_max_bytes=plan.trace.artifact_max_bytes,
        )
        phase = "grade"
        grading = await grade(case.graders, collected)
        for grader_result in grading.graders:
            if grader_result.error_type:
                diagnostics.append(
                    EvaluationDiagnostic(
                        phase="grade",
                        code="grader_error",
                        message=f"grader {grader_result.id} failed ({grader_result.error_type})",
                        error_type=grader_result.error_type,
                    )
                )
    except asyncio.CancelledError:
        cancelled = True
        if not execution_finished:
            outcome = ExecutionOutcome(status="cancelled", cause="runner")
        grading = Grade(
            status="invalid" if execution_finished else "not_run",
            reason=f"{phase}_cancelled",
        )
        diagnostics.append(
            EvaluationDiagnostic(
                phase=phase, code="cancelled", message=f"{phase} cancelled"
            )
        )
        raise
    except Exception as exc:
        if not execution_finished:
            outcome = ExecutionOutcome(
                status="failed",
                cause="configuration"
                if isinstance(exc, ConfigError)
                else "environment"
                if phase == "setup"
                else "runner",
            )
        grading = Grade(
            status="invalid" if execution_finished else "not_run",
            reason=f"{phase}_failed",
        )
        diagnostics.append(
            EvaluationDiagnostic(
                phase=phase,
                code=f"{phase}_failed",
                message=f"{phase} failed ({type(exc).__name__})",
                error_type=type(exc).__name__,
            )
        )
    finally:
        if agent is not None and not agent._closed:
            try:
                async with asyncio.timeout(5):
                    await agent.aclose()
            except Exception:
                cleanup_failed = True
        if recorder is not None:
            try:
                await recorder.aclose()
            except Exception:
                cleanup_failed = True
        health = recorder.health if recorder else TraceHealth(complete=False)
        if not (result_dir / "evidence.json").exists():
            try:
                await finish_io(
                    evidence.collect,
                    scratch / "workspace",
                    result_dir,
                    case,
                    plan.trace.content,
                    health,
                )
            except Exception as exc:
                diagnostics.append(
                    EvaluationDiagnostic(
                        phase="collect",
                        code="evidence_failed",
                        message=f"evidence collection failed ({type(exc).__name__})",
                        error_type=type(exc).__name__,
                    )
                )
        if created_scratch:
            try:
                async with asyncio.timeout(5):
                    await asyncio.to_thread(shutil.rmtree, scratch)
            except Exception:
                cleanup_failed = True
        if cleanup_failed:
            grading = grading.model_copy(
                update={"status": "invalid", "passed": None, "reason": "cleanup_failed"}
            )
        records = tuple(agent.request_ledger.records) if agent else ()
        trial_result = TrialResult(
            trial_id=spec.trial_id,
            execution_id=execution_id,
            case_id=case.id,
            case_revision=case.revision,
            repetition=spec.repetition,
            attempt=attempt,
            locked=locked,
            lock_digest=lock_digest,
            plan_digest=plan_digest,
            execution=outcome,
            grade=grading,
            usage=_usage_summary(records),
            trace=health,
            content=plan.trace.content,
            elapsed_ms=elapsed
            if execution_finished
            else round((time.monotonic() - started) * 1000),
            cleanup_failed=cleanup_failed,
            diagnostics=diagnostics,
            comparison=Comparison(locked=locked),
            served_models=sorted(
                {
                    redact_known_secrets(u.model, known_secrets(resolved.profile))
                    for u in records
                    if u.availability != "missing"
                }
            ),
        )
        cause = jobs.infrastructure_cause(trial_result)
        if cause in plan.execution.retry.retry_on:
            trial_result = trial_result.model_copy(
                update={"infrastructure_failure": cause}
            )
        try:
            await finish_io(
                write_json,
                result_dir / "result.json",
                trial_result.model_dump(mode="json"),
            )
        except Exception as exc:
            if not cancelled:
                raise
            _LOG.warning(
                "Cancelled trial result persistence failed (%s)", type(exc).__name__
            )
    return trial_result


async def _execute_job(
    job: Path,
    resolved: ResolvedSuite,
    plan: PlanIdentity,
    *,
    locked: bool,
    lock_digest: str | None,
    client_factory: ClientFactory | None,
    resume_stopped: bool = False,
) -> dict[str, Any]:
    interrupted = False
    diagnostic = None
    try:
        state = await asyncio.to_thread(jobs.scan, job, plan)
        restartable = (
            {e.trial.execution_id for e in state.executions}
            if resume_stopped
            else set()
        )
        for spec in plan.trials:
            while True:
                selection = jobs.select(
                    state.for_trial(spec.trial_id), plan.execution.retry
                )
                if selection.resolved and not (
                    selection.stopped
                    and selection.stopped.trial.execution_id in restartable
                ):
                    break
                attempts = state.for_trial(spec.trial_id)
                await _trial(
                    resolved,
                    spec,
                    job / "executions",
                    client_factory,
                    plan=plan,
                    attempt=max((e.trial.attempt for e in attempts), default=0) + 1,
                    locked=locked,
                    lock_digest=lock_digest,
                )
                state = await asyncio.to_thread(jobs.scan, job, plan)
            if jobs.select(
                state.for_trial(spec.trial_id), plan.execution.retry
            ).stopped:
                break
    except asyncio.CancelledError:
        interrupted = True
        raise
    except Exception as exc:
        diagnostic = EvaluationDiagnostic(
            phase="runner",
            code="runner_failed",
            message=f"job runner failed ({type(exc).__name__})",
            error_type=type(exc).__name__,
        )
    finally:
        state = await asyncio.to_thread(jobs.scan, job, plan)
        report = build_job_report(
            plan,
            state,
            locked=locked,
            lock_digest=lock_digest,
            plan_digest=value_digest(identity_body(plan)),
            interrupted=interrupted,
        )
        if diagnostic:
            report.update(
                complete=False, diagnostics=[diagnostic.model_dump(mode="json")]
            )
        try:
            await finish_io(write_json, job / "report.json", report, replace=True)
        except Exception as exc:
            if not interrupted:
                raise
            _LOG.warning(
                "Cancelled evaluation report persistence failed (%s)",
                type(exc).__name__,
            )
    return report


def _new_job(output: Path, on_job: Callable[[Path], None] | None) -> Path:
    job = output.expanduser().absolute() / ("job-" + uuid4().hex)
    with directory(job, create=True):
        pass
    if on_job:
        on_job(job)
    return job


async def run_suite(
    path: Path,
    output: Path,
    *,
    client_factory: ClientFactory | None = None,
    on_job: Callable[[Path], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    job = _new_job(output, on_job)
    executing = False
    planned = 0
    interrupted = False
    diagnostic: EvaluationDiagnostic | None = None
    try:
        declared = await asyncio.to_thread(load_yaml, path.absolute(), Suite)
        planned = len(declared.source.cases) * declared.execution.repetitions
        resolved = await asyncio.to_thread(
            resolve_suite, path.absolute(), require_credentials=client_factory is None
        )
        plan = await build_identity(resolved, path.absolute(), job)
        write_json(
            job / "job.json",
            {
                "schema_version": "lingcore.eval-job.v2",
                "mode": "unlocked_native",
                "locked": False,
                "lock_digest": None,
                "plan_digest": value_digest(identity_body(plan)),
            },
        )
        with job_lock(job):
            _write_plan(job, plan, locked=False, lock_digest=None)
            _write_original_grades(job, resolved, plan)
            executing = True
            report = await _execute_job(
                job,
                resolved,
                plan,
                locked=False,
                lock_digest=None,
                client_factory=client_factory,
            )
            return job, report
    except asyncio.CancelledError:
        interrupted = True
        raise
    except EvaluationConfigError as exc:
        diagnostic = exc.diagnostic
    except Exception as exc:
        diagnostic = EvaluationDiagnostic(
            phase="preflight",
            code="evaluation_failed",
            message=f"evaluation failed ({type(exc).__name__})",
            error_type=type(exc).__name__,
        )
    finally:
        if not executing:
            report = build_report(
                [], planned, configuration_error=diagnostic is not None
            )
            report["interrupted"] = interrupted
            report["complete"] = False
            if diagnostic:
                report.update(
                    configuration_error=diagnostic.message,
                    diagnostics=[diagnostic.model_dump(mode="json")],
                )
            if not (job / "job.json").exists():
                write_json(
                    job / "job.json",
                    {
                        "schema_version": "lingcore.eval-job.v2",
                        "mode": "unlocked_native",
                        "locked": False,
                    },
                )
            write_json(job / "report.json", report, replace=True)
    return job, report


async def _checked_lock(
    path: Path, *, require_credentials: bool
) -> tuple[Lock, bytes, ResolvedSuite]:
    raw = await asyncio.to_thread(read_bytes, path.absolute())
    lock = parse_lock(raw)
    drift = await validate(lock, path)
    if drift:
        raise EvaluationConfigError(
            "locked plan drift: " + ", ".join(d.describe() for d in drift),
            "lock_drift",
        )
    resolved = await asyncio.to_thread(
        resolve_suite,
        path.absolute().parent / lock.suite.path,
        require_credentials=require_credentials,
        allow_locked=True,
    )
    return lock, raw, resolved


async def run_locked(
    lock_path: Path,
    output: Path,
    *,
    client_factory: ClientFactory | None = None,
    on_job: Callable[[Path], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    lock, raw, resolved = await _checked_lock(
        lock_path, require_credentials=client_factory is None
    )
    job = _new_job(output, on_job)
    write_json(
        job / "job.json",
        {
            "schema_version": "lingcore.eval-job.v2",
            "mode": "locked_native",
            "locked": True,
            "lock_digest": lock.digest,
            "plan_digest": lock.plan_digest,
            "lock_location": str(lock_path.absolute()),
        },
    )
    write_bytes(job / "lock.json", raw)
    with job_lock(job):
        _write_plan(job, lock, locked=True, lock_digest=lock.digest)
        _write_original_grades(job, resolved, lock)
        report = await _execute_job(
            job,
            resolved,
            lock,
            locked=True,
            lock_digest=lock.digest,
            client_factory=client_factory,
        )
    return job, report


async def resume(
    job: Path,
    *,
    lock_path: Path | None = None,
    client_factory: ClientFactory | None = None,
) -> dict[str, Any]:
    job = job.absolute()
    with job_lock(job):
        meta = read_json(job / "job.json")
        if meta.get("schema_version") != "lingcore.eval-job.v2":
            raise EvaluationConfigError(
                "v1 jobs cannot be resumed; resolve a lock", "legacy_job"
            )
        if not meta.get("locked"):
            raise EvaluationConfigError(
                "unlocked jobs cannot be resumed; resolve a lock", "unlocked_job"
            )
        saved = parse_lock(read_bytes(job / "lock.json"))
        location = lock_path or Path(meta["lock_location"])
        supplied, _, resolved = await _checked_lock(
            location, require_credentials=client_factory is None
        )
        if supplied.digest != saved.digest or saved.digest != meta["lock_digest"]:
            raise EvaluationConfigError(
                "replacement lock does not match the job lock digest", "lock_mismatch"
            )
        _, provenance = _plan_record(job)
        if provenance["plan_digest"] != saved.plan_digest:
            raise EvaluationConfigError(
                "job plan does not match its lock", "invalid_job"
            )
        state = await asyncio.to_thread(jobs.scan, job, saved)
        for issue in state.issues:
            if issue.code == "unreadable_trial":
                await finish_io(jobs.quarantine, job, issue)
        for record in state.executions:
            if record.state != "finished":
                await finish_io(jobs.mark_interrupted, record)
        return await _execute_job(
            job,
            resolved,
            saved,
            locked=True,
            lock_digest=saved.digest,
            client_factory=client_factory,
            resume_stopped=True,
        )


def evaluation_command(argv: list[str]) -> int:
    from lingcore.evaluation.cli import main

    return main(argv)
