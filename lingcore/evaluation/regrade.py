"""New grade sets over preserved evidence; no agent or tool is invoked."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from lingcore.config import AgentProfile, _load_profile_env
from lingcore.evaluation import evidence, jobs
from lingcore.evaluation.config import EvaluationConfigError, load_yaml
from lingcore.evaluation.graders import grade
from lingcore.evaluation.identity import PlanIdentity, assert_secret_free, case_identity
from lingcore.evaluation.lock import load_lock, utc_now
from lingcore.evaluation.models import Case, Grade, Suite
from lingcore.evaluation.report import build_job_report
from lingcore.evaluation.runner import _plan_record
from lingcore.evaluation.storage import (
    canonical_json,
    directory,
    job_lock,
    read_json,
    write_json,
)
from lingcore.tracing.config import known_secrets

_LOG = logging.getLogger(__name__)


def current_cases(
    job: Path, plan: PlanIdentity, *, lock_path: Path | None = None
) -> tuple[list[Case], tuple[str, ...]]:
    meta = read_json(job / "job.json")
    base = job
    if meta.get("locked"):
        saved = load_lock(job / "lock.json")
        if saved.plan_digest != meta["plan_digest"]:
            raise EvaluationConfigError(
                "job plan does not match its lock", "invalid_job"
            )
        location = lock_path or Path(meta["lock_location"])
        supplied = load_lock(location)
        if supplied.digest != saved.digest or saved.digest != meta["lock_digest"]:
            raise EvaluationConfigError(
                "replacement lock does not match the job lock digest", "lock_mismatch"
            )
        base = location.absolute().parent
    elif lock_path is not None:
        raise EvaluationConfigError(
            "an unlocked job has no lock to relocate", "unlocked_job"
        )
    suite_path = base / plan.suite.path
    suite: Suite = load_yaml(suite_path, Suite)
    cases = [load_yaml(suite_path.parent / path, Case) for path in suite.source.cases]
    original = {c.id: c for c in plan.source.cases}
    if len({c.id for c in cases}) != len(cases) or set(original) != {
        c.id for c in cases
    }:
        raise EvaluationConfigError(
            "case membership differs from the job inputs", "input_drift"
        )
    # Read only the exact profile-local env file, without expanding/loading the
    # current profile or requiring its model connection settings.
    scoped = _load_profile_env(base / plan.candidate.profile.path)
    dummy = AgentProfile.model_validate({"llm": {"model": "offline-regrade"}})
    object.__setattr__(dummy, "_profile_env", scoped)
    environment = {**os.environ, **scoped}
    secrets = tuple(
        set(known_secrets(dummy))
        | {
            environment[c.name]
            for c in plan.candidate.credentials
            if environment.get(c.name)
        }
    )
    for case in cases:
        assert_secret_free(canonical_json(case.model_dump(mode="json")), secrets)
        if case_identity(case).input_digest != original[case.id].input_digest:
            raise EvaluationConfigError(
                f"case {case.id}: agent input differs from the job", "input_drift"
            )
    return cases, secrets


async def regrade(job: Path, *, lock_path: Path | None = None) -> dict[str, Any]:
    job = job.absolute()
    with job_lock(job):
        plan, provenance = _plan_record(job)
        cases, _ = await asyncio.to_thread(
            current_cases, job, plan, lock_path=lock_path
        )
        by_id = {case.id: case for case in cases}
        state = await asyncio.to_thread(jobs.scan, job, plan)
        finished = [e for e in state.executions if e.result is not None]
        # Check the whole pass before writing any grades. Incomplete capture is
        # evaluable as not_evaluable; an undeclared artifact cannot be recovered.
        for record in finished:
            manifest = evidence.load_manifest(record.directory)
            evidence.check_needs(by_id[record.trial.case_id], manifest)
        grade_id = "regrade-" + uuid4().hex
        created_at = utc_now()
        grades: dict[str, Grade] = {}
        interrupted = False
        failure = False
        try:
            for record in finished:
                case = by_id[record.trial.case_id]
                collected = await asyncio.to_thread(
                    evidence.load,
                    record.directory,
                    case,
                    artifact_max_bytes=plan.trace.artifact_max_bytes,
                )
                result = await grade(case.graders, collected)
                if record.result is not None and record.result.cleanup_failed:
                    result = result.model_copy(
                        update={
                            "status": "invalid",
                            "passed": None,
                            "reason": "cleanup_failed",
                        }
                    )
                write_json(
                    record.directory / "grades" / (grade_id + ".json"),
                    {
                        "schema_version": "lingcore.eval-grade.v1",
                        "grade_set": grade_id,
                        "execution_id": record.trial.execution_id,
                        "grade": result.model_dump(mode="json"),
                    },
                )
                grades[record.trial.execution_id] = result
        except asyncio.CancelledError:
            interrupted = True
            raise
        except Exception:
            failure = True
        finally:
            header = {
                "schema_version": "lingcore.eval-grades.v1",
                "id": grade_id,
                "created_at": created_at,
                "complete": not interrupted and not failure,
                "executions": list(grades),
                "cases": [
                    {
                        "id": c.id,
                        "graders_digest": case_identity(c).graders_digest,
                        "graders": [
                            g.model_dump(mode="json") for g in case_identity(c).graders
                        ],
                        "specs": [g.model_dump(mode="json") for g in c.graders],
                    }
                    for c in cases
                ],
            }
            report = build_job_report(
                plan,
                state,
                locked=provenance["locked"],
                lock_digest=provenance["lock_digest"],
                plan_digest=provenance["plan_digest"],
                grade_set=grade_id,
                grades=grades,
                interrupted=interrupted,
            )
            if failure:
                report.update(complete=False, grading_error="grade_set_incomplete")
            try:
                write_json(job / "grade-sets" / (grade_id + ".json"), header)
                write_json(job / "reports" / (grade_id + ".json"), report)
            except Exception as exc:
                if not interrupted:
                    raise
                _LOG.warning(
                    "Interrupted regrade persistence failed (%s)", type(exc).__name__
                )
        return report


def read_report(job: Path, grades: str = "original") -> dict[str, Any]:
    def report(path: Path) -> dict[str, Any]:
        value = read_json(path)
        if not isinstance(value, dict) or value.get("schema_version") not in {
            "lingcore.eval-report.v1",
            "lingcore.eval-report.v2",
        }:
            raise EvaluationConfigError(
                "invalid or unsupported report", "invalid_report"
            )
        return value

    if grades == "original":
        return report(job / "report.json")
    if grades == "latest":
        headers = []
        if not (job / "grade-sets").exists():
            return report(job / "report.json")
        with directory(job / "grade-sets") as root:
            for name, _, is_file, is_link in root.iter_entries():
                if (
                    is_file
                    and not is_link
                    and name != "original.json"
                    and name.endswith(".json")
                ):
                    headers.append(read_json(job / "grade-sets" / name))
        if not headers:
            return report(job / "report.json")
        grades = max(headers, key=lambda h: (h["created_at"], h["id"]))["id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", grades):
        raise EvaluationConfigError("invalid grade set identifier", "invalid_grades")
    return report(job / "reports" / (grades + ".json"))
