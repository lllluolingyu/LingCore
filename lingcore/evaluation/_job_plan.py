"""Frozen job plan serialization, validation, and original grade publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lingcore.evaluation.config import EvaluationConfigError, ResolvedSuite
from lingcore.evaluation.identity import PlanIdentity, identity_body, value_digest
from lingcore.evaluation.lock import utc_now
from lingcore.evaluation.storage import read_json, write_json


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
    if plan.source.kind == "harbor":
        write_json(
            job / "grade-sets/original.json",
            {
                "schema_version": "lingcore.eval-grades.v1",
                "id": "original",
                "created_at": utc_now(),
                "cases": [
                    {"id": task.id, "graders_digest": task.verifier_digest}
                    for task in plan.source.tasks
                ],
            },
        )
        return
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
