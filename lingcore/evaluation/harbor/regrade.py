"""Reverify preserved Harbor artifacts without constructing an Agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from lingcore.config import AgentProfile
from lingcore.evaluation import jobs
from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.harbor.outcomes import map_outcome
from lingcore.evaluation.harbor.process import (
    bridge_environment,
    find_bridge,
    invoke,
    invoke_cancellable,
)
from lingcore.evaluation.harbor.source import task_identity
from lingcore.evaluation.identity import (
    HarborExecutionIdentity,
    HarborGradingIdentity,
    HarborSourceIdentity,
    PlanIdentity,
)
from lingcore.evaluation.lock import load_lock, utc_now
from lingcore.evaluation.models import Grade
from lingcore.evaluation.report import build_job_report
from lingcore.evaluation.storage import finish_io, read_json, write_json
from lingcore.paths import resolve_confined


async def regrade_harbor(
    job: Path,
    plan: PlanIdentity,
    provenance: dict[str, Any],
    *,
    lock_path: Path | None,
    bridge: Path | None,
) -> dict[str, Any]:
    assert (
        isinstance(plan.source, HarborSourceIdentity)
        and isinstance(plan.grading, HarborGradingIdentity)
        and isinstance(plan.execution, HarborExecutionIdentity)
    )
    multipliers = plan.execution.harness["timeout_multipliers"]
    if any(multipliers[name] != 1 for name in ("verifier", "setup", "build")):
        raise EvaluationConfigError(
            "Harbor regrade cannot preserve non-default verifier, setup, or build timeout multipliers",
            "regrade_timeout_multipliers",
        )
    meta = read_json(job / "job.json")
    saved = load_lock(job / "lock.json")
    location = lock_path or Path(meta["lock_location"])
    if (
        load_lock(location).digest != saved.digest
        or saved.digest != meta["lock_digest"]
    ):
        raise EvaluationConfigError(
            "replacement lock does not match the job", "lock_mismatch"
        )
    dataset = location.absolute().parent / plan.source.dataset["path"]
    executable = find_bridge(bridge)
    # Regrading has no profile/model requirement and never loads .env values.
    dummy = AgentProfile.model_validate({"llm": {"model": "offline-regrade"}})
    env = bridge_environment(dummy, plan.execution.pass_env)
    for credential in plan.candidate.credentials:
        env.pop(credential.name, None)
    state = await asyncio.to_thread(jobs.scan, job, plan)
    finished = [e for e in state.executions if e.result is not None]
    eligible = {}
    for trial in plan.trials:
        selected = jobs.select(
            state.for_trial(trial.trial_id), plan.execution.retry
        ).selected
        if selected and selected.result:
            original_grade = selected.result.grade
            if original_grade.status == "valid" or (
                original_grade.status == "invalid"
                and original_grade.reason in {"verifier_failed", "reward_invalid"}
            ):
                eligible[selected.trial.execution_id] = selected
    cases = {r.trial.case_id for r in eligible.values()}
    description = await asyncio.to_thread(
        invoke, executable, ["describe", "--dataset", str(dataset)], env
    )
    by_id = {t["id"]: t for t in description["tasks"]}
    # Excluded attempts retain their original grades and verifier identity.
    # Grade sets must describe every planned case, including ungradable ones.
    current = {task.id: task for task in plan.source.tasks}
    for task in plan.source.tasks:
        if task.id not in cases:
            continue
        if task.id not in by_id or by_id[task.id]["status"] != "ok":
            raise EvaluationConfigError(
                f"task {task.id}: metadata is unreadable", "task_metadata"
            )
        identity = task_identity(by_id[task.id], dataset, ())
        if identity.input_digest != task.input_digest:
            raise EvaluationConfigError(
                f"task {task.id}: agent input differs from the job", "input_drift"
            )
        if (
            task.verifier["mode"] != "separate"
            or identity.verifier["mode"] != "separate"
        ):
            raise EvaluationConfigError(
                f"task {task.id}: regrade needs a separate verifier", "shared_verifier"
            )
        current[task.id] = identity
    sources = {}
    skipped: dict[str, str] = {}
    for record in eligible.values():
        assert record.result is not None
        if not record.result.harbor_trial:
            skipped[record.trial.execution_id] = "missing_evidence"
            continue
        source = resolve_confined(record.directory, record.result.harbor_trial)
        if not (source / "result.json").is_file():
            skipped[record.trial.execution_id] = "missing_evidence"
            continue
        sources[record.trial.execution_id] = source
    grade_id = "regrade-" + uuid4().hex
    grades: dict[str, Grade] = {}
    grade_details: dict[str, dict[str, Any]] = {}
    complete = False
    interrupted = False
    try:
        for record in finished:
            assert record.result is not None
            # Preserve excluded attempts and attempts with missing evidence.
            # A new verifier may repair only a selected, eligible execution.
            execution_id = record.trial.execution_id
            grade = record.result.grade
            details = {
                "rewards": record.result.rewards,
                "reward_valid": record.result.reward_valid,
            }
            if execution_id in sources:
                task = current[record.trial.case_id]
                output = record.directory / "regrades" / grade_id / "harbor"
                summary = await invoke_cancellable(
                    executable,
                    [
                        "regrade",
                        "--source",
                        str(sources[execution_id]),
                        "--task",
                        str(dataset / task.path),
                        "--output",
                        str(output),
                    ],
                    env,
                    timeout=task.verifier["timeout_sec"] + 4200,
                )
                await finish_io(
                    write_json, output.parent / "harbor-summary.json", summary
                )
                if any(
                    e["type"] == "RegradeError" for e in summary.get("exceptions", [])
                ):
                    skipped[execution_id] = "missing_evidence"
                else:
                    _, grade = map_outcome(
                        summary,
                        record.result.execution,
                        plan.grading.reward,
                        hello=True,
                        cleanup_failed=record.result.cleanup_failed,
                    )
                    details = {
                        "rewards": summary["rewards"],
                        "reward_valid": summary["reward_valid"],
                    }
            await finish_io(
                write_json,
                record.directory / "grades" / (grade_id + ".json"),
                {
                    "schema_version": "lingcore.eval-grade.v1",
                    "grade_set": grade_id,
                    "execution_id": execution_id,
                    "grade": grade.model_dump(mode="json"),
                    **details,
                    **(
                        {"regrade_skip_reason": skipped[execution_id]}
                        if execution_id in skipped
                        else {}
                    ),
                },
            )
            grades[execution_id] = grade
            grade_details[execution_id] = details
        complete = True
    except asyncio.CancelledError:
        interrupted = True
        raise
    finally:
        report = build_job_report(
            plan,
            state,
            locked=provenance["locked"],
            lock_digest=provenance["lock_digest"],
            plan_digest=provenance["plan_digest"],
            grade_set=grade_id,
            grades=grades,
            grade_details=grade_details,
            interrupted=interrupted,
        )
        report["complete"] &= complete
        skipped_records = [
            {
                "execution_id": execution_id,
                "case_id": eligible[execution_id].trial.case_id,
                "reason": reason,
            }
            for execution_id, reason in skipped.items()
        ]
        report["regrade_skipped"] = skipped_records
        await finish_io(
            write_json,
            job / "grade-sets" / (grade_id + ".json"),
            {
                "schema_version": "lingcore.eval-grades.v1",
                "id": grade_id,
                "created_at": utc_now(),
                "complete": complete,
                "harness": {k: v for k, v in description.items() if k != "tasks"},
                "executions": list(grades),
                "skipped": skipped_records,
                "cases": [
                    {"id": t.id, "graders_digest": t.verifier_digest}
                    for t in current.values()
                ],
            },
        )
        await finish_io(write_json, job / "reports" / (grade_id + ".json"), report)
    return report
