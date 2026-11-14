"""Admission, confirmation and journal recovery shared by tools and CLI."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from lingcore.delegation.config import RunLimits
from lingcore.delegation.supervisor import RunHandle, Supervisor
from lingcore.errors import ToolError
from lingcore.execution import ExecutionBudget
from lingcore.workflows.catalog import load_catalog, parse_spec
from lingcore.workflows.interpreter import Interpreter
from lingcore.workflows.spec import ValidatedSpec, WorkflowSpec


def validate(
    supervisor: Supervisor, spec: WorkflowSpec, snapshot: frozenset[str]
) -> ValidatedSpec:
    return ValidatedSpec(
        spec,
        supervisor.profile.delegation.limits,
        lambda role, **kwargs: supervisor.envelope(role, snapshot, **kwargs),
        load_catalog(supervisor.profile),
        models=frozenset(supervisor.profile.delegation.models),
    )


async def run_workflow(
    supervisor: Supervisor,
    snapshot: frozenset[str],
    *,
    name: str | None = None,
    spec: str | dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
    background: bool = False,
) -> Any:
    if (name is None) == (spec is None):
        raise ToolError("supply exactly one workflow name or spec")
    if spec is not None and not supervisor.profile.workflows.inline:
        raise ToolError("inline workflows are disabled")
    catalog = load_catalog(supervisor.profile)
    if name is not None and name not in catalog:
        from lingcore.workflows.api import plugin_workflows, run_python

        python_catalog = plugin_workflows(supervisor.profile)
        if name not in python_catalog:
            raise ToolError("unknown workflow")
        return await run_python(
            supervisor,
            snapshot,
            python_catalog[name],
            args or {},
            background=background,
        )
    parsed = catalog[name] if name is not None else parse_spec(spec)  # type: ignore[arg-type]
    validated = validate(supervisor, parsed, snapshot)
    handle = supervisor.new_run(
        kind="workflow", name=parsed.name, background=background, limits=parsed.limits
    )
    handle.record.spec = parsed.model_dump()
    handle.record.args = args or {}
    supervisor.journal.save(handle.record)
    await supervisor.confirm_run(handle, validated.roles, validated.bound)
    interpreter = Interpreter(supervisor, handle, validated, snapshot)
    return await supervisor.execute(handle, lambda: interpreter.run(args or {}))


async def resume_workflow(
    supervisor: Supervisor, run_id: str, snapshot: frozenset[str]
) -> Any:
    if supervisor.journal.store is None:
        raise ToolError("resume requires persistent sessions")
    record = supervisor.owned_run(run_id)
    live = supervisor.runs.get(run_id)
    if live and live.task and not live.task.done():
        raise ToolError("run is still active")
    if record.kind != "workflow":
        raise ToolError("use continue_worker for a delegate run")
    if record.spec is None:
        from lingcore.workflows.api import resume_python

        return await resume_python(supervisor, record, snapshot)
    # Named definitions may change; dependencies and node semantics decide reuse,
    # never the digest of the whole workflow.
    catalog = load_catalog(supervisor.profile)
    spec = catalog.get(record.name) or WorkflowSpec.model_validate(record.spec)
    validated = validate(supervisor, spec, snapshot)
    record.spec = spec.model_dump()
    handle = resume_handle(supervisor, record, validated.limits)
    interpreter = Interpreter(supervisor, handle, validated, snapshot, resume=True)
    await interpreter.prepare()
    nodes = [
        n for n in supervisor.journal.nodes(record.id) if "@attempt:" not in n.identity
    ]
    handle.record.progress["resume_preview"] = {
        "previous_live_readers": sum(n.classification == "live" for n in nodes),
        "cache_candidates": sum(
            n.classification in {"pure", "frozen"} and n.status == "completed"
            for n in nodes
        ),
        "live_worker_upper_bound": validated.bound,
        "note": "Live readers rerun when reached; cache candidates are revalidated. Changed dependency values can require more live workers. Effecting nodes never rerun.",
    }
    await supervisor.confirm_run(
        handle, validated.roles, validated.bound, force=True, preserve_status=True
    )
    supervisor.journal.revive(handle.record)
    return await supervisor.execute(handle, lambda: interpreter.run(record.args))


def resume_handle(supervisor: Supervisor, record: Any, limits: RunLimits) -> RunHandle:
    saved = record.limits
    effective = limits.model_dump()
    for key, old in saved.items():
        if key == "usage_missing":
            effective[key] = "stop" if "stop" in {old, effective[key]} else "ignore"
        elif old is not None:
            effective[key] = (
                min(old, effective[key]) if effective[key] is not None else old
            )
    limits = RunLimits.model_validate(effective)
    budget = ExecutionBudget(
        limits.execution(), parent=supervisor.parent.execution_budget
    )
    budget.agents, budget.requests, budget.tools = (
        record.agents,
        record.usage.requests,
        record.tool_calls,
    )
    budget.tokens = record.usage.input_tokens + record.usage.output_tokens
    budget.usage_complete = record.usage.complete
    budget.start()
    if limits.wall_seconds is not None:
        budget.deadline = time.monotonic() + max(
            0, limits.wall_seconds - record.wall_seconds_spent
        )
    record.background = False
    record.limits = effective
    record.started_at = supervisor.parent._turn_started_at
    handle = RunHandle(
        record,
        budget,
        asyncio.Semaphore(limits.max_concurrency),
        progress_sink=supervisor.parent.tool_ctx.progress,
    )
    supervisor.runs[record.id] = handle
    return handle
