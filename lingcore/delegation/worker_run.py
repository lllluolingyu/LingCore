"""Worker lifecycle: persist first, isolate, run, account, and always reap."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from lingcore.delegation.envelope import node_class
from lingcore.delegation.models import (
    Envelope,
    NodeRecord,
    UsageSummary,
    WorkerReport,
    digest,
    now,
)
from lingcore.delegation.worker import (
    WorkerFrontend,
    effective_options,
    inherited_context,
    make_brief,
    overlay,
)
from lingcore.delegation.workspaces import git
from lingcore.errors import ToolError
from lingcore.execution import run_headless
from lingcore.tools import Tool
from lingcore.usage import DelegatedUsage, TokenUsage

if TYPE_CHECKING:
    from lingcore.agent import Agent
    from lingcore.delegation.supervisor import RunHandle, Supervisor


async def owned_io(fn: Any, *args: Any, **kwargs: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


async def run_worker(
    supervisor: Supervisor,
    handle: RunHandle,
    prompt: str,
    role: str,
    envelope: Envelope,
    *,
    schema: dict[str, Any] | None = None,
    label: str | None = None,
    identity: str | None = None,
    key: str | None = None,
    continued: NodeRecord | None = None,
    extra_tools: tuple[Tool, ...] = (),
    ticket_workspace: Path | None = None,
    confirmation: Any = None,
    model: str | None = None,
    on_started: Any = None,
    on_idle: Any = None,
    on_input: Any = None,
) -> WorkerReport:
    from lingcore.agent import Agent

    if handle.cancelled.is_set():
        raise asyncio.CancelledError
    definition = supervisor.role(role)
    if model is not None:
        if model not in supervisor.profile.delegation.models:
            raise ToolError("unknown worker model alias")
        definition = definition.model_copy(update={"model": model})
    worker_id = continued.worker_id if continued else "w-" + uuid4().hex
    label = label or role
    worker_budget = handle.budget.child(
        supervisor.profile.delegation.limits.narrow(definition.limits).execution()
    )
    session_id = (
        continued.session_id
        if continued
        else (uuid4().hex if supervisor.journal.store else None)
    )
    node = NodeRecord(
        run_id=handle.record.id,
        identity=identity or worker_id,
        key=key or digest({"prompt": prompt, "envelope": envelope.identity()}),
        worker_id=worker_id,
        role=role,
        classification=node_class(envelope),
        envelope=envelope,
        session_id=session_id,
        prompt=prompt,
        schema=schema,
    )
    report = WorkerReport(
        worker_id=worker_id,
        role=role,
        label=label,
        status="failed",
        session_id=session_id,
    )
    child: Agent | None = None
    workspace = supervisor.parent.tool_ctx.workspace
    note = ""
    snapshot_created = False
    async with AsyncExitStack() as stack:
        if envelope.workspace == "shared" and node_class(envelope) == "effecting":
            await stack.enter_async_context(supervisor.writer)
        await stack.enter_async_context(supervisor.slots)
        for slots in handle.ancestor_slots:
            await stack.enter_async_context(slots)
        await stack.enter_async_context(handle.slots)
        handle.budget.reserve_agent()
        # Retain every billed attempt when a retry/resume replaces a logical node.
        for previous in supervisor.journal.nodes(handle.record.id):
            if previous.identity == node.identity:
                previous.identity += "@attempt:" + previous.worker_id
                supervisor.journal.node(previous)
        supervisor.journal.node(node)
        supervisor.account(handle)
        supervisor.journal.save(handle.record)
        supervisor.progress(handle, label, "running")
        try:
            if envelope.workspace == "ticket":
                if ticket_workspace is None:
                    raise ToolError("ticket workspace is host-only")
                workspace = ticket_workspace
            elif envelope.workspace == "scratch":
                workspace = supervisor.workspace.path(
                    continued.run_id if continued else handle.record.id, worker_id
                )
            elif envelope.workspace == "snapshot":
                snapshot_created = True
                workspace, _, skipped = await owned_io(
                    supervisor.workspace.capture, handle.record.id, worker_id
                )
                note = "Skipped files: " + ", ".join(skipped[:100])
                if continued and continued.report:
                    patches = [
                        p for p in continued.report.artifacts if p.endswith(".patch")
                    ]
                    if patches:
                        try:
                            patch = supervisor.workspace.read(patches[-1])
                            await owned_io(
                                git, workspace, "apply", "--3way", "-", data=patch
                            )
                        except ToolError:
                            await owned_io(git, workspace, "reset", "--hard", "HEAD")
                            note += f"\nPrevious patch could not be reapplied; inspect {patches[-1]}."
            elif envelope.workspace == "frozen":
                async with handle.frozen_lock:
                    if handle.record.input_revision is None:
                        workspace, revision, skipped = await owned_io(
                            supervisor.workspace.capture,
                            handle.record.id,
                            "input",
                            frozen=True,
                        )
                        (
                            handle.record.input_revision,
                            handle.record.input_captured_at,
                            handle.record.skipped,
                        ) = revision, now(), skipped
                        supervisor.journal.save(handle.record)
                    else:
                        workspace = supervisor.workspace.path(handle.record.id, "input")
                        if (
                            handle.record.input_revision not in handle.verified_inputs
                            and (
                                await owned_io(
                                    supervisor.workspace.revision,
                                    workspace,
                                    handle.record.skipped,
                                )
                                != handle.record.input_revision
                            )
                        ):
                            raise ToolError(
                                "frozen input is missing or altered; start a new run"
                            )
                    assert handle.record.input_revision is not None
                    handle.verified_inputs.add(handle.record.input_revision)
                note = (
                    "Pinned frozen input. Staged/unstaged split is not reproduced. Skipped: "
                    + ", ".join(handle.record.skipped[:100])
                )
            child_profile = overlay(supervisor.profile, definition, envelope, workspace)
            getattr(child_profile, "_worker_layers").extend(
                inherited_context(supervisor.parent, definition)
            )
            frontend = WorkerFrontend(
                supervisor,
                handle,
                worker_id,
                role,
                make_brief(prompt, handle.record.request, workspace, note),
                schema,
            )
            confirm = confirmation or supervisor.broker.for_worker(
                handle.record.id,
                label,
                handle.cancelled,
                background=handle.record.background,
                deny=envelope.confirm == "deny",
            )
            submit = frontend.tool()
            child = Agent.from_profile(
                child_profile,
                llm=supervisor.llm_factory(role, label)
                if supervisor.llm_factory
                else None,
                tool_options=effective_options(supervisor.parent),
                session_store=supervisor.journal.store,
                session_id=session_id,
                execution_budget=worker_budget,
                environment_binding=supervisor.parent.tool_ctx.environment_binding,
                confirm=confirm,
                extra_tools=(submit, *extra_tools),
                terminal_tools=frozenset({submit.name}),
            )
            child._trace_relation = "delegate"
            if on_idle is not None:
                frontend.on_idle = lambda: on_idle(child)
            frontend.on_input = on_input
            raw = getattr(child.llm, "client", child.llm)
            if child_profile.llm.send_prompt_cache_key and hasattr(
                raw, "_prompt_cache_key"
            ):
                raw._prompt_cache_key = f"{handle.record.id}:{role}"
            if child.tool_ctx.delegation is not None:
                child.tool_ctx.delegation.depth = envelope.depth
                child.tool_ctx.delegation.llm_factory = supervisor.llm_factory
            if session_id:
                supervisor.journal.lineage(
                    session_id,
                    {
                        "parent_session_id": supervisor.parent._session_id,
                        "anchor_seq": handle.record.anchor_seq,
                        "run_id": handle.record.id,
                        "node_key": node.key,
                        "role": role,
                        "envelope": envelope.model_dump(mode="json"),
                    },
                )
            if on_started is not None:
                await on_started(node)
            outcome = await run_headless(child, frontend=frontend, confirm=confirm)
            if outcome.outcome.status == "completed":
                if frontend.output_error:
                    report.error = frontend.output_error
                elif frontend.submission:
                    report.status = (
                        "blocked"
                        if frontend.submission.status == "blocked"
                        else "completed"
                    )
                    report.result, report.summary, report.question = (
                        frontend.submission.result,
                        frontend.submission.summary,
                        frontend.submission.question,
                    )
                    total = 0
                    for index, relative in enumerate(frontend.submission.artifacts):
                        path = Path(relative)
                        name = f"{worker_id}-{index}-{path.name}"
                        total += await owned_io(
                            supervisor.workspace.retain_artifact,
                            workspace,
                            relative,
                            handle.record.id,
                            name,
                            supervisor.workspace.config.snapshot_max_bytes - total,
                        )
                        report.artifacts.append(
                            str(Path(".lingcore/runs") / handle.record.id / name)
                        )
                elif schema is None:
                    report.status, report.result = "completed", frontend.final
                else:
                    report.error = "no_result"
            else:
                report.status = (
                    outcome.outcome.status
                    if outcome.outcome.status
                    in {"failed", "cancelled", "budget_exceeded", "timed_out"}
                    else "failed"
                )
                report.error = (
                    outcome.outcome.provider_error_category
                    or outcome.outcome.limit_type
                    or outcome.outcome.cause
                )
        except asyncio.CancelledError:
            report.status, report.error = "cancelled", "worker cancelled"
            raise
        except Exception as exc:
            report.status, report.error = "failed", str(exc)[:2000]
        finally:
            if child is not None:
                if not child._closed:
                    await child.aclose()
                records = child.request_ledger.records
                nested = child.tool_ctx.delegation
                nested_usage = (
                    [
                        nested.retired_usage,
                        *[h.record.usage for h in nested.runs.values()],
                    ]
                    if nested is not None
                    else []
                )
                report.usage = UsageSummary(
                    input_tokens=sum(u.input_tokens or 0 for u in records)
                    + sum(u.input_tokens for u in nested_usage),
                    output_tokens=sum(u.output_tokens or 0 for u in records)
                    + sum(u.output_tokens for u in nested_usage),
                    requests=worker_budget.requests,
                    complete=worker_budget.usage_complete,
                )
                if not handle.record.background:
                    for delegated in frontend.delegated:
                        supervisor.parent.usage_meter.record_delegated(delegated)
                    for u in records:
                        if u.availability != "missing":
                            supervisor.parent.usage_meter.record_delegated(
                                DelegatedUsage(
                                    handle.record.id,
                                    worker_id,
                                    role,
                                    TokenUsage(
                                        u.model,
                                        u.input_tokens or 0,
                                        u.output_tokens or 0,
                                        u.cached_input_tokens or 0,
                                        u.reasoning_tokens or 0,
                                        u.purpose,
                                    ),
                                )
                            )
            if snapshot_created:
                try:
                    patch = await owned_io(
                        supervisor.workspace.patch, handle.record.id, worker_id
                    )
                    if patch:
                        report.artifacts.append(patch)
                    if not supervisor.profile.delegation.workspace.keep_snapshots:
                        await owned_io(
                            supervisor.workspace.remove, handle.record.id, worker_id
                        )
                        await owned_io(
                            supervisor.workspace.remove,
                            handle.record.id,
                            "base-" + worker_id,
                        )
                except Exception as exc:
                    report.status, report.error = (
                        "failed",
                        f"snapshot finalization failed: {type(exc).__name__}",
                    )
            node.report, node.status = report, report.status
            node.report_digest = digest(report.model_dump(mode="json"))
            for artifact_path in report.artifacts:
                node.artifacts_digest[artifact_path] = await owned_io(
                    supervisor.workspace.artifact_digest, artifact_path
                )
            supervisor.journal.node(node)
            supervisor.account(handle)
            supervisor.journal.save(handle.record)
            supervisor.progress(handle, label, report.status)
    # Headless workers contain their own terminal signal. Run/ancestor limits
    # must still stop orchestration, while role-local failures remain reports.
    handle.budget.check()
    return report
