"""Run ownership, shared admission, broker, cancellation and retained workers."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from lingcore._sessions.runs import RunJournal
from lingcore.background import EventSource
from lingcore.delegation.broker import ConfirmationBroker
from lingcore.delegation.envelope import compute
from lingcore.delegation.models import (
    Envelope,
    RunRecord,
    UsageSummary,
    WorkerReport,
    WorkspaceMode,
)
from lingcore.delegation.roles import load_roles
from lingcore.delegation.workspaces import WorkspaceManager
from lingcore.errors import BudgetExceeded, ToolError
from lingcore.execution import ExecutionBudget

if TYPE_CHECKING:
    from lingcore.agent import Agent
    from lingcore.config import AgentProfile


@dataclass
class RunHandle:
    record: RunRecord
    budget: ExecutionBudget
    slots: asyncio.Semaphore
    ancestor_slots: tuple[asyncio.Semaphore, ...] = ()
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[Any] | None = None
    source: EventSource | None = None
    frozen_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    verified_inputs: set[str] = field(default_factory=set)
    progress_sink: Callable[[dict[str, Any]], None] | None = None


class Supervisor:
    def __init__(
        self,
        parent: Agent,
        profile: AgentProfile,
        *,
        depth: int = 0,
        llm_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.parent, self.profile, self.depth = parent, profile, depth
        self.roles = load_roles(profile)
        self.llm_factory = llm_factory
        self.journal = RunJournal(parent._session_store)
        self.workspace = WorkspaceManager(
            parent.tool_ctx.workspace,
            profile.delegation.workspace,
            excludes=set(
                profile.tool_options.get("search", {}).get("exclude_dirs", [])
            ),
        )
        self.slots = asyncio.Semaphore(profile.delegation.limits.max_concurrency)
        self.writer = asyncio.Lock()
        self.broker = ConfirmationBroker(
            parent.tool_ctx.confirm,
            async_confirm=parent.tool_ctx.supports_async_confirm,
            current=lambda: parent.tool_ctx.confirm,
        )
        self.runs: dict[str, RunHandle] = {}
        self.retired_usage = UsageSummary()
        self._turn = 0.0
        self._turn_workers = 0
        self._closed = False
        # Attaching the parent already acquired its session lease. Only that
        # parent's orphaned records can now be recovered safely.
        if parent._session_id is not None:
            for record in self.journal.list(parent._session_id):
                if record.status in {"pending", "running"}:
                    record.status, record.reason = "interrupted", "host restarted"
                    self.journal.save(record)
                    for node in self.journal.nodes(record.id):
                        if node.status == "running":
                            node.status = "interrupted"
                            self.journal.node(node)

        from lingcore.delegation.retention import prune

        prune(self.journal, self.workspace, profile.delegation.retention)

    def role(self, name: str) -> Any:
        try:
            return self.roles[name]
        except KeyError:
            raise ToolError(f"unknown worker role: {name}") from None

    def envelope(
        self,
        role: str,
        snapshot: frozenset[str],
        *,
        tools: list[str] | None = None,
        workspace: WorkspaceMode | None = None,
        recorded: Envelope | None = None,
    ) -> Envelope:
        return compute(
            self.parent,
            self.profile,
            self.role(role),
            snapshot,
            depth=self.depth + 1,
            tools=tools,
            workspace=workspace,
            recorded=recorded,
        )

    def owned_run(self, run_id: str, *, read_only: bool = False) -> RunRecord:
        run = self.journal.get(run_id)
        if run.parent_session_id != self.parent._session_id:
            if read_only and self.journal.store:
                sid = self.parent._session_id
                visited = set()
                boundary: int | None = None
                while sid and sid not in visited:
                    visited.add(sid)
                    meta = self.journal.store.get(sid)
                    if meta is None or meta.fork is None:
                        break
                    boundary = (
                        min(boundary, meta.fork.through_seq)
                        if boundary is not None
                        else meta.fork.through_seq
                    )
                    sid = meta.fork.parent_session_id
                    if sid == run.parent_session_id and run.anchor_seq <= boundary:
                        return run
            raise ToolError("run belongs to another session")
        return run

    def new_run(
        self,
        *,
        kind: Literal["delegate", "workflow", "ticket"] = "delegate",
        name: str = "",
        background: bool = False,
        limits: dict[str, Any] | None = None,
        request: str | None = None,
    ) -> RunHandle:
        if self._closed:
            raise ToolError("delegation supervisor is closed")
        if background and not (
            self.profile.delegation.background and self.parent.background.enabled
        ):
            raise ToolError("background delegation needs a consenting frontend")
        if self.depth >= self.profile.delegation.max_depth:
            raise ToolError("delegation depth limit reached")
        if kind == "delegate":
            if self._turn != self.parent._turn_started_at:
                self._turn, self._turn_workers = self.parent._turn_started_at, 0
            if self._turn_workers >= self.profile.delegation.max_workers_per_turn:
                raise ToolError("max_workers_per_turn reached")
            self._turn_workers += 1
        effective = self.profile.delegation.limits.narrow(limits or {})
        user = self.parent._turn_user_message
        original = (
            request
            if request is not None
            else ((user.input_text or user.content) if user else "")
        )
        anchor = getattr(self.parent.memory, "last_sequence", None)
        record = RunRecord(
            id="r-" + uuid4().hex,
            parent_session_id=self.parent._session_id,
            workspace=str(self.parent.tool_ctx.workspace.resolve()),
            anchor_seq=anchor if kind != "ticket" and anchor is not None else -1,
            kind=kind,
            started_at=0 if kind == "ticket" else self.parent._turn_started_at,
            background=background,
            name=name,
            limits=effective.model_dump(),
            request=original,
        )
        handle = RunHandle(
            record,
            ExecutionBudget(effective.execution(), parent=self.parent.execution_budget),
            asyncio.Semaphore(effective.max_concurrency),
        )
        handle.budget.start()
        handle.progress_sink = self.parent.tool_ctx.progress
        self.journal.save(record)
        self.runs[record.id] = handle
        self.compact_history()
        return handle

    def compact_history(self) -> None:
        for run_id, handle in list(self.runs.items()):
            if len(self.runs) <= 128:
                break
            if (
                handle.record.status in {"pending", "running"}
                or handle.task
                and not handle.task.done()
            ):
                continue
            usage = handle.record.usage
            self.retired_usage.input_tokens += usage.input_tokens
            self.retired_usage.output_tokens += usage.output_tokens
            self.retired_usage.requests += usage.requests
            self.retired_usage.complete &= usage.complete
            del self.runs[run_id]

    async def confirm_run(
        self,
        handle: RunHandle,
        envelopes: dict[str, Envelope],
        bound: int,
        *,
        force: bool = False,
        preserve_status: bool = False,
    ) -> None:
        if not (
            force
            or bound > self.profile.delegation.confirm_over_agents
            or any(e.high_risk for e in envelopes.values())
        ):
            return
        details = {
            name: {"tools": sorted(e.tools), "workspace": e.workspace}
            for name, e in envelopes.items()
        }
        prompt = (
            f"Run {handle.record.name or handle.record.kind}: at most {bound} agents, "
            f"concurrency {handle.record.limits['max_concurrency']}, "
            f"reported-token stopping threshold {handle.record.limits['reported_tokens']} "
            "(in-flight usage may overshoot), "
            + ("background" if handle.record.background else "foreground")
            + ".\n"
            + json.dumps(details)
        )
        if "resume_preview" in handle.record.progress:
            prompt += "\nResume: " + json.dumps(
                handle.record.progress["resume_preview"]
            )
        confirm = self.parent.tool_ctx.confirm
        if confirm is None or not await confirm(prompt):
            if not preserve_status:
                handle.record.status, handle.record.reason = (
                    "cancelled",
                    "run confirmation denied",
                )
                self.journal.save(handle.record)
            raise ToolError("run confirmation denied")

    async def execute(
        self, handle: RunHandle, operation: Callable[[], Awaitable[Any]]
    ) -> Any:
        async def drive() -> Any:
            started = time.monotonic()
            owner = asyncio.current_task()

            async def watch_anchor() -> None:
                while True:
                    await asyncio.sleep(0.25)
                    persisted = self.journal.get(handle.record.id)
                    if persisted.status == "detached":
                        handle.record.status, handle.record.reason = (
                            "detached",
                            persisted.reason,
                        )
                        handle.cancelled.set()
                        if handle.source:
                            handle.source.cancel()
                        if owner:
                            owner.cancel()
                        return

            watcher = asyncio.create_task(watch_anchor())
            handle.record.status = "running"
            self.journal.save(handle.record)
            try:
                remaining = (
                    None
                    if handle.budget.deadline is None
                    else max(0, handle.budget.deadline - time.monotonic())
                )
                async with asyncio.timeout(remaining):
                    result = await operation()
                handle.record.result = (
                    result.model_dump(mode="json")
                    if isinstance(result, WorkerReport)
                    else result
                )
                if handle.record.status == "running":
                    handle.record.status = (
                        result.status
                        if isinstance(result, WorkerReport)
                        else "completed"
                    )
                return result
            except TimeoutError:
                handle.cancelled.set()
                handle.record.status, handle.record.reason = "timed_out", "wall_seconds"
                return {"run_id": handle.record.id, "status": "timed_out"}
            except asyncio.CancelledError:
                handle.cancelled.set()
                if handle.record.status not in {"detached", "interrupted"}:
                    handle.record.status = "detached"
                raise
            except BudgetExceeded as exc:
                handle.record.status, handle.record.reason = (
                    "budget_exceeded",
                    exc.limit,
                )
                return {
                    "run_id": handle.record.id,
                    "status": "budget_exceeded",
                    "limit": exc.limit,
                }
            except Exception as exc:
                handle.record.status, handle.record.reason = "failed", str(exc)[:2000]
                raise
            finally:
                handle.record.wall_seconds_spent += time.monotonic() - started
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                self.account(handle)
                self.journal.save(handle.record)
                if handle.source and handle.record.status not in {
                    "detached",
                    "interrupted",
                    "cancelled",
                }:
                    handle.source.publish(
                        json.dumps(
                            {
                                "run": handle.record.id,
                                "workflow": handle.record.name,
                                "status": handle.record.status,
                                "summary": str(handle.record.result)[:600],
                                "usage": handle.record.usage.model_dump(),
                                "read": f"workflow_result {handle.record.id}",
                            }
                        )
                    )
                if handle.source:
                    handle.source.finish()

        if handle.record.background:
            handle.source = EventSource(
                self.parent.background, handle.record.id, kind="run"
            )

            # Background exceptions are retained on the run, never orphan tasks.
            async def detached() -> None:
                try:
                    await drive()
                except (Exception, asyncio.CancelledError):
                    pass

            handle.task = asyncio.create_task(detached())
            return {"run_id": handle.record.id, "status": "started"}
        handle.task = asyncio.create_task(drive())
        try:
            return await handle.task
        finally:
            if not handle.task.done():
                handle.task.cancel()
                await asyncio.gather(handle.task, return_exceptions=True)

    def account(self, handle: RunHandle) -> None:
        budget = self.runs.get(handle.record.id, handle).budget
        handle.record.agents = budget.agents
        handle.record.tool_calls = budget.tools
        handle.record.usage.requests = budget.requests
        handle.record.usage.complete = budget.usage_complete
        # Request ledgers retain input/output separately; the budget total is a
        # backstop for interrupted requests whose worker report is incomplete.
        reports = [n.report for n in self.journal.nodes(handle.record.id) if n.report]
        handle.record.usage.input_tokens = sum(r.usage.input_tokens for r in reports)
        handle.record.usage.output_tokens = sum(r.usage.output_tokens for r in reports)

    def progress(
        self, handle: RunHandle, label: str, status: str, phase: str = ""
    ) -> None:
        nodes = self.journal.nodes(handle.record.id)
        handle.record.progress = {
            "run_id": handle.record.id,
            "phase": phase or handle.record.progress.get("phase", ""),
            "latest": label,
            "status": status,
            "done": sum(n.status != "running" for n in nodes),
            "total": handle.record.agents,
            "failed": sum(n.status in {"failed", "blocked"} for n in nodes),
            "reported_tokens": handle.budget.tokens,
            "remaining_tokens": handle.budget.remaining_tokens,
            "usage_complete": handle.budget.usage_complete,
            "nodes": [
                {"id": n.identity, "role": n.role, "status": n.status}
                for n in nodes
                if "@attempt:" not in n.identity
            ][-24:],
        }
        if handle.progress_sink:
            handle.progress_sink(handle.record.progress)
        self.journal.save(handle.record)

    async def worker(
        self,
        handle: RunHandle,
        prompt: str,
        role: str,
        envelope: Envelope,
        **kwargs: Any,
    ) -> WorkerReport:
        from lingcore.delegation.worker_run import run_worker

        return await run_worker(self, handle, prompt, role, envelope, **kwargs)

    async def stop(
        self, run_id: str, *, reason: str = "stopped", status: str = "detached"
    ) -> None:
        record = self.owned_run(run_id)
        handle = self.runs.get(run_id)
        if handle:
            handle.record.status, handle.record.reason = status, reason
            handle.cancelled.set()
            if handle.source:
                handle.source.cancel()
            if handle.task and not handle.task.done():
                handle.task.cancel()
                await asyncio.gather(handle.task, return_exceptions=True)
            record = handle.record
        record.status, record.reason = status, reason
        self.journal.save(record)

    def rollback(self, started_at: float) -> None:
        for handle in self.runs.values():
            if (
                handle.record.kind != "ticket"
                and handle.record.started_at >= started_at
            ):
                handle.cancelled.set()
                if handle.source:
                    handle.source.cancel()
                handle.record.status, handle.record.reason = (
                    "detached",
                    "starting turn rolled back",
                )
                if handle.task and not handle.task.done():
                    handle.task.cancel()
                self.journal.save(handle.record)

    async def aclose(self) -> None:
        self._closed = True
        for run_id, handle in self.runs.items():
            if handle.task and not handle.task.done():
                status = (
                    "detached" if handle.record.status == "detached" else "interrupted"
                )
                await self.stop(run_id, reason="agent closed", status=status)

    async def continue_worker(
        self,
        worker_id: str,
        prompt: str,
        snapshot: frozenset[str],
        *,
        background: bool = False,
    ) -> Any:
        if self.journal.store is None:
            raise ToolError("worker continuation requires persistent sessions")
        old = self.journal.worker(worker_id)
        self.owned_run(old.run_id)
        if old.status == "running":
            raise ToolError("worker is still running")
        envelope = self.envelope(old.role, snapshot, recorded=old.envelope)
        handle = self.new_run(name=f"continue:{worker_id}", background=background)
        await self.confirm_run(handle, {old.role: envelope}, 1)
        return await self.execute(
            handle,
            lambda: self.worker(
                handle, prompt, old.role, envelope, schema=old.schema_, continued=old
            ),
        )
