"""Nested workflow caps share ancestor accounting without resetting limits."""

from __future__ import annotations

import asyncio

from lingcore.delegation.config import RunLimits
from lingcore.delegation.supervisor import RunHandle, Supervisor


def child_handle(
    supervisor: Supervisor,
    parent: RunHandle,
    limits: RunLimits,
    prefix: str,
    *,
    resume: bool,
) -> RunHandle:
    budget = parent.budget.child(limits.execution())
    if resume:
        nodes = [
            n
            for n in supervisor.journal.nodes(parent.record.id)
            if n.identity.startswith(prefix)
        ]
        budget.agents = len(nodes)
        reports = [n.report for n in nodes if n.report]
        budget.requests = sum(r.usage.requests for r in reports)
        budget.tokens = sum(
            r.usage.input_tokens + r.usage.output_tokens for r in reports
        )
        budget.usage_complete = all(r.usage.complete for r in reports)
    budget.start()
    return RunHandle(
        parent.record,
        budget,
        asyncio.Semaphore(limits.max_concurrency),
        ancestor_slots=(*parent.ancestor_slots, parent.slots),
        cancelled=parent.cancelled,
        task=parent.task,
        source=parent.source,
        frozen_lock=parent.frozen_lock,
        verified_inputs=parent.verified_inputs,
        progress_sink=parent.progress_sink,
    )
