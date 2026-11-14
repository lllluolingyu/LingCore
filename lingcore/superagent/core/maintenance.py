"""Bounded recovery and housekeeping commands for the pure domain engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lingcore.superagent.core.engine import RECOVERY_NOTE

if TYPE_CHECKING:
    from lingcore.superagent.core.engine import Engine


def tick(self: Engine, now: float) -> None:
    for ticket in self.state.tickets.values():
        for assignment in list(ticket.assignments):
            if (
                assignment.status == "pending"
                and assignment.awaiting_approval
                and now - assignment.created_at
                >= self.config.executor.approval_timeout_seconds
            ):
                self.finish(
                    ticket,
                    assignment,
                    "blocked",
                    "Extra-tool approval expired.",
                    None,
                    now,
                )
    for evaluation in list(self.state.evaluations.values()):
        if (
            evaluation.status == "running"
            and evaluation.started_at is not None
            and now - evaluation.started_at >= self.config.router.eval_timeout_seconds
        ):
            evaluation.attempts += 1
            if evaluation.attempts <= 2:
                evaluation.status = "pending"
                self.emit("assessment", {"eval_id": evaluation.id}, now)
            else:
                evaluation.status = "unavailable"
                self.assess(evaluation.thread, now)
    if now - self.state.last_maintenance >= 60:
        prune(self, now)


def prune(self: Engine, now: float) -> None:
    state = self.state
    cutoff = now - self.config.retention_days * 86400
    state.last_maintenance = now
    protected_tickets = {
        e.payload.get("ticket_id") for e in state.effects.values() if e.status != "done"
    }
    closed = sorted(
        (
            t
            for t in state.tickets.values()
            if t.status == "ended" and not t.question and t.id not in protected_tickets
        ),
        key=lambda t: max(
            (a.ended_at or a.created_at for a in t.assignments), default=0
        ),
    )
    excess = max(0, len(closed) - self.config.max_closed_tickets)
    for index, ticket in enumerate(closed):
        ended = max((a.ended_at or a.created_at for a in ticket.assignments), default=0)
        if index < excess or ended < cutoff:
            del state.tickets[ticket.id]
    protected_cards = {
        t.id
        for t in state.tickets.values()
        if any(a.awaiting_approval and a.status == "pending" for a in t.assignments)
    }
    done = sorted(
        (
            e
            for e in state.effects.values()
            if e.status == "done"
            and not (
                e.payload.get("card") and e.payload.get("ticket_id") in protected_cards
            )
        ),
        key=lambda e: (e.created_at, e.id),
    )
    for index, effect in enumerate(done):
        if (
            index < len(done) - self.config.max_history
            or effect.created_at < cutoff
            or effect.payload.get("ticket_id", "") not in state.tickets
            and effect.kind != "assessment"
        ):
            del state.effects[effect.id]
    active_evals = {
        e.payload.get("eval_id") for e in state.effects.values() if e.status != "done"
    }
    ended_evals = [
        e
        for e in state.evaluations.values()
        if e.status in {"ended", "unavailable"} and e.id not in active_evals
    ]
    for index, evaluation in enumerate(ended_evals):
        if (
            index < len(ended_evals) - self.config.max_history
            or (evaluation.started_at or 0) < cutoff
        ):
            del state.evaluations[evaluation.id]
    referenced = {k for q in state.thread_queues.values() for k in q}
    referenced.update(k for e in state.evaluations.values() for k in e.triggers)
    referenced.update(t.key for ticket in state.tickets.values() for t in ticket.inputs)
    referenced.update(ticket.mandate.trigger.key for ticket in state.tickets.values())
    referenced.update(
        str(e.payload["trigger"])
        for e in state.effects.values()
        if e.status != "done" and "trigger" in e.payload
    )
    state.triggers = {k: t for k, t in state.triggers.items() if k in referenced}
    state.thread_queues = {k: q for k, q in state.thread_queues.items() if q}
    state.seen = {k: stamp for k, stamp in state.seen.items() if stamp >= cutoff}


def recover(self: Engine, now: float) -> None:
    for effect in self.state.effects.values():
        if effect.status == "running":
            effect.status = "pending"
    for ticket in self.state.tickets.values():
        for assignment in list(ticket.assignments):
            if assignment.status == "running":
                assignment.status = "ended"
                assignment.ended_at = now
                assignment.result = "host restarted"
                if self.admitted(ticket):
                    ticket.notes.append(RECOVERY_NOTE)
                    self.assign(
                        ticket, now, followup=assignment.followup, recovery=True
                    )
    for evaluation in self.state.evaluations.values():
        if evaluation.status == "running":
            evaluation.status = "pending"
        if evaluation.status == "pending" and not any(
            e.kind == "assessment"
            and e.payload.get("eval_id") == evaluation.id
            and e.status in {"pending", "running"}
            for e in self.state.effects.values()
        ):
            self.emit("assessment", {"eval_id": evaluation.id}, now)
