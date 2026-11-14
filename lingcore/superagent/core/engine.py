"""The only domain writer: deterministic commands with transactional effects."""

from __future__ import annotations

from typing import Any, Literal

from lingcore.delegation.models import digest
from lingcore.errors import ToolError
from lingcore.superagent.config import SuperagentConfig
from lingcore.superagent.core.models import (
    Approval,
    Assignment,
    Effect,
    Evaluation,
    Mandate,
    State,
    Ticket,
    Trigger,
)

RECOVERY_NOTE = "Host restarted. Check the directory, git state and channel before repeating outside actions."


class Engine:
    def __init__(
        self,
        state: State,
        config: SuperagentConfig,
        *,
        ceiling: frozenset[str],
        roles: dict[str, frozenset[str]],
        router_tools: frozenset[str],
        constraints_digest: str,
    ) -> None:
        (
            self.state,
            self.config,
            self.ceiling,
            self.roles,
            self.router_tools,
            self.constraints,
        ) = state, config, ceiling, roles, router_tools, constraints_digest

    def identifier(self, prefix: str) -> str:
        self.state.sequence += 1
        return f"{prefix}-{self.state.sequence:08d}"

    def emit(self, kind: Any, payload: dict[str, Any], now: float) -> str:
        effect = Effect(
            id=self.identifier("effect"),
            kind=kind,
            payload=payload,
            created_at=now,
            due_at=now,
        )
        self.state.effects[effect.id] = effect
        return effect.id

    def gate(self, trigger: Trigger) -> Literal["explicit", "inferred"] | None:
        source = self.config.sources.get(trigger.source)
        if (
            source is None
            or not source.enabled
            or trigger.bot
            or trigger.actor in source.bots
            or trigger.actor not in source.allowlist
        ):
            return None
        mode = self.config.policy.accept_mode
        if mode == "off":
            return None
        if trigger.addressed:
            return "explicit"
        return "inferred" if mode == "auto" and self.config.policy.rules else None

    def ingest(self, trigger: Trigger, now: float) -> bool:
        if trigger.key in self.state.triggers or trigger.key in self.state.seen:
            return False
        self.state.seen[trigger.key] = now
        while len(self.state.seen) > self.config.max_history:
            self.state.seen.pop(next(iter(self.state.seen)))
        self.state.source_positions[trigger.source] = trigger.external_id
        source = self.config.sources.get(trigger.source)
        if (
            source is None
            or not source.enabled
            or trigger.bot
            or trigger.actor in source.bots
            or trigger.actor not in source.allowlist
        ):
            return False
        for ticket in self.state.tickets.values():
            if (
                trigger.thread in ticket.subscriptions
                and ticket.status == "open"
                and not ticket.withdrawn
                and trigger.key != ticket.mandate.trigger.key
            ):
                self.state.triggers[trigger.key] = trigger
                self.append_input(ticket, trigger, now)
        if self.gate(trigger):
            self.state.triggers[trigger.key] = trigger
            self.state.thread_queues.setdefault(trigger.thread, []).append(trigger.key)
            self.assess(trigger.thread, now)
        return True

    def assess(self, thread: str, now: float) -> None:
        if any(
            e.thread == thread and e.status in {"pending", "running"}
            for e in self.state.evaluations.values()
        ):
            return
        queued = self.state.thread_queues.get(thread, [])
        if not queued:
            return
        evaluation = Evaluation(
            id=self.identifier("eval"), thread=thread, triggers=queued[:20]
        )
        self.state.thread_queues[thread] = queued[20:]
        self.state.evaluations[evaluation.id] = evaluation
        self.emit("assessment", {"eval_id": evaluation.id}, now)

    def route(
        self,
        eval_id: str,
        decisions: list[dict[str, Any]],
        snapshot: frozenset[str],
        now: float,
    ) -> list[dict[str, str]]:
        evaluation = self.state.evaluations.get(eval_id)
        if evaluation is None or evaluation.status not in {"pending", "running"}:
            raise ToolError("evaluation is not active")
        if not 1 <= len(decisions) <= 20:
            raise ToolError("route accepts 1–20 decisions")
        results = []
        for decision in decisions:
            try:
                if decision.get("trigger") in evaluation.resolved:
                    results.append({"status": "ok", "ticket_id": ""})
                    continue
                result = self._decision(evaluation, decision, snapshot, now)
                evaluation.resolved.append(decision["trigger"])
                results.append({"status": "ok", "ticket_id": result or ""})
            except (ToolError, KeyError, ValueError) as exc:
                results.append({"status": "error", "reason": str(exc)[:300]})
        if set(evaluation.triggers) <= set(evaluation.resolved):
            evaluation.status = "ended"
            self.assess(evaluation.thread, now)
        return results

    def _decision(
        self,
        evaluation: Evaluation,
        decision: dict[str, Any],
        snapshot: frozenset[str],
        now: float,
    ) -> str | None:
        if set(decision) - {"action", "trigger", "ticket_id", "rules", "note"}:
            raise ToolError("unknown routing fields")
        note = decision.get("note", "")
        if len(note.split()) > 30:
            raise ToolError("routing notes are limited to 30 words")
        key = decision.get("trigger")
        if key not in evaluation.triggers:
            raise ToolError("trigger is not in this evaluation")
        trigger = self.state.triggers[key]
        strength = self.gate(trigger)
        if strength is None:
            raise ToolError("policy no longer admits trigger")
        action = decision.get("action")
        if action == "ignore":
            return None
        if action in {"attach", "stop"}:
            ticket = self.ticket(decision["ticket_id"])
            if (
                trigger.actor != ticket.mandate.requester
                and trigger.actor not in self.config.policy.approvers
            ):
                raise ToolError("only requester or approver may change this ticket")
            if action == "stop":
                ticket.withdrawn = True
                ticket.status = "ended"
                for assignment in ticket.assignments:
                    if assignment.status != "ended":
                        assignment.status = "ended"
                        assignment.ended_at = now
                        assignment.result = "stopped by person"
                self.emit("delivery", {"ticket_id": ticket.id, "stop": True}, now)
            else:
                if ticket.withdrawn:
                    raise ToolError("stopped mandate cannot be reopened")
                self.append_input(ticket, trigger, now)
                if not any(a.status != "ended" for a in ticket.assignments):
                    self.assign(ticket, now, followup=True)
            return ticket.id
        if action != "accept":
            raise ToolError("unknown routing action")
        # One accepted Ticket per original trigger, including retried assessments.
        for ticket in self.state.tickets.values():
            if ticket.mandate.trigger.key == trigger.key:
                return ticket.id
        matches = decision.get("rules", [])
        rules = {r.id: r for r in self.config.policy.rules}
        if strength == "inferred" and (
            not matches or any(r not in rules for r in matches)
        ):
            raise ToolError("inferred routing requires current rule ids")
        role = (
            self.config.executor.role
            if strength == "explicit"
            else self.config.executor.proactive_role
        )
        if strength == "inferred":
            selected = {rules[r].role for r in matches if rules[r].role}
            if len(selected) > 1:
                raise ToolError("matched rules disagree on role")
            if selected:
                role = str(next(iter(selected)))
        if role not in self.roles:
            raise ToolError("executor role does not exist")
        wanted = self.roles[role]
        if not wanted <= self.ceiling:
            raise ToolError("executor tools exceed profile ceiling")
        if self.config.policy.tools is not None and not wanted <= set(
            self.config.policy.tools
        ):
            raise ToolError("policy does not permit executor's requested tools")
        inherited = wanted & snapshot & self.ceiling
        mandate = Mandate(
            policy_version=self.config.policy.revision(),
            strength=strength,
            requester=trigger.actor,
            trigger=trigger,
            rules={r: rules[r].revision() for r in matches},
            role=role,
            inherited=inherited,
            requested=wanted,
            envelope_digest=digest(
                {"role": role, "tools": sorted(wanted), "constraints": self.constraints}
            ),
            constraints_digest=self.constraints,
        )
        ticket = Ticket(
            id=self.identifier("ticket"),
            name="",
            mandate=mandate,
            inputs=[trigger],
            subscriptions=[trigger.thread],
            notes=[note] if note else [],
        )
        ticket.name = ticket.id
        self.state.tickets[ticket.id] = ticket
        self.assign(ticket, now)
        return ticket.id

    def admitted(self, ticket: Ticket) -> bool:
        m = ticket.mandate
        if (
            ticket.withdrawn
            or m.role not in self.roles
            or m.constraints_digest != self.constraints
        ):
            return False
        if self.gate(m.trigger) != m.strength:
            return False
        rules = {r.id: r.revision() for r in self.config.policy.rules}
        return all(rules.get(k) == v for k, v in m.rules.items())

    def grant(self, ticket: Ticket) -> frozenset[str]:
        if not self.admitted(ticket):
            raise ToolError("mandate is no longer admitted")
        inherited = ticket.mandate.inherited & self.router_tools
        extras = (
            ticket.approval.tools
            if ticket.approval
            and ticket.approval.envelope_digest == ticket.mandate.envelope_digest
            and ticket.approval.actor in self.config.policy.approvers
            else frozenset()
        )
        policy = (
            frozenset(self.config.policy.tools)
            if self.config.policy.tools is not None
            else self.ceiling
        )
        return (
            (inherited | extras)
            & self.ceiling
            & policy
            & self.roles[ticket.mandate.role]
            & ticket.mandate.requested
        )

    def assign(
        self,
        ticket: Ticket,
        now: float,
        *,
        followup: bool = False,
        recovery: bool = False,
    ) -> Assignment:
        wanted = ticket.mandate.requested & self.roles[ticket.mandate.role]
        extra = wanted - self.grant(ticket)
        assignment = Assignment(
            id=self.identifier("assignment"),
            created_at=now,
            followup=followup,
            recovery=recovery,
            awaiting_approval=bool(extra),
        )
        ticket.assignments.append(assignment)
        ticket.status = "open"
        if extra:
            for approver in self.config.policy.approvers:
                self.emit(
                    "reply",
                    {
                        "ticket_id": ticket.id,
                        "purpose": "attention",
                        "approval": True,
                        "recipient": approver,
                        "tools": sorted(extra),
                        "digest": ticket.mandate.envelope_digest,
                    },
                    now,
                )
        else:
            self.emit(
                "launch", {"ticket_id": ticket.id, "assignment_id": assignment.id}, now
            )
        return assignment

    def approve(
        self,
        ticket_id: str,
        actor: str,
        tools: frozenset[str],
        envelope_digest: str,
        allow: bool,
        now: float,
    ) -> None:
        # Only authenticated transport actions call this method; it has no model tool.
        ticket = self.ticket(ticket_id)
        if actor not in self.config.policy.approvers:
            raise ToolError("actor is not an approver")
        if envelope_digest != ticket.mandate.envelope_digest:
            raise ToolError("stale approval")
        ungranted = ticket.mandate.requested - self.grant(ticket)
        wanted = ungranted & self.roles.get(ticket.mandate.role, frozenset())
        if self.config.policy.tools is not None:
            wanted &= frozenset(self.config.policy.tools)
        if not tools <= ungranted or not tools <= self.ceiling:
            raise ToolError("approval tools do not match this ticket")
        tools &= wanted
        pending = [
            a
            for a in ticket.assignments
            if a.status == "pending" and a.awaiting_approval
        ]
        if not pending:
            raise ToolError("ticket has no pending approval")
        if any(
            now - a.created_at >= self.config.executor.approval_timeout_seconds
            for a in pending
        ):
            raise ToolError("approval expired")
        if not allow:
            for assignment in pending:
                self.finish(
                    ticket, assignment, "blocked", "Extra tools were denied.", None, now
                )
            return
        ticket.approval = Approval(
            actor=actor, timestamp=now, tools=tools, envelope_digest=envelope_digest
        )
        for assignment in pending:
            assignment.awaiting_approval = False
            self.emit(
                "launch", {"ticket_id": ticket.id, "assignment_id": assignment.id}, now
            )

    def launch(
        self, ticket_id: str, assignment_id: str, now: float
    ) -> Assignment | None:
        ticket = self.ticket(ticket_id)
        assignment = self.assignment(ticket, assignment_id)
        if assignment.status != "pending" or assignment.awaiting_approval:
            return None
        if not self.admitted(ticket):
            self.finish(
                ticket,
                assignment,
                "blocked",
                "Mandate is no longer admitted.",
                None,
                now,
            )
            return None
        wanted = ticket.mandate.requested & self.roles[ticket.mandate.role]
        if ticket.approval:
            wanted &= ticket.mandate.inherited | ticket.approval.tools
        if not wanted <= self.grant(ticket):
            self.finish(
                ticket,
                assignment,
                "blocked",
                "Grant narrowed; a new approval is required.",
                None,
                now,
            )
            return None
        active = [
            a
            for t in self.state.tickets.values()
            for a in t.assignments
            if a.status == "running"
        ]
        cap = self.config.executor.max_workers + (1 if assignment.followup else 0)
        if len(active) >= cap:
            from lingcore.superagent.core.models import SlotsFull

            raise SlotsFull("executor slots are full")
        assignment.status = "running"
        assignment.started_at = now
        assignment.input_cursor = len(ticket.inputs)
        return assignment

    def report(
        self,
        ticket_id: str,
        assignment_id: str,
        status: str,
        summary: str,
        question: str | None,
        final: bool,
        now: float,
        *,
        notify: bool = True,
    ) -> None:
        ticket = self.ticket(ticket_id)
        assignment = self.assignment(ticket, assignment_id)
        if assignment.status != "running":
            raise ToolError("assignment is not running")
        if status not in {
            "progress",
            "waiting",
            "ready",
            "blocked",
            "failed",
            "dropped",
        }:
            raise ToolError("invalid ticket report")
        if status in {"waiting", "blocked"} and not question:
            raise ToolError("waiting or blocked requires one concrete question")
        if len(summary) > 2000 or (question and len(question) > 2000):
            raise ToolError("ticket report is too long")
        ticket.report, ticket.question = summary, question
        if final and status != "progress":
            self.finish(ticket, assignment, status, summary, question, now)
        elif notify:
            self.emit(
                "relay",
                {"ticket_id": ticket_id, "report": summary, "question": question},
                now,
            )

    def finish(
        self,
        ticket: Ticket,
        assignment: Assignment,
        status: str,
        summary: str,
        question: str | None,
        now: float,
    ) -> None:
        assignment.status = "ended"
        assignment.ended_at = now
        assignment.result = status
        ticket.status = "ended"
        ticket.report = summary
        ticket.question = question
        self.emit(
            "reply",
            {
                "ticket_id": ticket.id,
                "purpose": "attention" if question else "result",
                "text": question or summary,
            },
            now,
        )

        self.deliver(ticket.id, now)

    def tick(self, now: float) -> None:
        from lingcore.superagent.core.maintenance import tick

        tick(self, now)

    def recover(self, now: float) -> None:
        from lingcore.superagent.core.maintenance import recover

        recover(self, now)

    def ticket(self, ticket_id: str) -> Ticket:
        if ticket_id not in self.state.tickets:
            raise ToolError("unknown ticket")
        return self.state.tickets[ticket_id]

    @staticmethod
    def assignment(ticket: Ticket, assignment_id: str) -> Assignment:
        for assignment in ticket.assignments:
            if assignment.id == assignment_id:
                return assignment
        raise ToolError("unknown assignment")

    def claim_effect(self, effect_id: str) -> None:
        effect = self.state.effects[effect_id]
        if effect.status != "pending":
            raise ToolError("effect is not pending")
        effect.status = "running"

    def finish_effect(self, effect_id: str, now: float, error: str | None) -> None:
        effect = self.state.effects[effect_id]
        if effect.kind == "launch":
            ticket = self.state.tickets.get(effect.payload["ticket_id"])
            assignment = (
                next(
                    (
                        a
                        for a in ticket.assignments
                        if a.id == effect.payload["assignment_id"]
                    ),
                    None,
                )
                if ticket
                else None
            )
            if assignment is None or assignment.status == "ended":
                error = None
        if error is None:
            effect.status = "done"
            effect.last_error = None
        else:
            effect.status = "pending"
            effect.attempts += 1
            effect.last_error = error
            effect.due_at = now + min(
                300 if effect.kind == "reply" else 60,
                (5 if effect.kind == "reply" else 2)
                * 2 ** min(effect.attempts - 1, 10),
            )

    def start_eval(self, eval_id: str, now: float) -> Evaluation:
        evaluation = self.state.evaluations[eval_id]
        if evaluation.status != "pending":
            raise ToolError("evaluation already claimed")
        evaluation.status = "running"
        evaluation.started_at = now
        return evaluation.model_copy(deep=True)

    def note(self, ticket_id: str, text: str) -> None:
        ticket = self.ticket(ticket_id)
        ticket.notes.append(text[:2000])
        ticket.notes = ticket.notes[-100:]

    def follow(self, ticket_id: str, thread: str) -> None:
        if not any(t.thread == thread for t in self.state.triggers.values()):
            raise ToolError("unknown thread")
        ticket = self.ticket(ticket_id)
        if thread not in ticket.subscriptions:
            ticket.subscriptions.append(thread)

    def reply(self, ticket_id: str, purpose: str, text: str, now: float) -> None:
        ticket = self.ticket(ticket_id)
        if purpose not in {"ack", "answer", "result", "attention"} or len(text) > 2000:
            raise ToolError("invalid reply")
        if purpose == "attention" and ticket.question and text != ticket.question:
            raise ToolError("forward the worker question unchanged")
        self.emit(
            "reply", {"ticket_id": ticket_id, "purpose": purpose, "text": text}, now
        )

    def retry(self, ticket_id: str, now: float) -> None:
        ticket = self.ticket(ticket_id)
        if ticket.withdrawn or any(a.status != "ended" for a in ticket.assignments):
            raise ToolError("ticket cannot be retried")
        if not ticket.assignments or ticket.assignments[-1].result not in {
            "failed",
            "host restarted",
        }:
            raise ToolError("only interrupted or failed assignments can retry")
        self.assign(ticket, now, recovery=True)

    def bind_worker(
        self,
        ticket_id: str,
        assignment_id: str,
        run_id: str,
        worker_id: str | None,
        session_id: str | None,
    ) -> None:
        assignment = self.assignment(self.ticket(ticket_id), assignment_id)
        assignment.run_id, assignment.worker_id, assignment.session_id = (
            run_id,
            worker_id,
            session_id,
        )

    def save_approval_card(self, effect_id: str, card: dict[str, Any]) -> None:
        self.state.effects[effect_id].payload["card"] = card

    def prepare(self, ticket_id: str, *, complete: bool) -> None:
        ticket = self.ticket(ticket_id)
        ticket.prepare_started = True
        ticket.prepared = complete

    def defer_effect(self, effect_id: str, now: float) -> None:
        effect = self.state.effects[effect_id]
        effect.status, effect.due_at = "pending", now + 0.25

    def append_input(self, ticket: Ticket, trigger: Trigger, now: float) -> None:
        if not any(t.key == trigger.key for t in ticket.inputs):
            ticket.inputs.append(trigger)
            self.emit("delivery", {"ticket_id": ticket.id, "trigger": trigger.key}, now)

    def take_inputs(self, ticket_id: str, assignment_id: str) -> list[Trigger]:
        ticket = self.ticket(ticket_id)
        assignment = self.assignment(ticket, assignment_id)
        if assignment.status != "running":
            return []
        values = ticket.inputs[assignment.input_cursor :]
        assignment.input_cursor = len(ticket.inputs)
        return list(values)

    def deliver(self, ticket_id: str, now: float) -> None:
        ticket = self.ticket(ticket_id)
        if ticket.withdrawn or not ticket.assignments:
            return
        assignment = ticket.assignments[-1]
        if (
            assignment.status == "ended"
            and assignment.started_at is not None
            and assignment.input_cursor < len(ticket.inputs)
        ):
            self.assign(ticket, now, followup=True)

    def checkpoint(self, source: str, position: str) -> None:
        self.state.source_positions[source] = position

    def interrupt(self, ticket_id: str, assignment_id: str, now: float) -> None:
        ticket = self.ticket(ticket_id)
        assignment = self.assignment(ticket, assignment_id)
        if assignment.status != "running":
            return
        assignment.status, assignment.ended_at, assignment.result = (
            "ended",
            now,
            "host restarted",
        )
        if self.admitted(ticket):
            self.note(ticket_id, RECOVERY_NOTE)
            self.assign(ticket, now, followup=assignment.followup, recovery=True)
