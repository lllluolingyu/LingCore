"""Authenticated host actions for a needs-person panel; never model tools."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from lingcore.config import AgentProfile
from lingcore.delegation.roles import load_roles
from lingcore.errors import ToolError
from lingcore.superagent.authority import panel_engine
from lingcore.superagent.core.engine import Engine
from lingcore.superagent.core.models import Trigger
from lingcore.superagent.status import status_payload
from lingcore.superagent.store import SuperagentStore


class Panel:
    def __init__(self, profile: AgentProfile) -> None:
        if profile.superagent is None:
            raise ToolError("this profile has no superagent")
        self.profile, self.config = profile, profile.superagent
        self.roles = load_roles(profile)

    def store(self) -> SuperagentStore:
        path = getattr(self.profile, "_source_dir") / self.config.store
        if not path.is_file():
            raise ToolError("superagent has not started yet")
        return SuperagentStore(path)

    def snapshot(self) -> dict[str, Any]:
        store = self.store()
        try:
            state = store.read()
            value = status_payload(state)
            for ticket in value["tickets"]:
                ticket["approvals"] = [
                    e.payload["card"]
                    for e in state.effects.values()
                    if e.payload.get("card", {}).get("ticket_id") == ticket["id"]
                    and ticket["awaiting_approval"]
                ]
            value["can_act"] = self.config.panel_actor in self.config.policy.approvers
            return value
        finally:
            store.close()

    def action(
        self, ticket_id: str, kind: str, *, token: str = "", text: str = ""
    ) -> None:
        # The actor is bound by operator configuration, not a request field.
        actor = self.config.panel_actor
        if not actor or actor not in self.config.policy.approvers:
            raise ToolError("panel actor is not configured as an approver")
        store = self.store()
        try:

            def apply(engine: Engine) -> None:
                if actor not in engine.config.policy.approvers:
                    raise ToolError("panel actor is no longer an approver")
                ticket = engine.ticket(ticket_id)
                if kind in {"approve", "deny"}:
                    card = next(
                        (
                            e.payload["card"]
                            for e in engine.state.effects.values()
                            if e.payload.get("card", {}).get("token") == token
                            and e.payload["card"]["ticket_id"] == ticket_id
                        ),
                        None,
                    )
                    if card is None:
                        raise ToolError("unknown approval card")
                    engine.approve(
                        ticket_id,
                        actor,
                        frozenset(card["tools"]),
                        card["envelope_digest"],
                        kind == "approve",
                        time.time(),
                    )
                elif kind == "reply":
                    if not text.strip() or len(text) > 20000:
                        raise ToolError("reply must contain 1–20000 characters")
                    original = ticket.mandate.trigger
                    trigger = Trigger(
                        source=original.source,
                        external_id="panel-" + uuid4().hex,
                        actor=actor,
                        thread=original.thread,
                        address=original.address,
                        text=text,
                        addressed=True,
                        timestamp=time.time(),
                    )
                    if engine.gate(trigger) is None:
                        raise ToolError(
                            "panel actor is not allowed by the source policy"
                        )
                    engine.ingest(trigger, time.time())
                else:
                    raise ToolError("unknown panel action")

            store.transact(panel_engine, apply)
        finally:
            store.close()
