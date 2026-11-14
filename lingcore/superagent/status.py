"""Bounded, transport-neutral needs-person and service-health projection."""

from __future__ import annotations

from typing import Any

from lingcore.superagent.core.models import State


def status_payload(state: State) -> dict[str, Any]:
    return {
        "tickets": [
            {
                "id": t.id,
                "name": t.name,
                "status": t.status,
                "question": t.question,
                "report": t.report,
                "awaiting_approval": any(
                    a.awaiting_approval and a.status == "pending" for a in t.assignments
                ),
                "requested_tools": sorted(t.mandate.requested),
                "requester": t.mandate.requester,
                "source": t.mandate.trigger.source,
                "session_id": next(
                    (a.session_id for a in reversed(t.assignments) if a.session_id),
                    None,
                ),
            }
            for t in list(state.tickets.values())[-500:]
        ],
        "health": {
            "errors": [
                {"effect": e.id, "kind": e.kind, "failures": e.attempts}
                for e in state.effects.values()
                if e.status != "done" and e.attempts >= 3
            ]
        },
        "source_positions": state.source_positions,
    }
