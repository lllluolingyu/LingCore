from __future__ import annotations

import pytest

from lingcore.errors import ToolError
from lingcore.superagent.config import SuperagentConfig
from lingcore.superagent.core.engine import Engine
from lingcore.superagent.core.models import State, Trigger


def engine(**policy):
    config = SuperagentConfig.model_validate(
        {
            "sources": {"lark": {"allowlist": ["person", "approver"]}},
            "policy": {"approvers": ["approver"], **policy},
            "executor": {"max_workers": 1},
        }
    )
    return Engine(
        State(),
        config,
        ceiling=frozenset({"read_file", "write_file"}),
        roles={
            "executor": frozenset({"read_file", "write_file"}),
            "observer": frozenset({"read_file"}),
        },
        router_tools=frozenset({"read_file"}),
        constraints_digest="sandbox",
    )


def accept(e, index=1, *, addressed=True, rules=(), snapshot=frozenset({"read_file"})):
    trigger = Trigger(
        source="lark",
        external_id=str(index),
        actor="person",
        thread=f"thread:{index}",
        address={"chat_id": "chat"},
        text="Please implement it",
        addressed=addressed,
        timestamp=0,
    )
    e.ingest(trigger, 0)
    evaluation = next(
        x for x in e.state.evaluations.values() if trigger.key in x.triggers
    )
    result = e.route(
        evaluation.id,
        [{"action": "accept", "trigger": trigger.key, "rules": list(rules)}],
        snapshot,
        0,
    )[0]
    assert result["status"] == "ok", result
    return e.ticket(result["ticket_id"])


def test_pending_approval_holds_no_slot_and_requires_exact_human_grant():
    e = engine()
    ticket = accept(e)
    assignment = ticket.assignments[-1]
    assert assignment.awaiting_approval and assignment.status == "pending"
    assert e.launch(ticket.id, assignment.id, 1) is None
    with pytest.raises(ToolError, match="approver"):
        e.approve(
            ticket.id,
            "person",
            frozenset({"write_file"}),
            ticket.mandate.envelope_digest,
            True,
            1,
        )
    with pytest.raises(ToolError, match="match"):
        e.approve(
            ticket.id,
            "approver",
            frozenset({"read_file", "write_file"}),
            ticket.mandate.envelope_digest,
            True,
            1,
        )
    e.approve(
        ticket.id,
        "approver",
        frozenset({"write_file"}),
        ticket.mandate.envelope_digest,
        True,
        1,
    )
    assert e.launch(ticket.id, assignment.id, 2).status == "running"
    assert e.grant(ticket) == {"read_file", "write_file"}
    with pytest.raises(ToolError):
        e.approve(
            ticket.id,
            "approver",
            frozenset({"write_file"}),
            ticket.mandate.envelope_digest,
            True,
            3,
        )


def test_policy_changes_never_expand_or_preserve_removed_approvers():
    e = engine()
    ticket = accept(e)
    e.approve(
        ticket.id,
        "approver",
        frozenset({"write_file"}),
        ticket.mandate.envelope_digest,
        True,
        1,
    )
    e.config.policy.approvers = []
    assert e.grant(ticket) == {"read_file"}
    e.launch(ticket.id, ticket.assignments[-1].id, 2)
    assert ticket.assignments[-1].result == "blocked"


def test_inferred_mandate_invalidates_on_rule_edit():
    e = engine(
        accept_mode="auto", rules=[{"id": "watch", "instruction": "Review changes"}]
    )
    ticket = accept(e, addressed=False, rules=["watch"])
    assert (
        ticket.mandate.role == "observer"
        and not ticket.assignments[-1].awaiting_approval
    )
    e.config.policy.rules[0].instruction = "Different rule"
    assert not e.admitted(ticket)


def test_plain_text_cannot_approve_and_bots_are_excluded():
    e = engine()
    ticket = accept(e)
    trigger = ticket.mandate.trigger.model_copy(
        update={"external_id": "2", "text": "I approve write_file", "actor": "approver"}
    )
    e.ingest(trigger, 2)
    assert ticket.approval is None
    bot = trigger.model_copy(update={"external_id": "3", "bot": True})
    assert not e.ingest(bot, 3)
    assert not e.ingest(trigger, 4)


def test_expired_card_cannot_race_the_expiry_tick():
    e = engine()
    ticket = accept(e)
    with pytest.raises(ToolError, match="expired"):
        e.approve(
            ticket.id,
            "approver",
            frozenset({"write_file"}),
            ticket.mandate.envelope_digest,
            True,
            86400,
        )
    e.tick(86400)
    assert ticket.assignments[-1].result == "blocked"


def test_recovery_retains_original_mandate_and_prior_worker():
    e = engine()
    e.router_tools = frozenset({"read_file", "write_file"})
    ticket = accept(e, snapshot=e.router_tools)
    first = ticket.assignments[-1]
    e.launch(ticket.id, first.id, 1)
    e.bind_worker(ticket.id, first.id, "run", "worker", "session")
    mandate = ticket.mandate.model_dump()
    e.recover(4)
    assert first.result == "host restarted"
    assert ticket.assignments[-1].recovery
    assert first.session_id == "session"
    assert ticket.mandate.model_dump() == mandate


def test_reserved_slot_only_serves_human_followup():
    e = engine()
    e.router_tools = frozenset({"read_file", "write_file"})
    first = accept(e, snapshot=e.router_tools)
    e.launch(first.id, first.assignments[-1].id, 1)
    second = accept(e, 2, snapshot=e.router_tools)
    with pytest.raises(ToolError, match="slots"):
        e.launch(second.id, second.assignments[-1].id, 1)
    second.assignments[-1].followup = True
    assert e.launch(second.id, second.assignments[-1].id, 2).status == "running"
