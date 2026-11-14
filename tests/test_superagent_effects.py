from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from lingcore.superagent.effects import EffectRunner
from lingcore.superagent.store import SuperagentStore
from tests.test_superagent_core import engine


async def test_outbox_retry_preserves_order_and_transactional_state(tmp_path):
    e = engine()
    store = SuperagentStore(tmp_path / "service.sqlite3")

    def factory(state):
        e.state = state
        return e

    runtime = NS(store=store, command=lambda fn: store.transact(factory, fn))
    first = runtime.command(
        lambda core: core.emit("reply", {"ticket_id": "ticket", "text": "first"}, 0)
    )
    second = runtime.command(
        lambda core: core.emit("reply", {"ticket_id": "ticket", "text": "second"}, 1)
    )
    sent = []
    failed = False

    async def reply(effect):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("network unavailable")
        sent.append(effect.payload["text"])

    runner = EffectRunner(runtime, {"reply": reply})
    try:
        await runner.once()
        await asyncio.gather(*runner.tasks.values())
        assert store.read().effects[first].attempts == 1
        assert store.read().effects[second].status == "pending"
        await runner.once()
        assert not sent  # backoff blocks the second message in this ticket
        runtime.command(lambda core: setattr(core.state.effects[first], "due_at", 0))
        await runner.once()
        await asyncio.gather(*runner.tasks.values())
        await runner.once()
        await asyncio.gather(*runner.tasks.values())
        assert sent == ["first", "second"]
        assert all(x.status == "done" for x in store.read().effects.values())
    finally:
        await runner.close()
        store.close()


def test_command_failure_rolls_back_decisions_and_effects(tmp_path):
    e = engine()
    store = SuperagentStore(tmp_path / "service.sqlite3")

    def factory(state):
        e.state = state
        return e

    def broken(core):
        core.emit("reply", {"ticket_id": "t"}, 1)
        raise ValueError("abort")

    try:
        with pytest.raises(ValueError):
            store.transact(factory, broken)
        assert not store.read().effects
        assert store.conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
    finally:
        store.close()


def test_recovery_requeues_assessment_after_handler_completed_without_route():
    e = engine()
    from lingcore.superagent.core.models import Trigger

    t = Trigger(
        source="lark",
        external_id="1",
        actor="person",
        thread="thread",
        address={},
        text="hello",
        addressed=True,
        timestamp=0,
    )
    e.ingest(t, 0)
    evaluation = next(iter(e.state.evaluations.values()))
    effect = next(iter(e.state.effects.values()))
    e.start_eval(evaluation.id, 1)
    e.finish_effect(effect.id, 2, None)
    e.recover(3)
    assert evaluation.status == "pending"
    assert any(
        x.kind == "assessment" and x.status == "pending"
        for x in e.state.effects.values()
    )


async def test_unreachable_approver_does_not_block_other_recipients(tmp_path):
    e = engine()
    store = SuperagentStore(tmp_path / "service.db")

    def factory(state):
        e.state = state
        return e

    runtime = NS(store=store, command=lambda fn: store.transact(factory, fn))
    for recipient in ("unreachable", "available"):
        runtime.command(
            lambda core: core.emit(
                "reply", {"ticket_id": "same", "recipient": recipient}, 1
            )
        )
    sent = []

    async def reply(effect):
        if effect.payload["recipient"] == "unreachable":
            raise OSError("unreachable")
        sent.append(effect.payload["recipient"])

    runner = EffectRunner(runtime, {"reply": reply})
    try:
        await runner.once()
        await asyncio.gather(*runner.tasks.values())
        assert sent == ["available"]
    finally:
        await runner.close()
        store.close()
