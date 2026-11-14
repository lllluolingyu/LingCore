"""Worker spend survives persistence without changing own-agent usage."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from lingcore.events import (
    DelegatedUsageReported,
    ToolResultEvent,
    TurnStatsReported,
    UsageReported,
)
from lingcore.message import ToolCall
from lingcore.sessions import SessionStore
from lingcore.tools import Tool
from lingcore.usage import (
    DelegatedUsage,
    TokenUsage,
    TurnClock,
    UsageMeter,
    request_usage_observer,
    turn_stats_from_payload,
    turn_stats_payload,
)
from tests.fakes import ScriptedTurn
from tests.support.delegation import EmptyArgs, build_agent

RECORD = DelegatedUsage("run-1", "worker-1", "reviewer", TokenUsage("small", 10, 5))


def test_delegated_channel_has_independent_pending_and_tally_and_no_billing_observer():
    meter = UsageMeter()
    observed = []
    token = request_usage_observer.set(observed.append)
    try:
        meter.open_tally()
        meter.record_delegated(RECORD)
        assert meter.drain_with_purpose() == [] and observed == []
        assert meter.drain_delegated() == [RECORD]
        assert meter.drain_delegated() == []
        assert meter.close_tally() == ()
        assert meter.close_delegated_tally() == (RECORD,)
        meter.open_tally()
        assert meter.close_delegated_tally() == ()
    finally:
        request_usage_observer.reset(token)


def test_payload_roundtrip_preserves_rewrite_and_legacy_records():
    own = (
        TokenUsage("main", 100, 20),
        TokenUsage("rewrite", 50, 10, purpose="output_rewrite"),
    )
    stats = TurnClock().finish(own, (RECORD,))
    payload = turn_stats_payload(stats)
    assert turn_stats_from_payload(payload) == stats
    assert "purpose" not in payload["usage"][0]
    assert payload["usage"][1]["purpose"] == "output_rewrite"
    legacy = turn_stats_payload(replace(stats, delegated=()))
    assert "delegated" not in legacy
    assert turn_stats_from_payload(legacy).delegated == ()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        [None],
        [{}],
        [{"run_id": "r", "worker_id": "w", "role": "x", "usage": None}],
    ],
)
def test_malformed_delegated_payload_is_refused(bad):
    payload = turn_stats_payload(TurnClock().finish(()))
    payload["delegated"] = bad
    assert turn_stats_from_payload(payload) is None


async def test_delegated_usage_precedes_result_and_persists_separately(tmp_path):
    async def work(args, ctx):
        agent.usage_meter.record_delegated(RECORD)
        return "worker finished"

    with SessionStore(tmp_path / "sessions.db") as store:
        own = TokenUsage("parent", 1000, 100)
        agent, _ = build_agent(
            tmp_path,
            [
                ScriptedTurn(tool_calls=[ToolCall(id="d", name="work")], usage=own),
                ScriptedTurn(text="done", usage=own),
            ],
            extra_tools=(Tool("work", "work", EmptyArgs, work),),
            session_store=store,
        )
        async with agent:
            events = [e async for e in agent.run("go")]
            sid = agent.tool_ctx.session_id
        worker_event = next(e for e in events if isinstance(e, DelegatedUsageReported))
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert events.index(worker_event) < events.index(result_event)
        assert [e.usage for e in events if isinstance(e, UsageReported)] == [own, own]
        stats = next(e.stats for e in events if isinstance(e, TurnStatsReported))
        assert stats.usage == (own, own) and stats.delegated == (RECORD,)
        stored = store.turn_stats(sid)
        assert next(iter(stored.values())).delegated == (RECORD,)


async def test_stop_preserves_delegated_spend_for_drain_and_turn_stats(tmp_path):
    started = asyncio.Event()

    async def work(args, ctx):
        agent.usage_meter.record_delegated(RECORD)
        started.set()
        await asyncio.Event().wait()

    with SessionStore(tmp_path / "sessions.db") as store:
        agent, _ = build_agent(
            tmp_path,
            [ScriptedTurn(tool_calls=[ToolCall(id="d", name="work")])],
            extra_tools=(Tool("work", "work", EmptyArgs, work),),
            session_store=store,
        )

        async def consume():
            return [e async for e in agent.run("go")]

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(started.wait(), 2)
            agent.cancel_turn()
            with pytest.raises(asyncio.CancelledError):
                await task
            agent.finalize_cancelled_turn()
            assert agent.drain_usage() == [DelegatedUsageReported((RECORD,))]
            stats = next(iter(store.turn_stats(agent.tool_ctx.session_id).values()))
            assert stats.usage == () and stats.delegated == (RECORD,)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await agent.aclose()
