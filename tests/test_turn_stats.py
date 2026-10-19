"""Per-turn usage and generation time: measured by the loop, kept per session."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import lingcore.usage
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.events import Final, TurnStatsReported, UsageReported
from lingcore.llm import LLMChunk
from lingcore.message import Message, ToolCall
from lingcore.sessions import SessionStore, new_session_id
from lingcore.usage import (
    TokenUsage,
    TurnClock,
    TurnStats,
    turn_stats_from_payload,
    turn_stats_payload,
)

PROFILE = """
name: stats
workspace: {ws}
llm: {{model: m}}
tools: [read_file]
"""


def _profile(tmp_path: Path) -> AgentProfile:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "a.txt").write_text("x", encoding="utf-8")
    cfg = tmp_path / "prof" / "config.yaml"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(PROFILE.format(ws=ws), encoding="utf-8")
    return AgentProfile.load(cfg)


class _Metered:
    """Scripted turns; each request reports usage into the agent's meter."""

    def __init__(self, turns: list[dict]) -> None:
        self.turns = list(turns)
        self.agent: Agent | None = None

    async def stream(self, messages, tools=None):
        turn = self.turns.pop(0)
        assert self.agent is not None
        usage = TokenUsage("m", *turn["usage"])
        if not turn.get("usage_at_end"):
            self.agent.usage_meter.record(usage)
        if turn.get("hang"):
            yield LLMChunk(text_delta="partial")
            await asyncio.Event().wait()
        if turn.get("text"):
            yield LLMChunk(text_delta=turn["text"])
        if turn.get("usage_at_end"):
            self.agent.usage_meter.record(usage)
        yield LLMChunk(tool_calls=turn.get("tool_calls"), finish_reason="stop")


def _build(tmp_path: Path, store: SessionStore, turns: list[dict], sid=None):
    llm = _Metered(turns)
    agent = Agent.from_profile(
        _profile(tmp_path), llm=llm, session_store=store, session_id=sid
    )
    llm.agent = agent
    return agent, llm


READ = ToolCall(id="t1", name="read_file", arguments={"path": "a.txt"})


def test_clock_leaves_waits_out_of_generation_time(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(lingcore.usage.time, "monotonic", lambda: now[0])
    clock = TurnClock()
    now[0] = 102.0  # 2s generating
    with clock.waiting():
        now[0] = 107.0  # a 5s tool batch
    now[0] = 108.5  # 1.5s generating
    stats = clock.finish((TokenUsage("m", 1, 2),))
    assert (stats.elapsed_ms, stats.model_ms) == (8500, 3500)
    assert stats.usage == (TokenUsage("m", 1, 2),)
    assert stats.started_at <= stats.ended_at


def test_payload_round_trips_and_rejects_malformed_rows() -> None:
    at = datetime(2026, 10, 19, 7, 32, tzinfo=timezone.utc)
    stats = TurnStats((TokenUsage("m", 10, 2, 8, 1),), at, at, 900, 600)
    payload = json.loads(json.dumps(turn_stats_payload(stats)))
    assert turn_stats_from_payload(payload) == stats
    for broken in (
        None,
        {**payload, "usage": "x"},
        {**payload, "usage": [{"input_tokens": 1}]},
        {**payload, "ended_at": "yesterday"},
        {**payload, "ended_at": "2026-10-19T07:32:00"},  # naive
        {**payload, "model_ms": -1},
        {**payload, "elapsed_ms": True},
    ):
        assert turn_stats_from_payload(broken) is None
    # Generation time never exceeds the wall time it is part of.
    assert turn_stats_from_payload({**payload, "model_ms": 5000}).model_ms == 900


async def test_stats_close_the_turn_and_exclude_tool_time(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "s.db") as store:
        agent, _ = _build(
            tmp_path,
            store,
            [
                {"tool_calls": [READ], "usage": (100, 5, 80)},
                {"text": "done", "usage": (120, 7, 100)},
            ],
        )
        dispatch = agent._dispatch

        async def slow(calls):
            await asyncio.sleep(0.3)
            return await dispatch(calls)

        agent._dispatch = slow  # type: ignore[method-assign]
        events = [e async for e in agent.run("look")]
        sid = agent.memory.session_id  # type: ignore[attr-defined]

        assert isinstance(events[-1], Final)
        assert isinstance(events[-2], TurnStatsReported)
        stats = events[-2].stats
        reported = tuple(e.usage for e in events if isinstance(e, UsageReported))
        assert (
            stats.usage
            == reported
            == (
                TokenUsage("m", 100, 5, 80),
                TokenUsage("m", 120, 7, 100),
            )
        )
        assert stats.elapsed_ms - stats.model_ms >= 250
        # Persisted against the user message that started the turn.
        assert store.turn_stats(sid) == {0: stats}


async def test_stopped_turn_keeps_its_billed_stats(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "s.db") as store:
        agent, _ = _build(
            tmp_path,
            store,
            [
                {"text": "first", "usage": (10, 1, 0)},
                {"hang": True, "usage": (20, 2, 0)},
            ],
        )
        [e async for e in agent.run("one")]
        sid = agent.memory.session_id  # type: ignore[attr-defined]
        seen: list = []

        async def drive() -> None:
            async for event in agent.run("two"):
                seen.append(event)
                if len(seen) == 2:  # usage, then the first TextDelta
                    assert agent.cancel_turn()

        task = asyncio.create_task(drive())
        with pytest.raises(asyncio.CancelledError):
            await task
        agent.finalize_cancelled_turn()

        assert not any(isinstance(e, TurnStatsReported) for e in seen)
        assert [r.seq for r in store.message_records(sid)] == [0, 1, 2]
        stats = store.turn_stats(sid)
        assert sorted(stats) == [0, 2]  # written after Stop's truncation
        assert stats[2].usage == (TokenUsage("m", 20, 2),)


@pytest.mark.parametrize("pause_at", [UsageReported, TurnStatsReported, Final])
async def test_closing_stale_stream_preserves_next_turn_stats(
    tmp_path: Path, pause_at
) -> None:
    with SessionStore(tmp_path / "s.db") as store:
        agent, _ = _build(
            tmp_path,
            store,
            [
                {"text": "first", "usage": (10, 1, 0), "usage_at_end": True},
                {"tool_calls": [READ], "usage": (20, 2, 0)},
                {"text": "second", "usage": (30, 3, 0)},
            ],
        )
        async with agent:
            old_stream = agent.run("one")
            while not isinstance(await anext(old_stream), pause_at):
                pass
            sid = agent.memory.session_id  # type: ignore[attr-defined]
            first_stats = store.turn_stats(sid)[0]

            new_stream = agent.run("two")
            first_event = await anext(new_stream)
            assert isinstance(first_event, UsageReported)
            await old_stream.aclose()

            # Closing the old generator must not persist the live turn early.
            assert store.turn_stats(sid) == {0: first_stats}
            remaining = [event async for event in new_stream]
            assert isinstance(remaining[-1], Final)
            assert isinstance(remaining[-2], TurnStatsReported)
            stats = remaining[-2].stats
            assert stats.usage == (TokenUsage("m", 20, 2), TokenUsage("m", 30, 3))
            assert store.turn_stats(sid) == {0: first_stats, 2: stats}


def test_rewind_and_fork_follow_turn_stats(tmp_path: Path) -> None:
    at = datetime(2026, 10, 19, tzinfo=timezone.utc)
    first = TurnStats((TokenUsage("m", 1, 1),), at, at, 10, 5)
    second = TurnStats((TokenUsage("m", 2, 2),), at, at, 20, 10)
    with SessionStore(tmp_path / "s.db") as store:
        sid = new_session_id()
        store.append(sid, Message.user("first"))
        store.append(sid, Message.assistant(content="answer"))
        store.save_turn_stats(sid, 0, first)
        store.append(sid, Message.user("second"))
        store.append(sid, Message.assistant(content="again"))
        store.save_turn_stats(sid, 2, second)
        assert store.turn_stats(sid) == {0: first, 2: second}

        fork = store.fork_session(sid, through_seq=1)
        assert store.turn_stats(fork.id) == {0: first}

        store.rewind_to_user_message(sid, 2)
        assert store.turn_stats(sid) == {0: first}

        # A corrupt row is ignored on read and dropped, not copied, by a fork.
        store.append(sid, Message.user("third"))
        store._conn.execute(
            "INSERT INTO session_events (session_id, message_seq, kind, created_at, "
            "payload) VALUES (?, 2, 'turn_stats', '2026-01-01T00:00:00+00:00', ?)",
            (sid, json.dumps({"usage": "broken"})),
        )
        store._conn.commit()
        assert store.turn_stats(sid) == {0: first}
        forked = store.fork_session(sid)
        assert [e.message_seq for e in store.events(forked.id)] == [0]


def test_cli_footer_reports_the_turn_rate() -> None:
    from rich.console import Console

    from lingcore.io.cli import CLIFrontend

    at = datetime(2026, 10, 19, tzinfo=timezone.utc)
    cli = CLIFrontend(agent_name="t")
    cli.console = Console(record=True, width=200)
    usage = TokenUsage("m", 1000, 236)
    cli.render(UsageReported(usage))
    cli.render(TurnStatsReported(TurnStats((usage,), at, at, 3000, 1000)))
    cli.render(Final("done"))
    assert "236 tok/s" in cli.console.export_text()
