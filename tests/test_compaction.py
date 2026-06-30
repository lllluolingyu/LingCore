"""Tests for SummarizingMemory (compaction) and the loop's Compacted event.

Compaction summarizes the oldest history into one note when the window is
nearly full, keeping the recent tail verbatim; if the result is still over the
hard cap, the window's eviction floor (the "50% eviction") clamps it.
"""

from __future__ import annotations

from lingcore.events import Compacted
from lingcore.memory import SummarizingMemory, WindowMemory
from lingcore.message import Message
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure


def _fill(window: WindowMemory, n: int) -> None:
    for i in range(n):
        window.add(Message.user(("lorem ipsum " * 10) + f"#{i}"))


async def test_no_compaction_below_threshold():
    w = WindowMemory(max_tokens=10_000, model="gpt-4o")
    sm = SummarizingMemory(w, FakeLLMClient([ScriptedTurn(text="S")]))
    _fill(w, 3)  # tiny next to a 10k budget
    assert await sm.maybe_compact() is None
    assert all(m.name != "summary" for m in w.messages)


async def test_compaction_summarizes_old_head():
    w = WindowMemory(max_tokens=200, model="gpt-4o")
    summ = FakeLLMClient([ScriptedTurn(text="COMPACT SUMMARY")])
    sm = SummarizingMemory(
        w, summ, compact_at_ratio=0.5, keep_recent_ratio=0.25, max_summary_chars=1000
    )
    _fill(w, 30)  # well over 200 tokens
    ev = await sm.maybe_compact()
    assert isinstance(ev, Compacted)
    assert ev.after_tokens < ev.before_tokens
    assert ev.summarized_messages > 0
    msgs = w.messages
    assert msgs[0].name == "summary" and "COMPACT SUMMARY" in msgs[0].content
    assert summ.calls, "the summarizer was invoked"
    # recent tail is kept verbatim after the summary
    assert msgs[-1].content.endswith("#29")


async def test_compaction_failure_falls_back_without_crashing():
    w = WindowMemory(max_tokens=200, model="gpt-4o")
    summ = FakeLLMClient([StreamFailure(text="", reason="boom", retryable=True)])
    sm = SummarizingMemory(w, summ, compact_at_ratio=0.5, keep_recent_ratio=0.25)
    _fill(w, 30)
    before = list(w.messages)
    assert await sm.maybe_compact() is None  # summarizer failed → no-op
    assert w.messages == before  # untouched; render-time eviction still guards


async def test_compaction_still_too_long_triggers_eviction():
    # Pathological: a huge summary + tail still exceed the cap, so render's floor
    # (the 50% eviction) clamps it. Exercises the compact-then-evict composition.
    w = WindowMemory(max_tokens=200, evict_to_ratio=0.5, model="gpt-4o")
    summ = FakeLLMClient([ScriptedTurn(text="word " * 300)])  # ~300-token summary
    sm = SummarizingMemory(
        w, summ, compact_at_ratio=0.5, keep_recent_ratio=0.45, max_summary_chars=4000
    )
    _fill(w, 40)
    assert await sm.maybe_compact() is not None
    rendered = sm.render("SYS")
    assert w._floor > 0  # eviction fired after compaction
    toks = sum(w._tokens(m) for m in rendered if m.role != "system")
    assert toks <= 200  # hard cap respected


async def test_compaction_trigger_counts_system_prompt():
    # Messages alone stay under the trigger; a big system prompt pushes the true
    # footprint over it, so compaction must still fire (the eviction floor counts
    # the system prompt, so the compaction trigger must too — else a large
    # system prompt would let eviction win before compaction ever ran).
    w = WindowMemory(max_tokens=200, model="gpt-4o")
    summ = FakeLLMClient([ScriptedTurn(text="SUMMARY")])
    sm = SummarizingMemory(w, summ, compact_at_ratio=0.5, keep_recent_ratio=0.1)
    for i in range(8):
        w.add(Message.user(f"short message {i}"))
    assert await sm.maybe_compact() is None  # messages alone: under the trigger
    assert await sm.maybe_compact("pad " * 90) is not None  # + big system → fires


async def test_agent_run_emits_compacted_event(tmp_path):
    from lingcore.agent import Agent
    from lingcore.composer import StaticComposer
    from lingcore.tools import ToolContext, ToolRegistry

    w = WindowMemory(max_tokens=200, evict_to_ratio=0.5, model="gpt-4o")
    _fill(w, 30)  # pre-load the window past the compaction threshold
    summarizer = FakeLLMClient([ScriptedTurn(text="SUMMARY OF OLD STUFF")])
    sm = SummarizingMemory(
        w, summarizer, compact_at_ratio=0.5, keep_recent_ratio=0.25
    )
    main_llm = FakeLLMClient([ScriptedTurn(text="done")])
    agent = Agent(
        llm=main_llm,
        tools=ToolRegistry(),
        tool_ctx=ToolContext(workspace=tmp_path),
        composer=StaticComposer("SYS"),
        memory=sm,
    )
    events = [e async for e in agent.run("the next question")]
    assert any(isinstance(e, Compacted) for e in events)
    # the turn still completed normally after compacting
    from lingcore.events import Final

    assert isinstance(events[-1], Final) and events[-1].content == "done"
