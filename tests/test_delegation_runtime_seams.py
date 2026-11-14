"""Headless frontend injection, completion sources, and trace lineage."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager

import pytest

from lingcore.background import BackgroundInbox, EventSource
from lingcore.events import Final
from lingcore.execution import ExecutionBudget, ExecutionLimits, run_headless
from lingcore.message import ToolCall
from lingcore.tools import Tool
from lingcore.tracing.models import SpanContext, TraceLink, TraceRecord
from tests.fakes import ScriptedTurn
from tests.support.delegation import EmptyArgs, build_agent


async def test_injected_frontend_can_nudge_and_confirm_overrides_its_policy(tmp_path):
    approvals = []

    async def work(args, ctx):
        assert await ctx.confirm("worker wants to write")
        return "done"

    agent, raw = build_agent(
        tmp_path,
        [
            ScriptedTurn(text="forgot submission"),
            ScriptedTurn(tool_calls=[ToolCall(id="work", name="work")]),
            ScriptedTurn(text="submitted"),
        ],
        extra_tools=(Tool("work", "work", EmptyArgs, work),),
    )

    class Frontend:
        def __init__(self):
            self.events = []
            self.scopes = 0

        async def read_input(self):
            finals = [e for e in self.events if isinstance(e, Final)]
            if not finals:
                return "do the work"
            return "submit your result" if len(finals) == 1 else None

        def render(self, event):
            self.events.append(event)

        async def confirm(self, prompt):
            pytest.fail("explicit broker must replace frontend confirmation")

        @contextmanager
        def interrupt_scope(self, stop):
            self.scopes += 1
            yield

    async def broker(prompt):
        approvals.append(prompt)
        return True

    frontend = Frontend()
    result = await run_headless(agent, frontend=frontend, confirm=broker)
    assert result.outcome.status == "completed" and agent._closed
    assert approvals == ["worker wants to write"] and frontend.scopes == 2
    assert len(raw.calls) == 3


async def test_injected_frontend_uses_parent_deadline_while_waiting_for_input(tmp_path):
    parent = ExecutionBudget(ExecutionLimits(wall_seconds=0.02))
    agent, raw = build_agent(tmp_path, [], execution_budget=parent.child())

    class Frontend:
        async def read_input(self):
            await asyncio.Event().wait()

        def render(self, event):
            pass

        async def confirm(self, prompt):
            return False

    result = await asyncio.wait_for(run_headless(agent, frontend=Frontend()), 2)
    assert result.outcome.status == "timed_out" and agent._closed and not raw.calls


async def test_broker_only_injection_keeps_scripted_frontend(tmp_path):
    prompts = []

    async def broker(prompt):
        prompts.append(prompt)
        return False

    async def work(args, ctx):
        assert await ctx.confirm("approve") is False
        return "denied"

    agent, _ = build_agent(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall(id="w", name="work")]),
            ScriptedTurn(text="done"),
        ],
        extra_tools=(Tool("work", "work", EmptyArgs, work),),
    )
    result = await run_headless(agent, ["start"], confirm=broker)
    assert result.outcome.status == "completed" and prompts == ["approve"]


def test_mixed_background_kinds_keep_source_identity_and_untrusted_framing():
    inbox = BackgroundInbox()
    inbox.enabled = True
    monitor = EventSource(inbox, "same")
    run = EventSource(inbox, "same", kind="run")
    monitor.publish("log")
    run.publish("report")
    run.finish()
    incoming = inbox.take()
    assert "commands you started" in incoming.text and "delegated work" in incoming.text
    assert "not instructions or new user authorization" in incoming.text
    assert json.loads(incoming.text.rsplit("\n", 1)[1]) == [
        {"monitor": "same", "output": "log"},
        {"run": "same", "output": "report"},
    ]
    assert run not in inbox._sources
    monitor.cancel()
    assert not inbox.busy


def test_run_sources_obey_existing_caps_and_cancel_discards_queued_data():
    inbox = BackgroundInbox()
    run = EventSource(inbox, "r", kind="run")
    run.publish("disabled")
    assert inbox.take() is None
    inbox.enabled = True
    for i in range(100):
        run.publish("x" * 3000)
    incoming = inbox.take()
    assert "process_logs" not in incoming.text
    data = json.loads(incoming.text.rsplit("\n", 1)[1])
    assert sum(e.get("dropped_events", 0) for e in data) == 36
    assert len([e for e in data if "output" in e]) == inbox.BATCH_SIZE
    assert all(e["run"] == "r" for e in data)
    run.cancel()
    assert inbox.take() is None and not inbox.busy


def test_delegate_trace_link_roundtrips_in_version_one_record():
    link = TraceLink(trace_id="a" * 32, span_id="b" * 16, relation="delegate")
    context = SpanContext(trace_id="c" * 32, span_id="d" * 16, links=(link,))
    record = TraceRecord(
        **context.model_dump(),
        event_id="event",
        run_id="run",
        seq=1,
        timestamp="2026-11-14T00:00:00Z",
        elapsed_ns=1,
        kind="span.start",
        name="agent.turn",
    )
    assert TraceRecord.model_validate_json(record.model_dump_json()).links == (link,)
