from __future__ import annotations

import asyncio
import sqlite3

import pytest

from lingcore._sessions.runs import RunJournal
from lingcore.delegation.config import RetentionConfig
from lingcore.delegation.models import RunRecord
from lingcore.delegation.retention import prune
from lingcore.errors import ToolError
from lingcore.message import Message
from lingcore.sessions import SessionStore
from tests.support.workflows import build, result


def test_v3_migration_preserves_transcript_and_adds_journal(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionStore(path) as store:
        store.append("1" * 32, Message(role="user", content="kept"))
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE run_nodes")
    conn.execute("DROP TABLE runs")
    conn.execute("PRAGMA user_version=3")
    conn.commit()
    conn.close()
    with SessionStore(path) as store:
        assert store.messages("1" * 32)[0].content == "kept"
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 4
        journal = RunJournal(store)
        journal.save(RunRecord(id="r-test", parent_session_id="1" * 32))
        assert journal.get("r-test").status == "pending"


async def test_background_run_cancels_when_anchor_is_edited(tmp_path):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        agent.background.enabled = True
        agent.memory.add(Message(role="user", content="original"))
        agent.memory.add(Message(role="assistant", content="starting"))
        handle = supervisor.new_run(background=True)
        started, reaped = asyncio.Event(), asyncio.Event()

        async def work():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                reaped.set()

        async with agent:
            await supervisor.execute(handle, work)
            await started.wait()
            store.truncate_after(agent._session_id, 0)
            await asyncio.wait_for(reaped.wait(), 2)
            await asyncio.gather(handle.task, return_exceptions=True)
            assert supervisor.journal.get(handle.record.id).status == "detached"
            assert handle.cancelled.is_set()


async def test_continuation_reuses_worker_session_without_expanding_tools(tmp_path):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        supervisor.llm_factory = lambda *a: result("first")
        async with agent:
            handle = supervisor.new_run()
            envelope = supervisor.envelope("general", frozenset({"read_file"}))
            report = await supervisor.execute(
                handle, lambda: supervisor.worker(handle, "first", "general", envelope)
            )
            child = result("second")
            supervisor.llm_factory = lambda *a: child
            continued = await supervisor.continue_worker(
                report.worker_id, "continue", agent._authorized_tool_names()
            )
            assert continued.session_id == report.session_id
            names = {s["function"]["name"] for s in child.tool_schemas[0]}
            assert "read_file" in names and "write_file" not in names
            assert len(child.calls[0]) > 1


async def test_retention_never_prunes_interrupted_runs(tmp_path):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        async with agent:
            for index, status in enumerate(
                ["completed", "detached", "interrupted", "running"]
            ):
                supervisor.journal.save(
                    RunRecord(
                        id=f"r-{index}",
                        status=status,
                        workspace=str(supervisor.workspace.workspace),
                    )
                )
            outcome = prune(
                supervisor.journal, supervisor.workspace, RetentionConfig(max_runs=1)
            )
            assert outcome["removed"] == ["r-0", "r-1"]
            assert {r.id for r in supervisor.journal.list()} == {"r-2", "r-3"}


async def test_fork_can_read_ancestral_runs_but_cannot_mutate(tmp_path):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        agent.memory.add(Message(role="user", content="original"))
        agent.memory.add(Message(role="assistant", content="finished"))
        handle = supervisor.new_run()
        handle.record.status = "completed"
        supervisor.journal.save(handle.record)
        fork = store.fork_session(agent._session_id)
        await agent.aclose()
        from lingcore.agent import Agent
        from tests.fakes import FakeLLMClient

        child = Agent.from_profile(
            supervisor.profile,
            llm=FakeLLMClient([]),
            session_store=store,
            session_id=fork.id,
        )
        async with child:
            assert (
                child.tool_ctx.delegation.owned_run(handle.record.id, read_only=True).id
                == handle.record.id
            )
            with pytest.raises(ToolError, match="another session"):
                child.tool_ctx.delegation.owned_run(handle.record.id)
            assert child.tool_ctx.delegation.journal.list(fork.id) == []
