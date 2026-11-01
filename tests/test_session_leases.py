from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3
import time
from contextlib import contextmanager

import pytest

from lingcore.agent import Agent
from lingcore.errors import SessionError
from lingcore.events import Error
from lingcore.llm import LLMChunk
from lingcore.memory import WindowMemory
from lingcore.message import Message
from lingcore.sessions import SessionStore, attach_session, new_session_id
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.test_sessions import _profile


def expire(path, sid):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE session_leases SET heartbeat=0 WHERE session_id=?", (sid,)
        )


def worker(path, sid, pipe):
    with SessionStore(path) as store:
        pipe.send("ready")
        pipe.recv()
        try:
            memory, _, _ = attach_session(WindowMemory(), store, sid)
        except SessionError:
            pipe.send("busy")
            return
        pipe.send("owned")
        while True:
            command = pipe.recv()
            if command == "close":
                return
            try:
                if command == "append":
                    memory.add(Message.user("worker"))
                elif command == "truncate":
                    store.truncate_after(sid, 0, lease=memory.lease)
                elif command == "edit":
                    store.rewind_to_user_message(sid, 0, lease=memory.lease)
                elif command == "event":
                    store.append_event(
                        sid,
                        message_seq=0,
                        kind="todo_state",
                        payload={},
                        lease=memory.lease,
                    )
                elif command == "delete":
                    store.delete(sid, lease=memory.lease)
                elif command == "heartbeat":
                    store.heartbeat_session(memory.lease)
                elif command == "release":
                    memory.close()
                pipe.send("ok")
            except SessionError:
                pipe.send("lost")


@contextmanager
def process(path, sid):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    task = ctx.Process(target=worker, args=(path, sid, child))
    task.start()
    child.close()
    try:
        assert parent.poll(10)
        assert parent.recv() == "ready"
        yield task, parent
    finally:
        if task.is_alive():
            try:
                parent.send("close")
            except BrokenPipeError:
                # The busy worker may close its pipe before is_alive() observes
                # process exit. Joining below still owns its cleanup.
                pass
        task.join(3)
        if task.is_alive():
            task.terminate()
            task.join(3)
        parent.close()


def ask(pipe, command):
    pipe.send(command)
    assert pipe.poll(10), command
    return pipe.recv()


def test_simultaneous_process_attach_has_one_owner(tmp_path):
    path, sid = tmp_path / "sessions.db", new_session_id()
    with process(path, sid) as (_, one), process(path, sid) as (_, two):
        one.send("attach")
        two.send("attach")
        assert one.poll(10) and two.poll(10)
        outcomes = [one.recv(), two.recv()]
        assert sorted(outcomes) == ["busy", "owned"]
        assert ask([one, two][outcomes.index("owned")], "append") == "ok"
    with SessionStore(path) as store:
        assert [m.content for m in store.messages(sid)] == ["worker"]


def test_takeover_fences_other_process_and_stale_release(tmp_path):
    path, sid = tmp_path / "sessions.db", new_session_id()
    with process(path, sid) as (_, pipe), SessionStore(path) as store:
        assert ask(pipe, "attach") == "owned"
        assert ask(pipe, "append") == "ok"
        with pytest.raises(SessionError, match="fork_session"):
            attach_session(WindowMemory(), store, sid)
        for operation in (
            lambda: store.append(sid, Message.user("intruder")),
            lambda: store.truncate_after(sid, -1),
            lambda: store.rewind_to_user_message(sid, 0),
            lambda: store.delete(sid),
            lambda: store.rename(sid, "intruder"),
            lambda: store.discard_events_from_message_seq(sid, 0),
        ):
            with pytest.raises(SessionError, match="another agent"):
                operation()
        # A read-only fork of a complete prefix remains available while owned.
        child = store.fork_session(sid)
        assert child.id != sid
        expire(path, sid)
        memory, _, _ = attach_session(WindowMemory(), store, sid)
        assert memory.lease.epoch == 2
        memory.add(Message.assistant(content="replacement"))
        before = store.message_records(sid)
        for command in ("append", "truncate", "edit", "event", "delete", "heartbeat"):
            assert ask(pipe, command) == "lost"
        assert ask(pipe, "release") == "ok"
        store.heartbeat_session(memory.lease)
        assert store.message_records(sid) == before


def test_crashed_process_requires_expiry_then_allows_recovery(tmp_path):
    path, sid = tmp_path / "sessions.db", new_session_id()
    with process(path, sid) as (task, pipe):
        assert ask(pipe, "attach") == "owned"
        assert ask(pipe, "append") == "ok"
        task.terminate()
        task.join(3)
    with SessionStore(path) as store:
        with pytest.raises(SessionError, match="already attached"):
            store.acquire_session(sid)
        expire(path, sid)
        memory, _, _ = attach_session(WindowMemory(), store, sid)
        assert [m.content for m in memory.messages] == ["worker"]
        assert memory.lease.epoch == 2


def test_expired_lease_cannot_renew_or_mutate_without_takeover(tmp_path):
    path, sid = tmp_path / "sessions.db", new_session_id()
    with SessionStore(path) as store:
        memory, _, _ = attach_session(WindowMemory(), store, sid)
        expire(path, sid)
        with pytest.raises(SessionError, match="lease was lost"):
            store.heartbeat_session(memory.lease)
        with pytest.raises(SessionError, match="lease was lost"):
            memory.add(Message.user("must not reach either history"))
        assert not memory.messages and store.get(sid) is None


def test_idle_owner_heartbeat_and_lazy_session(tmp_path, monkeypatch):
    import lingcore.session_leases as leases

    monkeypatch.setattr(leases, "HEARTBEAT_SECONDS", 0.01)
    path, sid = tmp_path / "sessions.db", new_session_id()
    with SessionStore(path) as store:
        store.acquire_session(sid)
        with sqlite3.connect(path) as raw:
            first = raw.execute("SELECT heartbeat FROM session_leases").fetchone()[0]
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if (
                    raw.execute("SELECT heartbeat FROM session_leases").fetchone()[0]
                    > first
                ):
                    break
                time.sleep(0.01)
            else:
                pytest.fail("idle lease was not renewed")
        assert store.list() == []
    assert not store._leases.thread.is_alive()
    with SessionStore(path) as replacement:
        assert replacement.acquire_session(sid).epoch == 2


async def test_edit_rebuild_fences_old_agent_and_its_late_close(tmp_path):
    profile = _profile(tmp_path)
    with SessionStore(tmp_path / "sessions.db") as store:
        old = Agent.from_profile(
            profile, llm=FakeLLMClient([ScriptedTurn(text="old")]), session_store=store
        )
        async for _ in old.run("question"):
            pass
        sid = old.memory.session_id
        store.rewind_to_user_message(sid, 0)
        new = Agent.from_profile(
            profile,
            llm=FakeLLMClient([ScriptedTurn(text="new")]),
            session_store=store,
            session_id=sid,
        )
        assert new._session_lease.epoch > old._session_lease.epoch
        events = [event async for event in old.run("stale")]
        assert len(events) == 1 and isinstance(events[0], Error)
        assert "lease was lost" in events[0].message
        await old.aclose()
        async for _ in new.run("edited"):
            pass
        assert [m.content for m in store.messages(sid)] == ["edited", "new"]
        await new.aclose()


def test_failed_hydration_releases_lease(tmp_path, monkeypatch):
    with SessionStore(tmp_path / "sessions.db") as store:
        sid = new_session_id()

        def fail(*args):
            raise RuntimeError("bad hydration")

        monkeypatch.setattr(store, "latest_compaction", fail)
        with pytest.raises(RuntimeError):
            attach_session(WindowMemory(), store, sid)
        assert store.acquire_session(sid).epoch == 2


@pytest.mark.parametrize("stale", [False, True])
def test_failed_release_stops_renewal_without_forgetting_replacement(tmp_path, stale):
    path, sid = tmp_path / "sessions.db", new_session_id()
    with SessionStore(path) as store, sqlite3.connect(path) as blocker:
        lease = store.acquire_session(sid)
        if stale:
            expire(path, sid)
            replacement = store.acquire_session(sid)
        store._conn.execute("PRAGMA busy_timeout=1")
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                store.release_session(lease)
        finally:
            blocker.rollback()
        if stale:
            assert store._leases.owned[sid] == replacement
            store.heartbeat_session(replacement)
        else:
            assert sid not in store._leases.owned
            expire(path, sid)
            assert store.acquire_session(sid).epoch == lease.epoch + 1


@pytest.mark.parametrize("event", ["compaction", "skill", "todo"])
def test_state_save_checks_lease_before_reading_replaced_transcript(tmp_path, event):
    path, sid = tmp_path / "sessions.db", new_session_id()
    with SessionStore(path) as old, SessionStore(path) as replacement:
        lease = old.acquire_session(sid)
        old.append(sid, Message.user("old"), lease=lease)
        expire(path, sid)
        current = replacement.acquire_session(sid)
        replacement.truncate_after(sid, -1, lease=current)
        with pytest.raises(SessionError, match="lease was lost"):
            if event == "compaction":
                old.save_compaction(
                    sid,
                    messages=[Message(role="user", name="summary", content="summary")],
                    summarized_messages=1,
                    before_tokens=10,
                    after_tokens=1,
                    lease=lease,
                )
            elif event == "skill":
                old.save_skill_state(
                    sid, active=[], activated=[], deactivated=[], lease=lease
                )
            else:
                old.save_todo_state(sid, [], lease=lease)
        assert replacement.message_records(sid) == []
        assert replacement.events(sid) == []


async def test_stop_after_takeover_preserves_replacement_messages(tmp_path):
    entered = asyncio.Event()

    class WaitingClient:
        async def stream(self, messages, tools=None):
            entered.set()
            await asyncio.Event().wait()
            yield LLMChunk(text_delta="unreachable")

    profile = _profile(tmp_path)
    path = tmp_path / "sessions.db"
    with SessionStore(path) as first, SessionStore(path) as second:
        old = Agent.from_profile(profile, llm=WaitingClient(), session_store=first)

        async def drive():
            async for _ in old.run("old question"):
                pass

        task = asyncio.create_task(drive())
        await asyncio.wait_for(entered.wait(), 5)
        sid = old.memory.session_id
        expire(path, sid)
        new = Agent.from_profile(
            profile,
            llm=FakeLLMClient([ScriptedTurn(text="new reply")]),
            session_store=second,
            session_id=sid,
        )
        async for _ in new.run("new question"):
            pass
        saved = second.message_records(sid)
        assert old.cancel_turn()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = old.finalize_cancelled_turn()
        assert isinstance(result, Error) and "lease was lost" in result.message
        assert first.message_records(sid) == saved
        await old.aclose()
        second.heartbeat_session(new._session_lease)
        await new.aclose()
