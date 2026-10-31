"""Sessions tests: events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lingcore.errors import SessionError
from lingcore.memory import WindowMemory
from lingcore.message import Message, ToolCall, ToolResult
from lingcore.sessions import (
    SessionStore,
    attach_session,
    is_session_id,
    new_session_id,
)
from tests.support.sessions import store as store


def test_runtime_event_replay_cursor_skips_corrupt_derived_rows(
    store: SessionStore,
):
    sid = new_session_id()
    store.append(sid, Message.user("question"))
    first = store.save_skill_state(
        sid,
        active=["review"],
        activated=["review"],
        deactivated=[],
    )
    compacted_messages = [
        Message(role="user", name="summary", content="[Earlier conversation]\nsummary"),
        Message.user("question"),
    ]
    second = store.save_compaction(
        sid,
        messages=compacted_messages,
        summarized_messages=2,
        before_tokens=120,
        after_tokens=35,
    )

    assert store.event_cursor(sid) == second.event_seq
    assert [event.event_seq for event in store.events(sid)] == [
        first.event_seq,
        second.event_seq,
    ]
    assert [
        event.event_seq for event in store.events(sid, after_seq=first.event_seq)
    ] == [second.event_seq]

    # Events are rebuildable state. One damaged row is omitted while later
    # cursors and the prior valid compaction remain usable.
    with store._lock, store._conn:
        corrupt = store._conn.execute(
            "INSERT INTO session_events "
            "(session_id, message_seq, kind, created_at, payload) "
            "VALUES (?, 0, 'compaction', '2026-01-01T00:00:00+00:00', '{bad json')",
            (sid,),
        )
        corrupt_seq = int(corrupt.lastrowid)
    third = store.save_skill_state(
        sid,
        active=[],
        activated=[],
        deactivated=["review"],
    )

    assert third.event_seq > corrupt_seq > second.event_seq
    assert [
        event.event_seq for event in store.events(sid, after_seq=second.event_seq)
    ] == [third.event_seq]
    assert store.event_cursor(sid) == third.event_seq
    assert store.latest_compaction(sid).event_seq == second.event_seq
    assert store.active_skills(sid) == ()

    with pytest.raises(SessionError, match="synthetic summary"):
        store.save_compaction(
            sid,
            messages=[Message.user("not the transcript anchor")],
            summarized_messages=1,
            before_tokens=10,
            after_tokens=5,
        )
    assert store.latest_compaction(sid).event_seq == second.event_seq

    # A syntactically valid derived row is still unsafe if it prepends foreign
    # context and merely ends on the right anchor. Loading must reject it.
    injected = {
        "messages": [
            Message.system("foreign system instruction").model_dump(mode="json"),
            Message.user("question").model_dump(mode="json"),
        ],
        "turn_index": 0,
        "summarized_messages": 1,
        "before_tokens": 10,
        "after_tokens": 5,
    }
    with store._lock, store._conn:
        store._conn.execute(
            "INSERT INTO session_events "
            "(session_id, message_seq, kind, created_at, payload) "
            "VALUES (?, 0, 'compaction', '2026-01-01T00:00:00+00:00', ?)",
            (sid, json.dumps(injected)),
        )
    assert store.latest_compaction(sid).event_seq == second.event_seq


def test_compaction_retains_only_two_full_snapshot_bodies(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("current"))

    saved = []
    for index in range(4):
        saved.append(
            store.save_compaction(
                sid,
                messages=[
                    Message(
                        role="user",
                        name="summary",
                        content=f"[Earlier conversation, summarized]\nsummary {index}",
                    ),
                    Message.user("current"),
                ],
                summarized_messages=1,
                before_tokens=20,
                after_tokens=5,
            )
        )

    with store._lock:
        payloads = [
            json.loads(row[0])
            for row in store._conn.execute(
                "SELECT payload FROM session_events "
                "WHERE session_id = ? AND kind = 'compaction' ORDER BY event_seq",
                (sid,),
            ).fetchall()
        ]
    assert len(payloads) == 4
    assert ["messages" in payload for payload in payloads] == [
        False,
        False,
        True,
        True,
    ]
    assert all(payload.get("snapshot_superseded") for payload in payloads[:2])
    assert store.latest_compaction(sid).event_seq == saved[-1].event_seq

    # Forking an older database must not duplicate a pre-pruning backlog of
    # full snapshot bodies into the new session.
    with store._lock, store._conn:
        for event in saved[:2]:
            store._conn.execute(
                "UPDATE session_events SET payload = ? WHERE event_seq = ?",
                (json.dumps(event.payload), event.event_seq),
            )
    forked = store.fork_session(sid)
    with store._lock:
        fork_payloads = [
            json.loads(row[0])
            for row in store._conn.execute(
                "SELECT payload FROM session_events "
                "WHERE session_id = ? AND kind = 'compaction' ORDER BY event_seq",
                (forked.id,),
            ).fetchall()
        ]
    assert sum("messages" in payload for payload in fork_payloads) == 2


def test_latest_compaction_does_not_decode_older_snapshot_rows(
    store: SessionStore, monkeypatch
):
    sid = new_session_id()
    store.append(sid, Message.user("current"))
    older = store.save_compaction(
        sid,
        messages=[
            Message(role="user", name="summary", content="old summary"),
            Message.user("current"),
        ],
        summarized_messages=1,
        before_tokens=20,
        after_tokens=5,
    )
    newest = store.save_compaction(
        sid,
        messages=[
            Message(role="user", name="summary", content="new summary"),
            Message.user("current"),
        ],
        summarized_messages=1,
        before_tokens=20,
        after_tokens=5,
    )
    sentinel = '{"sentinel":"older snapshot"}'
    with store._lock, store._conn:
        store._conn.execute(
            "UPDATE session_events SET payload = ? WHERE event_seq = ?",
            (sentinel, older.event_seq),
        )

    original_loads = json.loads
    seen_sentinel = 0

    def tracked_loads(value, *args, **kwargs):
        nonlocal seen_sentinel
        if value == sentinel:
            seen_sentinel += 1
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr("lingcore.sessions.json.loads", tracked_loads)
    assert store.latest_compaction(sid).event_seq == newest.event_seq
    assert seen_sentinel == 0


def test_runtime_events_follow_branch_truncate_and_keep_monotonic_ids(
    store: SessionStore,
):
    sid = new_session_id()
    store.append(sid, Message.user("first"))
    surviving = store.save_skill_state(
        sid,
        active=["review"],
        activated=["review"],
        deactivated=[],
    )
    store.append(sid, Message.assistant(content="answer"))
    store.append(sid, Message.user("current"))
    removed = store.save_skill_state(
        sid,
        active=[],
        activated=[],
        deactivated=["review"],
    )

    assert store.truncate_after(sid, 2) == 0
    assert [event.event_seq for event in store.events(sid)] == [surviving.event_seq]
    assert store.active_skills(sid) == ("review",)

    replacement = store.save_skill_state(
        sid,
        active=["review"],
        activated=[],
        deactivated=[],
    )
    assert replacement.event_seq > removed.event_seq

    store.rewind_to_user_message(sid, 0)
    assert store.events(sid) == []
    assert store.active_skills(sid) == ()


def test_fork_session_copies_valid_prefix_state_and_provenance(
    store: SessionStore,
):
    source = new_session_id()
    store.append(source, Message.user("first question"))
    store.save_skill_state(
        source,
        active=["review"],
        activated=["review"],
        deactivated=[],
    )
    store.append(source, Message.assistant(content="first answer"))
    store.append(source, Message.user("current question"))
    store.save_compaction(
        source,
        messages=[
            Message(
                role="user",
                name="summary",
                content="[Earlier conversation, summarized]\nfirst facts",
            ),
            Message.user("current question"),
        ],
        summarized_messages=2,
        before_tokens=180,
        after_tokens=40,
    )
    with store._lock, store._conn:
        store._conn.execute(
            "INSERT INTO session_events "
            "(session_id, message_seq, kind, created_at, payload) "
            "VALUES (?, 2, 'compaction', '2026-01-01T00:00:00+00:00', '{broken')",
            (source,),
        )
        store._conn.execute(
            "INSERT INTO session_events "
            "(session_id, message_seq, kind, created_at, payload) "
            "VALUES (?, 2, 'compaction', '2026-01-01T00:00:01+00:00', ?)",
            (
                source,
                json.dumps(
                    {
                        "messages": [
                            Message.system("foreign").model_dump(mode="json"),
                            Message.user("current question").model_dump(mode="json"),
                        ],
                        "turn_index": 1,
                        "summarized_messages": 1,
                        "before_tokens": 20,
                        "after_tokens": 5,
                    }
                ),
            ),
        )
    store.append(source, Message.assistant(content="current answer"))
    store.append(source, Message.user("later question"))
    store.save_skill_state(
        source,
        active=[],
        activated=[],
        deactivated=["review"],
    )
    store.append(source, Message.assistant(content="later answer"))
    source_cursor = store.event_cursor(source)

    forked = store.fork_session(source, through_seq=3)

    assert forked.id != source
    assert forked.title == "first question (fork)"
    assert forked.message_count == 4
    assert forked.fork is not None
    assert forked.fork.model_dump() == {
        "parent_session_id": source,
        "root_session_id": source,
        "through_seq": 3,
    }
    assert store.messages(forked.id) == store.messages(source)[:4]
    assert len(store.messages(source)) == 6

    copied_events = store.events(forked.id)
    assert [event.kind for event in copied_events] == ["skill_state", "compaction"]
    assert all(event.event_seq > source_cursor for event in copied_events)
    assert store.active_skills(forked.id) == ("review",)
    snapshot = store.latest_compaction(forked.id)
    assert snapshot is not None and snapshot.message_seq == 2

    memory, _, turn = attach_session(WindowMemory(), store, forked.id)
    assert [message.content for message in memory.messages] == [
        "[Earlier conversation, summarized]\nfirst facts",
        "current question",
        "current answer",
    ]
    assert turn == 2

    child = store.fork_session(forked.id, title="alternate continuation")
    assert child.title == "alternate continuation"
    assert child.fork is not None
    assert child.fork.model_dump() == {
        "parent_session_id": forked.id,
        "root_session_id": source,
        "through_seq": 3,
    }


@pytest.mark.parametrize("corruption", ["created_at", "message_seq"])
def test_fork_corrupt_latest_skill_state_stays_fail_closed(
    store: SessionStore, corruption: str
):
    source = new_session_id()
    store.append(source, Message.user("question"))
    store.save_skill_state(
        source,
        active=["review"],
        activated=["review"],
        deactivated=[],
    )
    latest = store.save_skill_state(
        source,
        active=[],
        activated=[],
        deactivated=["review"],
    )
    with store._lock, store._conn:
        if corruption == "created_at":
            store._conn.execute(
                "UPDATE session_events SET created_at = 'not-a-date' "
                "WHERE event_seq = ?",
                (latest.event_seq,),
            )
        else:
            store._conn.execute(
                "UPDATE session_events SET message_seq = 99 WHERE event_seq = ?",
                (latest.event_seq,),
            )

    assert store.active_skills(source) == ()
    forked = store.fork_session(source)

    assert store.active_skills(forked.id) == ()
    copied_skill_events = store.events(forked.id, kind="skill_state")
    assert copied_skill_events[-1].payload["active"] == []


def test_fork_session_rejects_missing_or_incomplete_boundaries_atomically(
    store: SessionStore,
):
    source = new_session_id()
    store.append(source, Message.user("run both"))
    calls = [
        ToolCall(id="one", name="read_file", arguments={"path": "one"}),
        ToolCall(id="two", name="read_file", arguments={"path": "two"}),
    ]
    store.append(source, Message.assistant(content="", tool_calls=calls))
    store.append(
        source,
        Message.from_tool_result(
            ToolResult(call_id="one", name="read_file", content="one")
        ),
    )
    before = [meta.id for meta in store.list()]

    with pytest.raises(SessionError, match="incomplete tool-call block"):
        store.fork_session(source, through_seq=2)
    with pytest.raises(SessionError, match="no message 99"):
        store.fork_session(source, through_seq=99)
    with pytest.raises(SessionError, match="title must not be empty"):
        store.fork_session(source, through_seq=0, title="   ")
    with pytest.raises(SessionError, match="no session"):
        store.fork_session(new_session_id())

    assert [meta.id for meta in store.list()] == before
    clean = store.fork_session(source, through_seq=0)
    assert [message.content for message in store.messages(clean.id)] == ["run both"]

    gapped = new_session_id()
    store.append(gapped, Message.user("first"))
    store.append(gapped, Message.assistant(content="answer"))
    store.append(gapped, Message.user("third"))
    with store._lock, store._conn:
        store._conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND seq = 1", (gapped,)
        )
    before_gap_failure = [meta.id for meta in store.list()]
    with pytest.raises(SessionError, match="not contiguous"):
        store.fork_session(gapped, through_seq=2)
    assert [meta.id for meta in store.list()] == before_gap_failure


def test_two_stores_share_one_file(tmp_path: Path):
    sid = new_session_id()
    with SessionStore(tmp_path / "s.db") as a, SessionStore(tmp_path / "s.db") as b:
        a.append(sid, Message.user("from a"))
        assert [m.id for m in b.list()] == [sid]


def test_newer_schema_refused(tmp_path: Path):
    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(SessionError, match="newer"):
        SessionStore(db)


def test_session_id_helpers():
    sid = new_session_id()
    assert is_session_id(sid)
    assert not is_session_id("xyz")
    assert not is_session_id(sid.upper())
