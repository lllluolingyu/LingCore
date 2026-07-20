"""Session store, recording memory, resume hydration, and path policy."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile, LLMCfg
from lingcore.errors import ConfigError, SessionError
from lingcore.events import Error, Final
from lingcore.llm import LLMChunk
from lingcore.memory import WindowMemory
from lingcore.message import Attachment, Message, ToolCall, ToolResult
from lingcore.sessions import (
    SessionMemory,
    SessionStore,
    attach_session,
    is_session_id,
    new_session_id,
    open_store,
    trim_dangling,
)
from tests.fakes import FakeLLMClient, ScriptedTurn

PROFILE_YAML = """
name: testprof
workspace: .
llm:
  model: gpt-4o
tools: []
"""


def _profile(tmp_path: Path, yaml_text: str = PROFILE_YAML) -> AgentProfile:
    d = tmp_path / "prof"
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(yaml_text, encoding="utf-8")
    return AgentProfile.load(d)


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    with SessionStore(tmp_path / "sessions.db") as s:
        yield s


def _tool_block() -> list[Message]:
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})
    return [
        Message.assistant(content="", tool_calls=[call]),
        Message.from_tool_result(
            ToolResult(call_id="c1", name="read_file", content="hello")
        ),
    ]


# --------------------------------------------------------------------------- #
# SessionStore basics                                                          #
# --------------------------------------------------------------------------- #


def test_append_and_messages_roundtrip(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("question 一二三"))
    for m in _tool_block():
        store.append(sid, m)
    store.append(sid, Message.assistant(content="done"))

    msgs = store.messages(sid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[0].content == "question 一二三"
    assert msgs[1].tool_calls[0].arguments == {"path": "a.txt"}
    assert msgs[2].tool_call_id == "c1" and msgs[2].name == "read_file"


def test_message_records_keep_stable_sequences(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("one"))
    store.append(sid, Message.assistant(content="two"))

    records = store.message_records(sid)
    assert [record.seq for record in records] == [0, 1]
    assert [record.message.content for record in records] == ["one", "two"]


def test_truncate_after_keeps_current_user_message(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("keep me"))
    store.append(sid, Message.assistant(content="partial reply"))

    assert store.truncate_after(sid, 0) == 1
    assert [message.content for message in store.messages(sid)] == ["keep me"]
    assert store.truncate_after(sid, 0) == 0


def test_rewind_to_user_message_deletes_tail_and_refreshes_auto_title(
    store: SessionStore,
):
    sid = new_session_id()
    store.append(sid, Message.user("original question"))
    store.append(sid, Message.assistant(content="old answer"))

    target = store.rewind_to_user_message(sid, 0)
    assert target.content == "original question"
    assert store.messages(sid) == []
    assert store.get(sid).title == ""

    store.append(sid, Message.user("edited question"))
    assert store.message_records(sid)[0].seq == 0
    assert store.get(sid).title == "edited question"


def test_rewind_refuses_non_user_and_preserves_custom_title(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("question"))
    store.append(sid, Message.assistant(content="answer"))
    store.rename(sid, "my custom title")

    with pytest.raises(SessionError, match="ordinary user"):
        store.rewind_to_user_message(sid, 1)
    assert len(store.messages(sid)) == 2

    store.rewind_to_user_message(sid, 0)
    assert store.get(sid).title == "my custom title"


def test_attachment_fallback_text_round_trips(store: SessionStore):
    # A fallback computed under a text-only model must survive resume — the
    # conversion is paid once, not once per session load.
    import base64

    sid = new_session_id()
    att = Attachment(
        kind="image",
        media_type="image/png",
        data=base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode("ascii"),
        name="p.png",
        fallback_text="a bar chart of rent by month",
    )
    store.append(sid, Message.user("look", attachments=[att]))
    loaded = store.messages(sid)[0]
    assert loaded.attachments[0].fallback_text == "a bar chart of rent by month"
    assert loaded.attachments[0].data == att.data


def test_text_and_binary_attachments_round_trip(store: SessionStore):
    import base64

    sid = new_session_id()
    txt = Attachment(
        kind="text",
        media_type="text/x-python",
        data=base64.b64encode(b"x = 1\n").decode("ascii"),
        name="a.py",
        fallback_text="x = 1\n",
    )
    binary = Attachment(
        kind="binary",
        media_type="application/octet-stream",
        data=base64.b64encode(b"\x00\x01\x02\x03").decode("ascii"),
        name="b.bin",
        fallback_text="[binary file saved to attachments/b.bin]",
    )
    store.append(sid, Message.user("two files", attachments=[txt, binary]))
    loaded = store.messages(sid)[0]
    assert loaded.attachments[0].kind == "text"
    assert loaded.attachments[0].fallback_text == "x = 1\n"
    assert loaded.attachments[1].kind == "binary"
    assert base64.b64decode(loaded.attachments[1].data) == b"\x00\x01\x02\x03"


def test_messages_salvages_rows_with_stale_attachments(store: SessionStore):
    # A row whose attachments fail *today's* validation (e.g. written before
    # the rules tightened) must not brick the session: the text survives, the
    # media is dropped, and the loss is marked on the message.
    import json

    sid = new_session_id()
    store.append(sid, Message.user("look at this"))
    bad = {
        "role": "user",
        "content": "[media from tool results: a.png]",
        "tool_calls": [],
        "tool_call_id": None,
        "name": "media",
        "attachments": [
            # invalid base64 + spoofed type: fails Attachment validation today
            {"kind": "image", "media_type": "image/png", "data": "@@@", "name": "a.png"}
        ],
    }
    with store._lock, store._conn:
        store._conn.execute(
            "INSERT INTO messages (session_id, seq, role, created_at, payload)"
            " VALUES (?, 1, 'user', '2026-01-01T00:00:00+00:00', ?)",
            (sid, json.dumps(bad)),
        )

    msgs = store.messages(sid)
    assert len(msgs) == 2
    assert msgs[1].attachments == []
    assert "no longer pass validation" in msgs[1].content
    assert msgs[1].name == "media"


def test_messages_still_raises_on_truly_corrupt_row(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("ok"))
    with store._lock, store._conn:
        store._conn.execute(
            "INSERT INTO messages (session_id, seq, role, created_at, payload)"
            " VALUES (?, 1, 'user', '2026-01-01T00:00:00+00:00', ?)",
            (sid, "{not json"),
        )
    with pytest.raises(SessionError, match="corrupt message row"):
        store.messages(sid)


def test_messages_still_raises_on_non_list_attachments(store: SessionStore):
    import json

    sid = new_session_id()
    store.append(sid, Message.user("ok"))
    bad = {
        "role": "user",
        "content": "bad",
        "tool_calls": [],
        "tool_call_id": None,
        "name": None,
        "attachments": 1,
    }
    with store._lock, store._conn:
        store._conn.execute(
            "INSERT INTO messages (session_id, seq, role, created_at, payload)"
            " VALUES (?, 1, 'user', '2026-01-01T00:00:00+00:00', ?)",
            (sid, json.dumps(bad)),
        )
    with pytest.raises(SessionError, match="corrupt message row"):
        store.messages(sid)


def test_rows_are_lazy(store: SessionStore):
    sid = new_session_id()
    assert store.get(sid) is None
    assert store.messages(sid) == []
    assert store.list() == []
    store.append(sid, Message.user("hi"))
    meta = store.get(sid)
    assert meta is not None and meta.message_count == 1


def test_append_rejects_bad_id(store: SessionStore):
    with pytest.raises(SessionError, match="invalid session id"):
        store.append("not-an-id", Message.user("hi"))


def test_auto_title_first_user_message_only(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("  first   line\nsecond line  "))
    store.append(sid, Message.assistant(content="reply"))
    store.append(sid, Message.user("a different question"))
    assert store.get(sid).title == "first line"


def test_auto_title_truncates(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("x" * 100))
    title = store.get(sid).title
    assert len(title) == 60 and title.endswith("…")


def test_list_orders_by_activity(store: SessionStore, monkeypatch):
    import lingcore.sessions as sessions_mod

    clock = iter(f"2026-06-10T00:00:00.{i:06d}+00:00" for i in range(100))
    monkeypatch.setattr(sessions_mod, "_utcnow", lambda: next(clock))

    s1, s2 = new_session_id(), new_session_id()
    store.append(s1, Message.user("one"))
    store.append(s2, Message.user("two"))
    assert [m.id for m in store.list()] == [s2, s1]
    assert store.latest().id == s2

    store.append(s1, Message.user("one again"))
    assert [m.id for m in store.list()] == [s1, s2]
    assert store.latest().id == s1


def test_resolve_prefix(store: SessionStore):
    s1 = "aa" + "0" * 30
    s2 = "ab" + "0" * 30
    store.append(s1, Message.user("first"))
    store.append(s2, Message.user("second"))

    assert store.resolve_prefix("aa").id == s1
    assert store.resolve_prefix(s1).id == s1
    with pytest.raises(SessionError, match="ambiguous"):
        store.resolve_prefix("a")
    with pytest.raises(SessionError, match="no session matching"):
        store.resolve_prefix("ff")
    with pytest.raises(SessionError, match="invalid session id prefix"):
        store.resolve_prefix("not hex!")


def test_rename(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("hi"))
    meta = store.rename(sid, "  my   chat  ")
    assert meta.title == "my chat"
    with pytest.raises(SessionError, match="no session"):
        store.rename(new_session_id(), "nope")


def test_delete_cascades(store: SessionStore, tmp_path: Path):
    sid = new_session_id()
    store.append(sid, Message.user("hi"))
    store.append(sid, Message.assistant(content="yo"))

    assert store.delete(sid) is True
    assert store.get(sid) is None
    assert store.delete(sid) is False

    raw = sqlite3.connect(tmp_path / "sessions.db")
    (n,) = raw.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
    ).fetchone()
    raw.close()
    assert n == 0


def test_persists_across_reopen(tmp_path: Path):
    sid = new_session_id()
    with SessionStore(tmp_path / "s.db") as store:
        store.append(sid, Message.user("hello"))
    with SessionStore(tmp_path / "s.db") as store:
        assert [m.content for m in store.messages(sid)] == ["hello"]
        assert store.get(sid).title == "hello"


def test_schema_v1_migrates_without_losing_messages(tmp_path: Path):
    db = tmp_path / "s.db"
    sid = new_session_id()
    with SessionStore(db) as store:
        store.append(sid, Message.user("survives migration"))

    # Schema v1 had only sessions + messages. Recreate that on-disk shape and
    # verify opening with v2 adds the derived event log in place.
    raw = sqlite3.connect(db)
    raw.execute("DROP TABLE session_events")
    raw.execute("PRAGMA user_version = 1")
    raw.commit()
    raw.close()

    with SessionStore(db) as store:
        assert [m.content for m in store.messages(sid)] == ["survives migration"]
        event = store.save_skill_state(
            sid,
            active=["review"],
            activated=["review"],
            deactivated=[],
        )
        assert event.event_seq >= 1
        assert store.active_skills(sid) == ("review",)


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
    assert [event.event_seq for event in store.events(sid, after_seq=first.event_seq)] == [
        second.event_seq
    ]

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
    assert [event.event_seq for event in store.events(sid, after_seq=second.event_seq)] == [
        third.event_seq
    ]
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
        saved.append(store.save_compaction(
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
        ))

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
                json.dumps({
                    "messages": [
                        Message.system("foreign").model_dump(mode="json"),
                        Message.user("current question").model_dump(mode="json"),
                    ],
                    "turn_index": 1,
                    "summarized_messages": 1,
                    "before_tokens": 20,
                    "after_tokens": 5,
                }),
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


# --------------------------------------------------------------------------- #
# trim_dangling                                                                #
# --------------------------------------------------------------------------- #


def test_trim_keeps_clean_history():
    msgs = [Message.user("q"), *_tool_block(), Message.assistant(content="a")]
    assert trim_dangling(msgs) == msgs


def test_trim_keeps_trailing_user():
    msgs = [Message.user("q"), Message.assistant(content="a"), Message.user("crashed turn")]
    assert trim_dangling(msgs) == msgs


def test_trim_drops_unanswered_tool_calls():
    call = ToolCall(id="c9", name="search", arguments={})
    msgs = [Message.user("q"), Message.assistant(content="", tool_calls=[call])]
    assert trim_dangling(msgs) == [Message.user("q")]


def test_trim_drops_partially_answered_block():
    c1 = ToolCall(id="c1", name="a", arguments={})
    c2 = ToolCall(id="c2", name="b", arguments={})
    msgs = [
        Message.user("q"),
        Message.assistant(content="", tool_calls=[c1, c2]),
        Message.from_tool_result(ToolResult(call_id="c1", name="a", content="x")),
    ]
    assert trim_dangling(msgs) == [Message.user("q")]


def test_trim_drops_mid_list_dangling_block():
    # A crashed turn left a dangling block, then a *resumed* session appended
    # more turns after it. The dangling block must go even though it is not
    # at the tail.
    call = ToolCall(id="c1", name="a", arguments={})
    msgs = [
        Message.user("q1"),
        Message.assistant(content="", tool_calls=[call]),  # never answered
        Message.user("q2"),
        Message.assistant(content="fine"),
    ]
    assert trim_dangling(msgs) == [
        Message.user("q1"),
        Message.user("q2"),
        Message.assistant(content="fine"),
    ]


def test_trim_drops_orphan_tool_results():
    msgs = [
        Message.from_tool_result(ToolResult(call_id="c0", name="a", content="x")),
        Message.user("q"),
    ]
    assert trim_dangling(msgs) == [Message.user("q")]


# --------------------------------------------------------------------------- #
# SessionMemory + attach_session                                               #
# --------------------------------------------------------------------------- #


def test_session_memory_records_and_delegates(store: SessionStore):
    sid = new_session_id()
    mem = SessionMemory(WindowMemory(), store, sid)
    mem.add(Message.user("hello"))
    mem.add(Message.assistant(content="hi"))

    assert [m.role for m in mem.messages] == ["user", "assistant"]
    assert [m.role for m in store.messages(sid)] == ["user", "assistant"]
    assert mem.last_sequence == 1
    rendered = mem.render("SYS")
    assert rendered[0].role == "system" and len(rendered) == 3


def test_window_trim_does_not_shrink_storage(store: SessionStore):
    sid = new_session_id()
    mem = SessionMemory(WindowMemory(max_messages=2), store, sid)
    for i in range(6):
        mem.add(Message.user(f"m{i}"))
    assert len(mem.render("SYS")) < 7  # window applies to rendering
    assert len(store.messages(sid)) == 6  # storage keeps everything


def test_attach_session_fresh_and_resume(store: SessionStore):
    mem, sid, turn = attach_session(WindowMemory(), store, None)
    assert is_session_id(sid) and turn == 0
    assert store.get(sid) is None  # still lazy: nothing said yet

    mem.add(Message.user("q"))
    mem.add(Message.assistant(content="a"))

    mem2, sid2, turn2 = attach_session(WindowMemory(), store, sid)
    assert sid2 == sid and turn2 == 1
    assert [m.content for m in mem2.messages] == ["q", "a"]
    # replay must not have re-appended to the store
    assert len(store.messages(sid)) == 2


def test_attach_session_trims_dangling(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("q"))
    call = ToolCall(id="c1", name="a", arguments={})
    store.append(sid, Message.assistant(content="", tool_calls=[call]))

    mem, _, turn = attach_session(WindowMemory(), store, sid)
    assert [m.role for m in mem.messages] == ["user"]
    assert turn == 0
    # the store itself still holds the raw history (display keeps everything)
    assert len(store.messages(sid)) == 2


def test_attach_session_restores_latest_compaction_then_replays_raw_tail(
    store: SessionStore,
):
    sid = new_session_id()
    store.append(sid, Message.user("old question"))
    store.append(sid, Message.assistant(content="old answer"))
    store.append(sid, Message.user("current question"))
    store.save_compaction(
        sid,
        messages=[
            Message(
                role="user",
                name="summary",
                content="[Earlier conversation, summarized]\nold facts",
            ),
            Message.user("current question"),
        ],
        summarized_messages=2,
        before_tokens=200,
        after_tokens=40,
    )
    store.append(sid, Message.assistant(content="current answer"))
    store.append(sid, Message.user("tail question"))
    store.append(sid, Message.assistant(content="tail answer"))

    memory, _, turn = attach_session(WindowMemory(), store, sid)

    assert [message.content for message in memory.messages] == [
        "[Earlier conversation, summarized]\nold facts",
        "current question",
        "current answer",
        "tail question",
        "tail answer",
    ]
    assert turn == 3
    # The canonical transcript stays lossless for display and branch edits.
    assert [message.content for message in store.messages(sid)] == [
        "old question",
        "old answer",
        "current question",
        "current answer",
        "tail question",
        "tail answer",
    ]


def test_compaction_turn_index_ignores_dangling_assistant_rows(
    store: SessionStore,
):
    sid = new_session_id()
    store.append(sid, Message.user("old question"))
    store.append(
        sid,
        Message.assistant(
            tool_calls=[ToolCall(id="dead", name="search", arguments={})]
        ),
    )
    memory, _, restored_turns = attach_session(WindowMemory(), store, sid)
    assert restored_turns == 0

    # A later valid turn appends after the raw crash residue. The snapshot must
    # preserve the same validated turn count that attach_session restored.
    memory.add(Message.user("new question"))
    store.save_compaction(
        sid,
        messages=[
            Message(role="user", name="summary", content="summary"),
            Message.user("new question"),
        ],
        summarized_messages=1,
        before_tokens=20,
        after_tokens=5,
    )
    assert store.latest_compaction(sid).turn_index == 0


# --------------------------------------------------------------------------- #
# Agent integration                                                            #
# --------------------------------------------------------------------------- #


async def test_agent_records_session(tmp_path: Path):
    profile = _profile(tmp_path)
    store, notice = open_store(profile)
    assert notice is None and store is not None
    with store:
        fake = FakeLLMClient([ScriptedTurn(text="hi there")])
        agent = Agent.from_profile(
            profile, llm=fake, base_dir=tmp_path, session_store=store
        )
        async for _ in agent.run("hello"):
            pass
        sid = agent.memory.session_id
        assert agent._session_id == sid
        assert [m.role for m in store.messages(sid)] == ["user", "assistant"]
        assert store.get(sid).title == "hello"


async def test_agent_contains_session_append_failure_and_remains_usable(
    tmp_path: Path, monkeypatch
):
    profile = _profile(tmp_path)
    store, _ = open_store(profile)
    assert store is not None

    with store:
        original_append = store.append
        original_truncate = store.truncate_after
        failed = False
        rollback_failed = False

        def fail_first_assistant(session_id, message):
            nonlocal failed
            if message.role == "assistant" and not failed:
                failed = True
                raise sqlite3.OperationalError("database is locked")
            return original_append(session_id, message)

        def fail_first_rollback(session_id, seq):
            nonlocal rollback_failed
            if not rollback_failed:
                rollback_failed = True
                raise sqlite3.OperationalError("database is still locked")
            return original_truncate(session_id, seq)

        monkeypatch.setattr(store, "append", fail_first_assistant)
        monkeypatch.setattr(store, "truncate_after", fail_first_rollback)
        fake = FakeLLMClient(
            [ScriptedTurn(text="lost answer"), ScriptedTurn(text="recovered")]
        )
        agent = Agent.from_profile(
            profile, llm=fake, base_dir=tmp_path, session_store=store
        )

        events = [event async for event in agent.run("first question")]
        assert isinstance(events[-1], Error)
        assert "OperationalError: database is locked" in events[-1].message
        assert "rollback failed: OperationalError" in events[-1].message
        assert not any(isinstance(event, Final) for event in events)
        assert agent._turn_checkpoint is None
        assert agent._turn_index == 0
        sid = agent.memory.session_id
        assert [message.role for message in agent.memory.messages] == ["user"]
        assert [message.role for message in store.messages(sid)] == ["user"]

        recovered = [event async for event in agent.run("try again")]
        assert isinstance(recovered[-1], Final)
        assert recovered[-1].content == "recovered"
        assert [message.role for message in store.messages(sid)] == [
            "user",
            "user",
            "assistant",
        ]


async def test_agent_retries_failed_durable_rollback_before_next_turn(
    tmp_path: Path, monkeypatch
):
    class FailFirstPostOutput:
        def __init__(self):
            self.failed = False

        async def pre_input(self, text):
            return text

        async def post_output(self, text):
            if not self.failed:
                self.failed = True
                raise RuntimeError("post-output guard failed")
            return text

    profile = _profile(tmp_path)
    store, _ = open_store(profile)
    assert store is not None

    with store:
        original_truncate = store.truncate_after
        rollback_failed = False

        def fail_first_rollback(session_id, seq):
            nonlocal rollback_failed
            if not rollback_failed:
                rollback_failed = True
                raise sqlite3.OperationalError("database is locked")
            return original_truncate(session_id, seq)

        monkeypatch.setattr(store, "truncate_after", fail_first_rollback)
        agent = Agent.from_profile(
            profile,
            llm=FakeLLMClient(
                [ScriptedTurn(text="ghost"), ScriptedTurn(text="recovered")]
            ),
            base_dir=tmp_path,
            session_store=store,
        )
        agent.guardrail = FailFirstPostOutput()

        failed = [event async for event in agent.run("first question")]
        assert isinstance(failed[-1], Error)
        assert "rollback failed: OperationalError" in failed[-1].message
        sid = agent.memory.session_id
        assert agent._pending_turn_message_seq == 0
        assert [message.content for message in agent.memory.messages] == [
            "first question"
        ]
        assert [message.content for message in store.messages(sid)] == [
            "first question",
            "ghost",
        ]

        # The next run repairs the durable branch before it appends anything.
        recovered = [event async for event in agent.run("try again")]
        assert isinstance(recovered[-1], Final)
        assert agent._pending_turn_message_seq is None
        assert [message.content for message in store.messages(sid)] == [
            "first question",
            "try again",
            "recovered",
        ]


async def test_agent_cancellation_truncates_persisted_tool_tail(tmp_path: Path):
    profile = _profile(
        tmp_path,
        """
name: cancel-session
workspace: .
llm:
  model: gpt-4o
tools:
  - read_file
""",
    )
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    store, _ = open_store(profile)
    assert store is not None

    class ToolThenBlock:
        def __init__(self):
            self.calls = 0
            self.blocking = asyncio.Event()

        async def stream(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield LLMChunk(
                    tool_calls=[
                        ToolCall(
                            id="read",
                            name="read_file",
                            arguments={"path": "a.txt"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
                return
            self.blocking.set()
            await asyncio.Event().wait()
            yield LLMChunk(finish_reason="stop")

    with store:
        llm = ToolThenBlock()
        agent = Agent.from_profile(
            profile,
            llm=llm,
            base_dir=tmp_path,
            session_store=store,
        )

        async def drive():
            async for _ in agent.run("read it"):
                pass

        task = asyncio.create_task(drive())
        await llm.blocking.wait()
        sid = agent.memory.session_id
        assert [message.role for message in store.messages(sid)] == [
            "user",
            "assistant",
            "tool",
        ]
        assert agent.cancel_turn() is True
        with pytest.raises(asyncio.CancelledError):
            await task
        agent.finalize_cancelled_turn()

        assert [message.role for message in agent.memory.messages] == ["user"]
        assert [message.role for message in store.messages(sid)] == ["user"]
        resumed, _, turns = attach_session(WindowMemory(), store, sid)
        assert [message.role for message in resumed.messages] == ["user"]
        assert turns == 0


async def test_agent_resume_sees_history_and_turn_index(tmp_path: Path):
    profile = _profile(tmp_path)
    store, _ = open_store(profile)
    with store:
        fake1 = FakeLLMClient([ScriptedTurn(text="answer one")])
        a1 = Agent.from_profile(
            profile, llm=fake1, base_dir=tmp_path, session_store=store
        )
        async for _ in a1.run("first question"):
            pass
        sid = a1.memory.session_id

        fake2 = FakeLLMClient([ScriptedTurn(text="answer two")])
        a2 = Agent.from_profile(
            profile, llm=fake2, base_dir=tmp_path, session_store=store, session_id=sid
        )
        assert a2._turn_index == 1
        async for _ in a2.run("second question"):
            pass

        contents = [m.content for m in fake2.calls[0]]
        assert "first question" in contents and "answer one" in contents
        assert [m.role for m in store.messages(sid)] == [
            "user", "assistant", "user", "assistant",
        ]


async def test_agent_resume_excludes_dangling_block(tmp_path: Path):
    profile = _profile(tmp_path)
    store, _ = open_store(profile)
    with store:
        sid = new_session_id()
        store.append(sid, Message.user("q"))
        store.append(sid, Message.assistant(content="ok"))
        call = ToolCall(id="dead", name="search", arguments={})
        store.append(sid, Message.assistant(content="", tool_calls=[call]))

        fake = FakeLLMClient([ScriptedTurn(text="recovered")])
        agent = Agent.from_profile(
            profile, llm=fake, base_dir=tmp_path, session_store=store, session_id=sid
        )
        async for _ in agent.run("again"):
            pass
        assert not any(m.tool_calls for m in fake.calls[0])


async def test_agent_without_store_unchanged(tmp_path: Path):
    profile = _profile(tmp_path)
    fake = FakeLLMClient([ScriptedTurn(text="plain")])
    agent = Agent.from_profile(profile, llm=fake, base_dir=tmp_path)
    async for _ in agent.run("hi"):
        pass
    assert isinstance(agent.memory, WindowMemory)
    assert agent._session_id is None


async def test_agent_compaction_snapshot_survives_rebuild(tmp_path: Path):
    profile = _profile(
        tmp_path,
        PROFILE_YAML
        + "memory:\n"
        "  max_messages: 50\n"
        "  max_tokens: 120\n"
        "  compaction:\n"
        "    enabled: true\n"
        "    compact_at_ratio: 0.1\n"
        "    keep_recent_ratio: 0.05\n"
        "    max_summary_chars: 1000\n",
    )
    store, _ = open_store(profile)
    assert store is not None
    with store:
        fake = FakeLLMClient([
            ScriptedTurn(text="first answer"),
            ScriptedTurn(text="durable compact summary"),
            ScriptedTurn(text="second answer"),
        ])
        first = Agent.from_profile(
            profile, llm=fake, base_dir=tmp_path, session_store=store
        )
        async for _ in first.run("old " + ("padding " * 40)):
            pass
        async for _ in first.run("current question"):
            pass
        sid = first.memory.session_id

        snapshot = store.latest_compaction(sid)
        assert snapshot is not None
        assert snapshot.summarized_messages >= 1
        assert snapshot.before_tokens > snapshot.after_tokens
        assert any("durable compact summary" in m.content for m in snapshot.messages)

        resumed = Agent.from_profile(
            profile,
            llm=FakeLLMClient([ScriptedTurn(text="unused")]),
            base_dir=tmp_path,
            session_store=store,
            session_id=sid,
        )
        working = [message.content for message in resumed.memory.messages]
        assert any("durable compact summary" in content for content in working)
        assert not any("padding padding" in content for content in working)
        assert "second answer" in working
        assert resumed._turn_index == 2
        assert any("padding padding" in message.content for message in store.messages(sid))


async def test_agent_dynamic_skill_state_survives_rebuild(tmp_path: Path):
    profile = _profile(
        tmp_path,
        """
name: skill-session
workspace: .
llm:
  model: gpt-4o
tools:
  - activate_skill
  - read_file
  - search
initial_tools:
  - activate_skill
""",
    )
    store, _ = open_store(profile)
    assert store is not None
    with store:
        activate = ToolCall(
            id="activate-review",
            name="activate_skill",
            arguments={"name": "code-review"},
        )
        first_llm = FakeLLMClient([
            ScriptedTurn(tool_calls=[activate], finish_reason="tool_calls"),
            ScriptedTurn(text="activated"),
        ])
        first = Agent.from_profile(
            profile, llm=first_llm, base_dir=tmp_path, session_store=store
        )
        async for _ in first.run("review this"):
            pass
        sid = first.memory.session_id
        assert store.active_skills(sid) == ("code-review",)

        second_llm = FakeLLMClient([ScriptedTurn(text="still active")])
        resumed = Agent.from_profile(
            profile,
            llm=second_llm,
            base_dir=tmp_path,
            session_store=store,
            session_id=sid,
        )
        assert resumed.skill_state is not None
        assert resumed.skill_state.active == ["code-review"]
        async for _ in resumed.run("continue"):
            pass

        system = second_llm.calls[0][0].content
        assert "Group findings by severity" in system
        names = {
            schema["function"]["name"]
            for schema in (second_llm.tool_schemas[0] or [])
        }
        assert {"activate_skill", "read_file", "search"} <= names


def test_skill_restore_refuses_unapproved_high_risk_grant_drift(tmp_path: Path):
    profile = _profile(
        tmp_path,
        """
name: expanded-skill-session
workspace: .
llm:
  model: gpt-4o
tools:
  - activate_skill
  - read_file
  - search
  - run_shell
initial_tools:
  - activate_skill
""",
    )
    store, _ = open_store(profile)
    assert store is not None
    with store:
        sid = new_session_id()
        store.append(sid, Message.user("activated under an older safe ceiling"))
        # Legacy/safe activation: code-review previously received read/search,
        # but no high-risk grant was approved.
        store.save_skill_state(
            sid,
            active=["code-review"],
            activated=["code-review"],
            deactivated=[],
        )

        resumed = Agent.from_profile(
            profile,
            llm=FakeLLMClient([ScriptedTurn(text="unused")]),
            base_dir=tmp_path,
            session_store=store,
            session_id=sid,
        )
        assert resumed.skill_state is not None
        assert resumed.skill_state.active == []
        assert "run_shell" not in resumed._authorized_tool_names()

        # Once that exact risky grant has recorded consent, durable restoration
        # is allowed without widening it further.
        store.save_skill_state(
            sid,
            active=["code-review"],
            activated=["code-review"],
            deactivated=[],
            approved_high_risk={"code-review": ["run_shell"]},
        )
        approved = Agent.from_profile(
            profile,
            llm=FakeLLMClient([ScriptedTurn(text="unused")]),
            base_dir=tmp_path,
            session_store=store,
            session_id=sid,
        )
        assert approved.skill_state is not None
        assert approved.skill_state.active == ["code-review"]
        assert "run_shell" in approved._authorized_tool_names()


def test_corrupt_latest_skill_state_fails_closed(store: SessionStore):
    sid = new_session_id()
    store.append(sid, Message.user("question"))
    store.save_skill_state(
        sid,
        active=["review"],
        activated=["review"],
        deactivated=[],
    )
    with store._lock, store._conn:
        corrupt = store._conn.execute(
            "INSERT INTO session_events "
            "(session_id, message_seq, kind, created_at, payload) "
            "VALUES (?, 0, 'skill_state', '2026-01-01T00:00:00+00:00', '{bad')",
            (sid,),
        )
    assert store.active_skills(sid) == ()

    # A syntactically valid activation without its claimed message anchor is
    # equally untrusted and must not restore authorization.
    orphaned = {
        "active": ["review"],
        "activated": ["review"],
        "deactivated": [],
        "approved_high_risk": {},
    }
    with store._lock, store._conn:
        store._conn.execute(
            "UPDATE session_events SET message_seq = 99, payload = ? "
            "WHERE event_seq = ?",
            (json.dumps(orphaned), int(corrupt.lastrowid)),
        )
    assert store.active_skills(sid) == ()


# --------------------------------------------------------------------------- #
# open_store path policy                                                       #
# --------------------------------------------------------------------------- #


def test_open_store_default_path(tmp_path: Path):
    profile = _profile(tmp_path)
    store, notice = open_store(profile)
    assert notice is None
    with store:
        assert store.db_path == (tmp_path / "prof" / "sessions.db").resolve()


def test_open_store_disabled(tmp_path: Path):
    yaml_text = PROFILE_YAML + "sessions:\n  enabled: false\n"
    store, notice = open_store(_profile(tmp_path, yaml_text))
    assert store is None and notice is None


def test_open_store_rejects_escape(tmp_path: Path):
    yaml_text = PROFILE_YAML + "sessions:\n  path: ../outside.db\n"
    with pytest.raises(ConfigError, match="escapes profile directory"):
        open_store(_profile(tmp_path, yaml_text))


def test_open_store_rejects_bare_absolute(tmp_path: Path):
    yaml_text = PROFILE_YAML + f"sessions:\n  path: {tmp_path / 'abs.db'}\n"
    with pytest.raises(ConfigError, match="allow_absolute_path"):
        open_store(_profile(tmp_path, yaml_text))


def test_open_store_absolute_with_flag(tmp_path: Path):
    yaml_text = PROFILE_YAML + (
        f"sessions:\n  path: {tmp_path / 'elsewhere' / 'abs.db'}\n"
        "  allow_absolute_path: true\n"
    )
    store, notice = open_store(_profile(tmp_path, yaml_text))
    assert notice is None
    with store:
        sid = new_session_id()
        store.append(sid, Message.user("hi"))
        assert (tmp_path / "elsewhere" / "abs.db").is_file()


def test_open_store_refuses_package_tree(tmp_path: Path, monkeypatch):
    import lingcore.sessions as sessions_mod

    monkeypatch.setattr(sessions_mod, "_PACKAGE_DIR", tmp_path.resolve())
    store, notice = open_store(_profile(tmp_path))
    assert store is None
    assert "inside the installed" in notice


def test_bundled_profiles_live_outside_package_and_persist(tmp_path: Path):
    import lingcore.sessions as sessions_mod

    repo_profiles = Path(__file__).parent.parent / "profiles"
    assert not repo_profiles.resolve().is_relative_to(sessions_mod._PACKAGE_DIR)
    copy = tmp_path / "daily"
    shutil.copytree(repo_profiles / "daily", copy)
    store, notice = open_store(AgentProfile.load(copy))
    assert store is not None and notice is None
    store.close()


def test_open_store_without_source_dir():
    profile = AgentProfile(llm=LLMCfg(model="m"))
    store, notice = open_store(profile)
    assert store is None and "no source directory" in notice


def test_sessions_cfg_typo_is_loud(tmp_path: Path):
    yaml_text = PROFILE_YAML + "sessions:\n  enabld: true\n"
    with pytest.raises(ConfigError, match="invalid profile"):
        _profile(tmp_path, yaml_text)
