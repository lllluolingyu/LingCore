"""Sessions tests: core behavior."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lingcore.errors import SessionError
from lingcore.message import Attachment, Message
from lingcore.sessions import (
    SessionStore,
    new_session_id,
)
from tests.support.sessions import _profile as _profile
from tests.support.sessions import _tool_block
from tests.support.sessions import store as store


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
