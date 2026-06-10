"""Session store, recording memory, resume hydration, and path policy."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile, LLMCfg
from lingcore.errors import ConfigError, SessionError
from lingcore.memory import WindowMemory
from lingcore.message import Message, ToolCall, ToolResult
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
    assert "bundled profile" in notice


def test_open_store_refuses_real_bundled_profile():
    bundled = Path(__file__).parent.parent / "lingcore" / "profiles" / "daily"
    profile = AgentProfile.load(bundled)
    store, notice = open_store(profile)
    assert store is None and "bundled profile" in notice


def test_open_store_without_source_dir():
    profile = AgentProfile(llm=LLMCfg(model="m"))
    store, notice = open_store(profile)
    assert store is None and "no source directory" in notice


def test_sessions_cfg_typo_is_loud(tmp_path: Path):
    yaml_text = PROFILE_YAML + "sessions:\n  enabld: true\n"
    with pytest.raises(ConfigError, match="invalid profile"):
        _profile(tmp_path, yaml_text)
