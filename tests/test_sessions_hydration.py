"""Sessions tests: hydration."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.events import Error, Final
from lingcore.llm import LLMChunk
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
from tests.support.sessions import PROFILE_YAML, _profile, _tool_block
from tests.support.sessions import store as store


def test_trim_keeps_clean_history():
    msgs = [Message.user("q"), *_tool_block(), Message.assistant(content="a")]
    assert trim_dangling(msgs) == msgs


def test_trim_keeps_trailing_user():
    msgs = [
        Message.user("q"),
        Message.assistant(content="a"),
        Message.user("crashed turn"),
    ]
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

    mem.close()
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

        def fail_first_assistant(session_id, message, **kwargs):
            nonlocal failed
            if message.role == "assistant" and not failed:
                failed = True
                raise sqlite3.OperationalError("database is locked")
            return original_append(session_id, message, **kwargs)

        def fail_first_rollback(session_id, seq, **kwargs):
            nonlocal rollback_failed
            if not rollback_failed:
                rollback_failed = True
                raise sqlite3.OperationalError("database is still locked")
            return original_truncate(session_id, seq, **kwargs)

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

        def fail_first_rollback(session_id, seq, **kwargs):
            nonlocal rollback_failed
            if not rollback_failed:
                rollback_failed = True
                raise sqlite3.OperationalError("database is locked")
            return original_truncate(session_id, seq, **kwargs)

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
        await agent.aclose()
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
        await a1.aclose()
        a2 = Agent.from_profile(
            profile, llm=fake2, base_dir=tmp_path, session_store=store, session_id=sid
        )
        assert a2._turn_index == 1
        async for _ in a2.run("second question"):
            pass

        contents = [m.content for m in fake2.calls[0]]
        assert "first question" in contents and "answer one" in contents
        assert [m.role for m in store.messages(sid)] == [
            "user",
            "assistant",
            "user",
            "assistant",
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
        PROFILE_YAML + "memory:\n"
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
        fake = FakeLLMClient(
            [
                ScriptedTurn(text="first answer"),
                ScriptedTurn(text="durable compact summary"),
                ScriptedTurn(text="second answer"),
            ]
        )
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

        await first.aclose()
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
        assert any(
            "padding padding" in message.content for message in store.messages(sid)
        )
