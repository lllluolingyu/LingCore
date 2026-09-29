"""todo_write: validation, loop events, persistence, Stop/rewind, compaction."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, ToolError
from lingcore.events import Final, TodoUpdated, ToolResultEvent
from lingcore.memory import SummarizingMemory, WindowMemory
from lingcore.message import Message, ToolCall
from lingcore.sessions import SessionStore, new_session_id
from lingcore.todos import (
    TodoItem,
    TodoState,
    render_todos,
    todos_from_payload,
    validate_todos,
)
from lingcore.tools import ToolContext
from lingcore.tools.builtin.todo import TODO_STATE_KEY, TodoWriteArgs, todo_write
from tests.fakes import FakeLLMClient, ScriptedTurn

PROFILE = """
name: todo
workspace: {ws}
llm: {{model: m}}
tools: [todo_write, run_shell]
tool_options:
  run_shell:
    require_confirmation: true
"""


def _profile(tmp_path: Path) -> AgentProfile:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    cfg = tmp_path / "prof" / "config.yaml"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(PROFILE.format(ws=ws), encoding="utf-8")
    return AgentProfile.load(cfg)


def _write(call_id: str, *todos: tuple[str, str]) -> ScriptedTurn:
    return ScriptedTurn(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="todo_write",
                arguments={"todos": [{"content": c, "status": s} for c, s in todos]},
            )
        ],
        finish_reason="tool_calls",
    )


async def _drain(agent: Agent, text: str) -> list:
    return [event async for event in agent.run(text)]


# --- validation and the tool itself ---------------------------------------


def test_validation_rules():
    item = TodoItem(content="  run   the\ntests ", status="in_progress")
    assert item.content == "run the tests"
    with pytest.raises(ValueError):
        TodoItem(content="   ")
    with pytest.raises(ValueError, match="at most one"):
        validate_todos([item, item])
    with pytest.raises(ValueError, match="at most 2"):
        validate_todos([TodoItem(content=str(i)) for i in range(3)], max_items=2)
    assert render_todos([item, TodoItem(content="b", status="completed")]) == (
        "[>] run the tests\n[x] b"
    )
    assert todos_from_payload({"todos": [{"content": "x", "status": "bogus"}]}) is None
    assert todos_from_payload({"todos": "nope"}) is None


async def test_tool_replaces_list_and_honors_max_items(tmp_path: Path):
    state = TodoState()
    ctx = ToolContext(workspace=tmp_path, options={TODO_STATE_KEY: state})
    out = await todo_write(
        TodoWriteArgs(
            todos=[
                TodoItem(content="a", status="completed"),
                TodoItem(content="b", status="in_progress"),
            ]
        ),
        ctx,
    )
    assert "(1/2 completed)" in out and "[>] b" in out
    assert [i.content for i in state.items] == ["a", "b"]

    await todo_write(TodoWriteArgs(todos=[]), ctx)
    assert state.items == ()

    ctx.options["todo_write"] = {"max_items": 1}
    with pytest.raises(ToolError, match="at most 1"):
        await todo_write(
            TodoWriteArgs(todos=[TodoItem(content="a"), TodoItem(content="b")]), ctx
        )


async def test_tool_without_state_errors(tmp_path: Path):
    with pytest.raises(ToolError, match="not available"):
        await todo_write(TodoWriteArgs(todos=[]), ToolContext(workspace=tmp_path))


def test_profile_rejects_unknown_todo_options(tmp_path: Path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "name: p\nllm: {model: m}\ntools: [todo_write]\n"
        "tool_options: {todo_write: {max_itemz: 3}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="max_itemz"):
        AgentProfile.load(cfg)


# --- loop integration -------------------------------------------------------


async def test_loop_emits_todo_updated_after_the_tool_batch(tmp_path: Path):
    llm = FakeLLMClient(
        [
            _write("c1", ("inspect", "in_progress"), ("fix", "pending")),
            _write("c2", ("inspect", "in_progress"), ("fix", "pending")),  # no-op
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent.from_profile(_profile(tmp_path), llm=llm)
    events = await _drain(agent, "go")

    updates = [e for e in events if isinstance(e, TodoUpdated)]
    assert len(updates) == 1  # an unchanged rewrite emits nothing
    assert [i.content for i in updates[0].todos] == ["inspect", "fix"]
    first_result = next(
        i for i, e in enumerate(events) if isinstance(e, ToolResultEvent)
    )
    assert events.index(updates[0]) > first_result
    assert isinstance(events[-1], Final)
    # The model sees the list in the tool result, never in the system prompt.
    assert "inspect" not in llm.calls[-1][0].content


async def test_todos_persist_and_restore_on_resume(tmp_path: Path):
    profile = _profile(tmp_path)
    with SessionStore(tmp_path / "s.db") as store:
        llm = FakeLLMClient([_write("c1", ("plan", "completed")), ScriptedTurn("ok")])
        agent = Agent.from_profile(profile, llm=llm, session_store=store)
        await _drain(agent, "go")
        sid = agent.memory.session_id  # type: ignore[attr-defined]
        assert [i.content for i in store.latest_todos(sid)] == ["plan"]

        resumed = Agent.from_profile(
            profile, llm=FakeLLMClient([]), session_store=store, session_id=sid
        )
        assert resumed.todo_state is not None
        assert [i.status for i in resumed.todo_state.items] == ["completed"]


async def test_stop_restores_the_pre_turn_list(tmp_path: Path):
    profile = _profile(tmp_path)
    with SessionStore(tmp_path / "s.db") as store:
        llm = FakeLLMClient([_write("c1", ("old", "pending")), ScriptedTurn("ok")])
        agent = Agent.from_profile(profile, llm=llm, session_store=store)
        await _drain(agent, "first")
        sid = agent.memory.session_id  # type: ignore[attr-defined]

        async def hang(command: str) -> bool:
            await asyncio.sleep(10)
            return True

        agent.tool_ctx.confirm = hang
        agent.llm = FakeLLMClient(
            [
                _write("c2", ("new", "in_progress")),
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(id="c3", name="run_shell", arguments={"command": "ls"})
                    ],
                    finish_reason="tool_calls",
                ),
            ]
        )

        async def drive() -> None:
            async for event in agent.run("second"):
                if isinstance(event, TodoUpdated):
                    assert agent.cancel_turn()

        task = asyncio.create_task(drive())
        with pytest.raises(asyncio.CancelledError):
            await task
        agent.finalize_cancelled_turn()

        assert agent.todo_state is not None
        assert [i.content for i in agent.todo_state.items] == ["old"]
        assert [i.content for i in store.latest_todos(sid)] == ["old"]


def test_rewind_and_fork_follow_todo_rows(tmp_path: Path):
    with SessionStore(tmp_path / "s.db") as store:
        sid = new_session_id()
        store.append(sid, Message.user("first"))
        store.save_todo_state(sid, [TodoItem(content="a")])
        store.append(sid, Message.assistant(content="answer"))
        store.append(sid, Message.user("second"))
        store.save_todo_state(sid, [TodoItem(content="b")])
        assert [i.content for i in store.latest_todos(sid)] == ["b"]

        fork = store.fork_session(sid, through_seq=1)
        assert [i.content for i in store.latest_todos(fork.id)] == ["a"]
        assert [e.kind for e in store.events(fork.id)] == ["todo_state"]

        store.rewind_to_user_message(sid, 2)
        assert [i.content for i in store.latest_todos(sid)] == ["a"]

        # A corrupt latest row yields an empty list (it grants nothing) and is
        # dropped, not copied, by a fork.
        store.append(sid, Message.user("third"))
        store._conn.execute(
            "INSERT INTO session_events (session_id, message_seq, kind, created_at, "
            "payload) VALUES (?, 2, 'todo_state', '2026-01-01T00:00:00+00:00', ?)",
            (sid, json.dumps({"todos": [{"content": "", "status": "x"}]})),
        )
        store._conn.commit()
        assert store.latest_todos(sid) == ()
        forked = store.fork_session(sid)
        assert [e.payload for e in store.events(forked.id)] == [
            {"todos": [{"content": "a", "status": "pending"}]}
        ]


# --- compaction keeps the list verbatim -------------------------------------


class _Summarizer:
    async def stream(self, messages, tools=None):
        from lingcore.llm import LLMChunk

        yield LLMChunk(text_delta="older work happened")
        yield LLMChunk(finish_reason="stop")


async def test_compaction_pins_the_todo_list():
    window = WindowMemory(max_messages=100, max_tokens=200)
    memory = SummarizingMemory(
        window, _Summarizer(), compact_at_ratio=0.5, keep_recent_ratio=0.2
    )
    for i in range(12):
        memory.add(Message.user(f"message number {i} " + "x" * 40))
        memory.add(Message.assistant(content=f"reply {i}"))
    state = TodoState(items=(TodoItem(content="ship it", status="in_progress"),))
    memory.set_pinned_note(state.pinned_note())

    assert await memory.maybe_compact() is not None
    summary = memory.messages[0]
    assert summary.name == "summary"
    assert "older work happened" in summary.content
    assert summary.content.endswith("[Current todo list]\n[>] ship it")

    # A cleared note (empty list) is omitted from the next summary.
    memory.set_pinned_note("")
    for i in range(12):
        memory.add(Message.user("more " + "y" * 60))
    await memory.maybe_compact()
    assert "[Current todo list]" not in memory.messages[0].content


def test_eviction_repins_the_note_and_keeps_the_prefix_stable():
    window = WindowMemory(max_messages=1000, max_tokens=400, evict_to_ratio=0.5)
    previous = window.render("sys")
    evictions = 0
    for i in range(60):
        note = f"[Current todo list]\n[>] step {i // 10}"
        window.set_pinned_note(note)
        window.add(Message.user(f"message {i} " + "x" * 80))
        rendered = window.render("sys")
        if rendered[: len(previous)] == previous:
            # Between evictions the list is append-only (cache-stable prefix),
            # even though the note changed.
            assert len(rendered) == len(previous) + 1
        else:
            evictions += 1
            pinned = [m for m in rendered if m.name == "pinned"]
            assert len(pinned) == 1 and rendered[1] is pinned[0]
            assert pinned[0].content.endswith(note)  # the note current then
        previous = rendered
    assert evictions >= 2


async def test_evicted_todo_result_stays_visible_to_the_model(tmp_path: Path):
    profile = _profile(tmp_path)
    profile.memory.max_messages = 4
    llm = FakeLLMClient(
        [
            _write("c1", ("inspect", "completed"), ("fix", "in_progress")),
            ScriptedTurn(text="working"),
            ScriptedTurn(text="still working"),
            ScriptedTurn(text="and more"),
        ]
    )
    agent = Agent.from_profile(profile, llm=llm)
    for text in ("go", "next", "again"):
        await _drain(agent, text)

    last_request = llm.calls[-1]
    # The todo_write call/result pair was evicted from the window...
    assert all(m.role != "tool" for m in last_request)
    # ...but the live list still reaches the model.
    rendered = "\n".join(m.content or "" for m in last_request)
    assert "[x] inspect\n[>] fix" in rendered


# --- frontends --------------------------------------------------------------


def test_cli_renders_checklist_and_hides_tool_noise():
    from rich.console import Console

    from lingcore.io.cli import CLIFrontend
    from lingcore.message import ToolResult

    cli = CLIFrontend(agent_name="t")
    cli.console = Console(record=True, width=120)
    call = ToolCall(id="c", name="todo_write", arguments={"todos": []})
    from lingcore.events import ToolCallStarted

    cli.render(ToolCallStarted(call))
    cli.render(ToolResultEvent(ToolResult(call_id="c", name="todo_write", content="x")))
    cli.render(
        TodoUpdated(
            (
                TodoItem(content="read [code]", status="completed"),
                TodoItem(content="fix", status="in_progress"),
                TodoItem(content="test"),
            )
        )
    )
    out = cli.console.export_text()
    assert "todo_write" not in out
    assert "todos 1/3" in out
    assert "✓ read [code]" in out and "▸ fix" in out and "○ test" in out


async def test_telegram_status_summarizes_todos():
    from lingcore.integrations.telegram.rendering import TelegramTurnRenderer
    from tests.test_telegram_rendering import FakeSender

    sender = FakeSender()
    renderer = TelegramTurnRenderer(sender, 1, edit_interval=0)
    await renderer.handle(
        TodoUpdated(
            (
                TodoItem(content="a", status="completed"),
                TodoItem(content="b", status="in_progress"),
            )
        )
    )
    assert any("todos: 1/2 done — now: b" in text for text in sender.messages.values())
