"""Rewriting through the real agent loop, persistence, Stop, and budgets."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.events import (
    Error,
    Final,
    TextDelta,
    TurnCancelled,
    TurnStatsReported,
    UsageReported,
)
from lingcore.execution import ExecutionBudget, ExecutionLimits
from lingcore.llm import LLMChunk
from lingcore.message import ToolCall
from lingcore.sessions import SessionStore
from lingcore.usage import TokenUsage
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure


def _profile(tmp_path, **rewrite_options):
    return AgentProfile.model_validate(
        {
            "llm": {"model": "main"},
            "workspace": str(tmp_path / "workspace"),
            "tools": ["list_dir"],
            "output_rewrite": {
                "enabled": True,
                "llm": {"model": "editor", "stream_retries": 0},
                **rewrite_options,
            },
        }
    )


async def test_only_final_reply_is_rewritten_guarded_and_saved(tmp_path):
    class Guardrail:
        async def pre_input(self, text):
            return text

        async def post_output(self, text):
            return text + " [checked]"

    main = FakeLLMClient(
        [
            ScriptedTurn(
                text="Checking files.",
                tool_calls=[ToolCall(id="ls", name="list_dir", arguments={})],
            ),
            ScriptedTurn(text="Stilted draft", reasoning="private reasoning"),
        ]
    )
    editor = FakeLLMClient([ScriptedTurn(text="Natural reply")])
    profile = _profile(tmp_path)
    with SessionStore(tmp_path / "sessions.db") as store:
        async with Agent.from_profile(
            profile, llm=main, rewrite_llm=editor, session_store=store
        ) as agent:
            agent.guardrail = Guardrail()
            events = [event async for event in agent.run("list my files")]
            assert events[-1] == Final("Natural reply [checked]")
            shown = "".join(
                event.text for event in events if isinstance(event, TextDelta)
            )
            assert shown == "Checking files.Natural reply [checked]"
            assert len(editor.calls) == 1
            assert [(m.role, m.content) for m in editor.calls[0]][1:] == [
                ("user", "Stilted draft")
            ]
            assert len(editor.calls[0]) == 2
            assert editor.tool_schemas == [None]
            assert all(m.reasoning_content is None for m in editor.calls[0])
            session_id = agent.tool_ctx.session_id
            saved = store.messages(session_id)[-1]
            assert saved.content == "Natural reply [checked]"
            assert saved.reasoning_content == "private reasoning"
        next_main = FakeLLMClient([ScriptedTurn(text="Followup draft")])
        async with Agent.from_profile(
            profile,
            llm=next_main,
            rewrite_llm=FakeLLMClient([ScriptedTurn(text="Followup reply")]),
            session_store=store,
            session_id=session_id,
        ) as resumed:
            [event async for event in resumed.run("continue")]
            assert next_main.calls[0][-2].content == "Natural reply [checked]"


@pytest.mark.parametrize(
    "turn",
    [
        ScriptedTurn(text=""),
        ScriptedTurn(text="   "),
        ScriptedTurn(text="partial", finish_reason="length"),
        ScriptedTurn(text="partial", finish_reason="content_filter"),
        ScriptedTurn(text="partial", finish_reason=""),
        ScriptedTurn(text="too long" * 10),
        ScriptedTurn(
            text="partial", tool_calls=[ToolCall(id="x", name="list_dir", arguments={})]
        ),
        StreamFailure(text="partial", reason="secret-sentinel", retryable=False),
    ],
)
async def test_failed_rewrite_falls_back_without_exposing_partial_text(
    tmp_path, caplog, turn
):
    async with Agent.from_profile(
        _profile(tmp_path, max_output_chars=20),
        llm=FakeLLMClient([ScriptedTurn(text="Original reply")]),
        rewrite_llm=FakeLLMClient([turn]),
    ) as agent:
        events = [event async for event in agent.run("question")]
        assert events[-1] == Final("Original reply")
        assert [event.text for event in events if isinstance(event, TextDelta)] == [
            "Original reply"
        ]
        assert agent.memory.messages[-1].content == "Original reply"
    assert "secret-sentinel" not in caplog.text


@pytest.mark.parametrize("draft", ["", "large" * 10])
async def test_empty_or_oversized_draft_skips_rewrite(tmp_path, draft):
    editor = FakeLLMClient([])
    async with Agent.from_profile(
        _profile(tmp_path, max_input_chars=20),
        llm=FakeLLMClient([ScriptedTurn(text=draft)]),
        rewrite_llm=editor,
    ) as agent:
        events = [event async for event in agent.run("question")]
        assert events[-1] == Final(draft)
    assert editor.calls == []


async def test_incomplete_main_reply_skips_rewrite(tmp_path):
    editor = FakeLLMClient([])
    async with Agent.from_profile(
        _profile(tmp_path),
        llm=FakeLLMClient([ScriptedTurn(text="unfinished", finish_reason="length")]),
        rewrite_llm=editor,
    ) as agent:
        events = [event async for event in agent.run("question")]
        assert isinstance(events[-1], Error)
        assert events[-1].partial_content == "unfinished"
        assert editor.calls == []


async def test_retry_discards_partial_rewrite(tmp_path):
    editor = FakeLLMClient(
        [StreamFailure(text="discarded rewrite"), ScriptedTurn(text="Natural reply")]
    )
    profile = _profile(tmp_path)
    profile.output_rewrite.llm.stream_retries = 1
    async with Agent.from_profile(
        profile,
        llm=FakeLLMClient([ScriptedTurn(text="Draft")]),
        rewrite_llm=editor,
    ) as agent:
        events = [event async for event in agent.run("question")]
        assert events[-1] == Final("Natural reply")
        assert [event.text for event in events if isinstance(event, TextDelta)] == [
            "Natural reply"
        ]
        assert len(editor.calls) == 2


class WaitingEditor:
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def stream(self, messages, tools=None):
        try:
            self.started.set()
            yield LLMChunk(text_delta="unfinished rewrite")
            await asyncio.Event().wait()
        finally:
            self.closed = True


async def test_rewrite_wall_timeout_closes_stream_and_uses_original(tmp_path):
    editor = WaitingEditor()
    async with Agent.from_profile(
        _profile(tmp_path, timeout_seconds=0.01),
        llm=FakeLLMClient([ScriptedTurn(text="Original")]),
        rewrite_llm=editor,
    ) as agent:
        events = [event async for event in agent.run("question")]
        assert events[-1] == Final("Original")
        assert editor.closed


async def test_stop_during_rewrite_rolls_back_and_releases_stream(tmp_path):
    editor = WaitingEditor()
    with SessionStore(tmp_path / "sessions.db") as store:
        async with Agent.from_profile(
            _profile(tmp_path),
            llm=FakeLLMClient([ScriptedTurn(text="Draft")]),
            rewrite_llm=editor,
            session_store=store,
        ) as agent:
            shown = []

            async def drive():
                async for event in agent.run("question"):
                    shown.append(event)

            task = asyncio.create_task(drive())
            await asyncio.wait_for(editor.started.wait(), 2)
            assert agent.cancel_turn()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert isinstance(agent.finalize_cancelled_turn(), TurnCancelled)
            assert editor.closed
            assert not any(isinstance(event, TextDelta | Final) for event in shown)
            assert [(m.role, m.content) for m in agent.memory.messages] == [
                ("user", "question")
            ]
            assert len(store.messages(agent.tool_ctx.session_id)) == 1
            agent.llm = FakeLLMClient([ScriptedTurn(text="Next draft")])
            agent.output_rewriter.client = FakeLLMClient(
                [ScriptedTurn(text="Next reply")]
            )
            assert [event async for event in agent.run("again")][-1] == Final(
                "Next reply"
            )


async def test_rewrite_usage_is_accounted_and_request_budget_is_enforced(tmp_path):
    profile = _profile(tmp_path)
    main_usage = TokenUsage("main", input_tokens=10, output_tokens=3)
    rewrite_usage = TokenUsage(
        "editor", input_tokens=5, output_tokens=4, purpose="output_rewrite"
    )
    budget = ExecutionBudget(ExecutionLimits(model_requests=2))
    async with Agent.from_profile(
        profile,
        llm=FakeLLMClient([ScriptedTurn(text="Draft", usage=main_usage)]),
        rewrite_llm=FakeLLMClient([ScriptedTurn(text="Reply", usage=rewrite_usage)]),
        execution_budget=budget,
    ) as agent:
        events = [event async for event in agent.run("question")]
        assert events[-1] == Final("Reply")
        stats = next(
            event.stats for event in events if isinstance(event, TurnStatsReported)
        )
        assert stats.usage == (main_usage, rewrite_usage)
        reported = [event for event in events if isinstance(event, UsageReported)]
        assert [event.purpose for event in reported] == ["main", "output_rewrite"]
        from lingcore.io.cli import CLIFrontend

        cli = CLIFrontend()
        for event in reported:
            cli.render(event)
        assert cli._context_tokens == 13
        assert cli._session_usage.requests == 2
        assert cli._session_usage.input_tokens == 15
        assert cli._session_usage.output_tokens == 7
        assert budget.requests == 2 and budget.tokens == 22
        assert [record.purpose for record in agent.request_ledger.records] == [
            "main",
            "output_rewrite",
        ]
    editor = FakeLLMClient([])
    async with Agent.from_profile(
        profile,
        llm=FakeLLMClient([ScriptedTurn(text="Draft")]),
        rewrite_llm=editor,
        execution_budget=ExecutionBudget(ExecutionLimits(model_requests=1)),
    ) as agent:
        events = [event async for event in agent.run("question")]
        assert isinstance(events[-1], Error)
        assert agent.execution_outcome.status == "budget_exceeded"
        assert editor.calls == []
        assert [m.role for m in agent.memory.messages] == ["user"]


@pytest.mark.parametrize("backend", ["anthropic", "openai-responses"])
async def test_rewritten_native_replay_keeps_reasoning_and_updated_text(
    tmp_path, backend
):
    if backend == "anthropic":
        snapshot = [
            {"type": "thinking", "thinking": "private", "signature": "signed"},
            {"type": "text", "text": "Draft", "citations": []},
        ]
        field = "anthropic_content"
    else:
        snapshot = [
            {
                "type": "reasoning",
                "id": "r",
                "encrypted_content": "opaque",
                "summary": [],
            },
            {
                "type": "message",
                "id": "m",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Draft", "annotations": []}
                ],
            },
        ]
        field = "responses_content"
    original = deepcopy(snapshot)

    class NativeClient(FakeLLMClient):
        async def stream(self, messages, tools=None):
            async for chunk in super().stream(messages, tools):
                if chunk.finish_reason:
                    setattr(chunk, field, snapshot)
                yield chunk

    async with Agent.from_profile(
        _profile(tmp_path),
        llm=NativeClient([ScriptedTurn(text="Draft")]),
        rewrite_llm=FakeLLMClient([ScriptedTurn(text="Natural reply")]),
    ) as agent:
        events = [event async for event in agent.run("question")]
        assert events[-1] == Final("Natural reply")
        saved = agent.memory.messages[-1]
        native = getattr(saved, field)
        assert native[0] == original[0]
        assert "Natural reply" in str(native)
        assert "Draft" not in str(native)
        wire = saved.to_anthropic() if backend == "anthropic" else saved.to_responses()
        assert "Natural reply" in str(wire) and "Draft" not in str(wire)
    assert snapshot == original
