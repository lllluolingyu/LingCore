from __future__ import annotations

import asyncio
import json
import time

import pytest

from lingcore.delegation.config import RetentionConfig
from lingcore.delegation.models import RunRecord
from lingcore.delegation.retention import prune
from lingcore.errors import ToolError
from lingcore.sessions import SessionStore
from lingcore.workflows.runtime import resume_workflow, run_workflow
from tests.support.workflows import build, result, step


async def test_background_deadline_publishes_terminal_event_and_releases_source(
    tmp_path,
):
    agent, supervisor = build(tmp_path)
    agent.background.enabled = True
    async with agent:
        handle = supervisor.new_run(background=True)
        handle.budget.deadline = time.monotonic() + 0.02
        await supervisor.execute(handle, asyncio.Event().wait)
        await asyncio.wait_for(handle.task, 1)
        assert supervisor.journal.get(handle.record.id).status == "timed_out"
        assert not agent.background._sources
        assert "timed_out" in agent.background.take().text


async def test_only_guarded_structured_results_reach_parent_and_optionals_are_null(
    tmp_path, monkeypatch
):
    class Guard:
        async def pre_input(self, value):
            return value

        async def post_output(self, value):
            return value.replace("secret", "redacted")

    monkeypatch.setattr("lingcore.agent.build_guardrail", lambda *a: Guard())
    agent, supervisor = build(tmp_path)
    outputs = iter([result({"text": "secret"}), result("done")])
    calls = []

    def factory(*args):
        llm = next(outputs)
        calls.append(llm)
        return llm

    supervisor.llm_factory = factory
    async with agent:
        out = await run_workflow(
            supervisor,
            agent._authorized_tool_names(),
            spec={
                "name": "guarded",
                "steps": [
                    step(
                        "a",
                        output={
                            "type": "object",
                            "required": ["text"],
                            "properties": {
                                "text": {"type": "string"},
                                "optional": {"type": "string"},
                            },
                        },
                    ),
                    step("b", "Optional: ${a.optional}"),
                ],
                "result": "${a}",
            },
        )
        assert out["result"] == {"text": "redacted", "optional": None}
        assert "Optional: null" in calls[1].calls[0][-1].content


async def test_guardrail_that_breaks_result_schema_fails_closed(tmp_path, monkeypatch):
    class Guard:
        async def pre_input(self, text):
            return text

        async def post_output(self, text):
            return "output blocked"

    monkeypatch.setattr("lingcore.agent.build_guardrail", lambda *a: Guard())
    agent, supervisor = build(tmp_path)
    supervisor.llm_factory = lambda *a: result("secret")
    async with agent:
        out = await run_workflow(
            supervisor,
            agent._authorized_tool_names(),
            spec={
                "name": "blocked",
                "steps": [step("a")],
            },
        )
        assert out["result"] is None and out["failed"] == 1


async def test_guarded_result_is_revalidated_even_when_python_values_compare_equal(
    tmp_path, monkeypatch
):
    class Guard:
        async def pre_input(self, text):
            return text

        async def post_output(self, text):
            value = json.loads(text)
            value["result"] = True  # True == 1, but violates an integer schema.
            return json.dumps(value)

    monkeypatch.setattr("lingcore.agent.build_guardrail", lambda *a: Guard())
    agent, supervisor = build(tmp_path)
    supervisor.llm_factory = lambda *a: result(1)
    async with agent:
        out = await run_workflow(
            supervisor,
            agent._authorized_tool_names(),
            spec={
                "name": "guarded-type",
                "steps": [step("a", output={"type": "integer"})],
            },
        )
        assert out["result"] is None and out["failed"] == 1


async def test_declining_resume_preserves_interrupted_run(tmp_path):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        supervisor.llm_factory = lambda *a: result()
        async with agent:
            await run_workflow(
                supervisor,
                agent._authorized_tool_names(),
                spec={"name": "resume", "steps": [step("a")]},
            )
            record = supervisor.journal.list()[0]
            record.status = "interrupted"
            supervisor.journal.save(record)

            async def no(prompt):
                return False

            agent.tool_ctx.confirm = no
            with pytest.raises(ToolError, match="denied"):
                await resume_workflow(
                    supervisor, record.id, agent._authorized_tool_names()
                )
            assert supervisor.journal.get(record.id).status == "interrupted"


async def test_retention_keeps_foreign_and_unlocated_legacy_runs(tmp_path):
    agent, supervisor = build(tmp_path)
    async with agent:
        local = str(supervisor.workspace.workspace)
        for name, workspace in [
            ("r-a", local),
            ("r-foreign", str(tmp_path / "elsewhere")),
            ("r-legacy", None),
            ("r-b", local),
        ]:
            supervisor.journal.save(
                RunRecord(id=name, status="completed", workspace=workspace)
            )
        removed = prune(
            supervisor.journal, supervisor.workspace, RetentionConfig(max_runs=1)
        )
        assert removed["removed"] == ["r-a"]
        assert {r.id for r in supervisor.journal.list()} == {
            "r-b",
            "r-foreign",
            "r-legacy",
        }


async def test_live_handle_history_is_bounded_without_losing_usage(tmp_path):
    agent, supervisor = build(tmp_path)
    async with agent:
        for _ in range(200):
            handle = supervisor.new_run(kind="workflow")
            handle.record.status = "completed"
            handle.record.usage.input_tokens = 10
        supervisor.compact_history()
        assert len(supervisor.runs) <= 128
        assert (
            supervisor.retired_usage.input_tokens
            + sum(h.record.usage.input_tokens for h in supervisor.runs.values())
            == 2000
        )


async def test_worker_lookup_uses_index_instead_of_scanning_runs(tmp_path, monkeypatch):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        supervisor.llm_factory = lambda *a: result()
        async with agent:
            await run_workflow(
                supervisor,
                agent._authorized_tool_names(),
                spec={"name": "lookup", "steps": [step("a")]},
            )
            run = supervisor.journal.list()[0]
            worker = supervisor.journal.nodes(run.id)[0].worker_id
            monkeypatch.setattr(
                supervisor.journal, "list", lambda *a: pytest.fail("full journal scan")
            )
            assert supervisor.journal.worker(worker).worker_id == worker
