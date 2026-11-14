from __future__ import annotations

import asyncio

import pytest

from lingcore.sessions import SessionStore
from lingcore.workflows.runtime import resume_workflow, run_workflow
from tests.support.workflows import build, result, step


async def test_pipeline_advances_fast_item_before_slowest_review(tmp_path):
    agent, supervisor = build(tmp_path)
    supervisor.llm_factory = lambda *a: result()
    stage_two = asyncio.Event()
    original = supervisor.worker
    seen = []

    async def worker(handle, prompt, *args, **kwargs):
        if prompt == "first:slow":
            await asyncio.wait_for(stage_two.wait(), 2)
        if prompt == "second:fast":
            stage_two.set()
        seen.append(prompt)
        return await original(handle, prompt, *args, **kwargs)

    supervisor.worker = worker
    spec = {
        "name": "pipeline",
        "steps": [
            {
                "map": {
                    "over": ["slow", "fast"],
                    "as": "item",
                    "stages": [
                        step("one", "first:${item}"),
                        step("two", "second:${item}"),
                    ],
                }
            }
        ],
    }
    async with agent:
        outcome = await run_workflow(
            supervisor, agent._authorized_tool_names(), spec=spec
        )
    assert outcome["result"] == ["same", "same"]
    assert seen.index("second:fast") < seen.index("first:slow")


async def test_resume_uses_dependency_values_and_keeps_previous_usage(tmp_path):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        calls = []
        supervisor.llm_factory = lambda role, label: calls.append(label) or result()
        spec = {"name": "cache", "steps": [step("a"), step("b", "from ${a}")]}
        async with agent:
            await run_workflow(supervisor, agent._authorized_tool_names(), spec=spec)
            record = supervisor.journal.list()[0]
            assert len(calls) == 2
            # Only the first node's local specification changes. Its output is
            # identical, so the dependent consumer is still reusable.
            record.spec["steps"][0]["agent"]["prompt"] = "changed"
            supervisor.journal.save(record)
            outcome = await resume_workflow(
                supervisor, record.id, agent._authorized_tool_names()
            )
            assert outcome["result"] == "same"
            assert len(calls) == 3
            saved = supervisor.journal.get(record.id)
            assert saved.agents == 3 and saved.usage.requests == 3
            assert len(supervisor.journal.nodes(record.id)) == 3


async def test_changed_effecting_step_blocks_consumers_but_independent_work_runs(
    tmp_path,
):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(
            tmp_path,
            store=store,
            roles={
                "pure": {"tools": [], "context": []},
                "writer": {"tools": ["write_file"], "context": []},
            },
        )
        calls = []
        supervisor.llm_factory = lambda role, label: calls.append(role) or result()
        effect = step("effect", role="writer")
        spec = {
            "name": "effects",
            "steps": [effect, step("consumer", "${effect}"), step("independent")],
        }
        async with agent:
            await run_workflow(supervisor, agent._authorized_tool_names(), spec=spec)
            record = supervisor.journal.list()[0]
            record.spec["steps"][0]["agent"]["prompt"] = "changed effect"
            record.spec["steps"][2]["agent"]["prompt"] = "changed independent"
            supervisor.journal.save(record)
            outcome = await resume_workflow(
                supervisor, record.id, agent._authorized_tool_names()
            )
            assert len(outcome["blocked"]) == 2
            assert calls == ["writer", "pure", "pure", "pure"]
            assert supervisor.journal.get(record.id).status == "blocked"


async def test_parallel_values_keyed_and_one_child_cannot_change_sibling_schema(
    tmp_path,
):
    agent, supervisor = build(tmp_path)
    supervisor.llm_factory = lambda *a: result()
    spec = {
        "name": "barrier",
        "steps": [{"id": "both", "parallel": [step("left"), step("right")]}],
        "result": "${both}",
    }
    async with agent:
        outcome = await run_workflow(
            supervisor, agent._authorized_tool_names(), spec=spec
        )
    assert outcome["result"] == {"left": "same", "right": "same"}


async def test_frozen_input_is_reused_and_tampering_refuses_resume(tmp_path):
    from lingcore.errors import ToolError

    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(
            tmp_path,
            store=store,
            roles={
                "pure": {"tools": ["read_file"], "workspace": "frozen", "context": []}
            },
        )
        (tmp_path / "ws" / "file.txt").write_text("first")
        calls = []
        supervisor.llm_factory = lambda *a: calls.append(1) or result()
        async with agent:
            await run_workflow(
                supervisor,
                agent._authorized_tool_names(),
                spec={"name": "frozen", "steps": [step("read")]},
            )
            record = supervisor.journal.list()[0]
            (tmp_path / "ws" / "file.txt").write_text("new parent state")
            await resume_workflow(supervisor, record.id, agent._authorized_tool_names())
            assert calls == [1]
            pinned = supervisor.workspace.path(record.id, "input") / "file.txt"
            pinned.chmod(0o600)
            pinned.write_text("tampered")
            with pytest.raises(ToolError, match="altered"):
                await resume_workflow(
                    supervisor, record.id, agent._authorized_tool_names()
                )


async def test_shipped_review_workflow_executes_all_40_workers_on_fakes(tmp_path):
    from pathlib import Path

    import yaml

    agent, supervisor = build(
        tmp_path,
        roles={
            "reviewer": {"tools": ["read_file"], "workspace": "frozen", "context": []},
            "skeptic": {"tools": ["read_file"], "workspace": "frozen", "context": []},
            "writer": {"tools": [], "context": []},
        },
    )
    findings = [
        {"file": "file.py", "line": n + 1, "claim": f"defect {n}"} for n in range(4)
    ]

    def factory(role, label):
        if role == "reviewer":
            return result({"findings": findings})
        if role == "skeptic":
            return result({"refuted": False, "reason": "confirmed"})
        return result("review report")

    supervisor.llm_factory = factory
    spec = yaml.safe_load(
        Path("profiles/coding/workflows/review-changes.yaml").read_text()
    )
    async with agent:
        outcome = await run_workflow(
            supervisor,
            agent._authorized_tool_names(),
            spec=spec,
            args={"focus": "correctness"},
        )
        record = supervisor.journal.list()[0]
        assert record.agents == 40 and record.usage.requests == 40
    assert outcome["result"]["report"] == "review report"
    assert len(outcome["result"]["findings"]) == 12
    assert not outcome["failed"] and not outcome["blocked"]
