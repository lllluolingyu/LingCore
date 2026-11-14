from __future__ import annotations

import asyncio
import time

from lingcore.sessions import SessionStore
from lingcore.workflows.runtime import resume_workflow, run_workflow
from tests.support.workflows import build, result, step
from tests.test_workflow_spec import validate


def test_map_and_loop_prev_bounds_do_not_inherit_outer_result_shape():
    outer = step(
        "old", output={"type": "array", "maxItems": 1, "items": {"type": "integer"}}
    )
    nested = {
        "map": {
            "over": "${prev}",
            "as": "inner",
            "max_items": 5,
            "stages": [step("worker")],
        }
    }
    mapped = {"map": {"over": [[1, 2, 3, 4, 5]], "as": "item", "stages": [nested]}}
    assert validate({"name": "map", "steps": [outer, mapped]}).bound == 6
    loop = {
        "loop": {"max_rounds": 2, "steps": [nested], "collect": {"from": "${prev}"}}
    }
    assert validate({"name": "loop", "steps": [outer, loop]}).bound == 11


async def test_blocked_map_preserves_dependency_block_for_downstream_consumers(
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
        supervisor.llm_factory = lambda role, label: calls.append(label) or result()
        spec = {
            "name": "blocked-map",
            "steps": [
                {
                    "id": "mapped",
                    "map": {
                        "over": [1, 2],
                        "as": "item",
                        "stages": [step("effect", role="writer")],
                    },
                },
                step("consumer", "Use ${mapped}", label="consumer"),
                step("independent", label="independent"),
            ],
        }
        async with agent:
            await run_workflow(supervisor, agent._authorized_tool_names(), spec=spec)
            record = supervisor.journal.list()[0]
            record.spec["steps"][0]["map"]["stages"][0]["agent"]["prompt"] = "changed"
            record.spec["steps"][2]["agent"]["prompt"] = "changed independent"
            supervisor.journal.save(record)
            calls.clear()
            out = await resume_workflow(
                supervisor, record.id, agent._authorized_tool_names()
            )
            assert len(out["blocked"]) == 3
            assert calls == ["independent"]


async def test_frozen_verification_runs_once_off_loop_and_reports_parent_changes(
    tmp_path, monkeypatch
):
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(
            tmp_path,
            store=store,
            roles={
                "pure": {"tools": ["read_file"], "workspace": "frozen", "context": []}
            },
        )
        supervisor.llm_factory = lambda *a: result()
        (tmp_path / "ws/file").write_text("initial")
        async with agent:
            await run_workflow(
                supervisor,
                agent._authorized_tool_names(),
                spec={"name": "read", "steps": [step("a"), step("b")]},
            )
            record = supervisor.journal.list()[0]
            (tmp_path / "ws/file").write_text("changed")
            prompts, checks = [], []
            original = supervisor.workspace.revision

            def revision(*args, **kwargs):
                if not kwargs.get("source"):
                    checks.append(1)
                    time.sleep(0.05)
                return original(*args, **kwargs)

            async def yes(prompt):
                prompts.append(prompt)
                return True

            monkeypatch.setattr(supervisor.workspace, "revision", revision)
            agent.tool_ctx.confirm = yes
            task = asyncio.create_task(
                resume_workflow(supervisor, record.id, agent._authorized_tool_names())
            )
            beats = 0
            while not task.done():
                beats += 1
                await asyncio.sleep(0.005)
            out = await task
            assert checks == [1] and beats >= 3
            assert out["parent_workspace_changed"] is True
            assert '"cache_candidates": 2' in prompts[0]
            assert '"live_worker_upper_bound": 2' in prompts[0]
