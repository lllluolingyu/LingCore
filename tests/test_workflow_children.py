from __future__ import annotations

import asyncio

import pytest
import yaml

from lingcore.workflows.runtime import run_workflow
from tests.support.workflows import build, result, step


@pytest.mark.parametrize("python_parent", [False, True])
async def test_optional_child_inputs_resolve_like_top_level_inputs(
    tmp_path, python_parent
):
    from lingcore.workflows.api import run_python, workflow

    agent, supervisor = build(tmp_path)
    supervisor.llm_factory = lambda *a: result()
    directory = tmp_path / "workflows"
    directory.mkdir()
    (directory / "child.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "child",
                "inputs": {
                    "type": "object",
                    "properties": {"focus": {"type": "string"}},
                },
                "steps": [step("a", "Focus: ${inputs.focus}")],
                "result": "${inputs.focus}",
            }
        )
    )
    async with agent:
        if python_parent:

            @workflow(name="parent")
            async def parent(wf, args):
                return await wf.workflow("child")

            output = await run_python(
                supervisor, agent._authorized_tool_names(), parent, {}
            )
            assert output is None
        else:
            output = await run_workflow(
                supervisor,
                agent._authorized_tool_names(),
                spec={
                    "name": "parent",
                    "steps": [{"workflow": {"name": "child"}}],
                },
            )
            assert output["result"] is None
        assert supervisor.journal.list()[0].status == "completed"


def child_spec(tmp_path, limits=None):
    directory = tmp_path / "workflows"
    directory.mkdir()
    (directory / "child.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "child",
                "limits": limits or {},
                "steps": [{"parallel": [step("a"), step("b")]}],
            }
        )
    )


async def test_child_workers_share_parent_concurrency_cap(tmp_path):
    agent, supervisor = build(tmp_path)
    child_spec(tmp_path)
    active = maximum = 0

    class ObservedLLM:
        usage_sink = None

        async def stream(self, messages, tools=None):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.02)
                async for chunk in result().stream(messages, tools):
                    yield chunk
            finally:
                active -= 1

    supervisor.llm_factory = lambda *a: ObservedLLM()
    async with agent:
        output = await run_workflow(
            supervisor,
            agent._authorized_tool_names(),
            spec={
                "name": "parent",
                "limits": {"max_concurrency": 1},
                "steps": [
                    {
                        "parallel": [
                            {"id": "left", "workflow": {"name": "child"}},
                            {"id": "right", "workflow": {"name": "child"}},
                        ]
                    }
                ],
            },
        )
        assert output["result"] == {
            "left": {"a": "same", "b": "same"},
            "right": {"a": "same", "b": "same"},
        }
        assert maximum == 1
        assert supervisor.journal.list()[0].agents == 4


async def test_child_request_cap_exhausts_without_resetting_parent_accounting(tmp_path):
    agent, supervisor = build(tmp_path)
    child_spec(tmp_path, {"model_requests": 1})
    supervisor.llm_factory = lambda *a: result()
    async with agent:
        output = await run_workflow(
            supervisor,
            agent._authorized_tool_names(),
            spec={
                "name": "parent",
                "steps": [{"workflow": {"name": "child"}}],
            },
        )
        assert output["status"] == "budget_exceeded"
        saved = supervisor.journal.list()[0]
        assert saved.status == "budget_exceeded" and saved.usage.requests == 1
