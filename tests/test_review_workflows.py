from __future__ import annotations

import asyncio

import pytest
import yaml

from lingcore.agent import Agent
from lingcore.errors import ConfigError, ToolError
from lingcore.execution import run_headless
from lingcore.message import ToolCall
from lingcore.sessions import SessionStore
from lingcore.workflows.api import WorkflowAPI
from lingcore.workflows.expressions import BLOCKED, predicate, transform
from lingcore.workflows.runtime import resume_workflow, run_workflow
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.plugin_foundation import write_plugin
from tests.support.workflows import build, result, step
from tests.test_delegation_kernel import profile
from tests.test_workflow_spec import validate


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", ["x"]),
        ("phase", {}),
        ("retry_on", 1),
        ("retry_on", [{}]),
        ("retry_on", "provider_transient"),
    ],
)
def test_malformed_specs_raise_config_errors(field, value):
    node = step("a")
    (node if field == "phase" else node["agent"])[field] = value
    with pytest.raises(ConfigError):
        validate({"name": "invalid", "steps": [node]})


@pytest.mark.parametrize("value", [1, True, "scalar", {}])
@pytest.mark.parametrize(
    "operation", ["count", "take", "votes", "count_at_least", "pick"]
)
def test_expression_collection_type_errors_are_config_errors(value, operation):
    if operation == "pick" and value == {}:
        return
    with pytest.raises(ConfigError):
        if operation == "votes":
            predicate(
                {
                    "votes": {
                        "over": value,
                        "field": "ok",
                        "equals": True,
                        "at_least": 1,
                    }
                },
                {},
            )
        elif operation == "count_at_least":
            predicate({"count_at_least": [value, 1]}, {})
        else:
            transform(
                {
                    operation: value
                    if operation == "count"
                    else {"over": value, "n": 1, "fields": ["x"]}
                },
                {},
            )


def test_missing_optional_object_pick_stays_null():
    assert transform({"pick": {"over": None, "fields": ["x"]}}, {}) is None


@pytest.mark.parametrize("error", [ConfigError, ToolError, RuntimeError])
async def test_python_parallel_propagates_orchestration_errors_and_reaps_siblings(
    tmp_path, error
):
    agent, supervisor = build(tmp_path)
    async with agent:
        api = WorkflowAPI(
            supervisor,
            supervisor.new_run(kind="workflow"),
            agent._authorized_tool_names(),
            {},
        )
        stopped = asyncio.Event()

        async def sibling():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def failed():
            await asyncio.sleep(0)
            raise error("bad orchestration")

        with pytest.raises(error, match="bad orchestration"):
            await api.parallel(sibling(), failed())
        assert stopped.is_set()


async def test_python_child_block_and_counts_are_preserved(tmp_path, monkeypatch):
    agent, supervisor = build(tmp_path)
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows/child.yaml").write_text(
        yaml.safe_dump({"name": "child", "steps": [step("a")]})
    )

    async def blocked(self, steps, values, path, control):
        self.blocked.append({"node": "a", "reason": "blocked"})
        self.failed, self.dropped = 2, 3
        values["prev"] = BLOCKED

    monkeypatch.setattr("lingcore.workflows.interpreter.Interpreter.sequence", blocked)
    async with agent:
        handle = supervisor.new_run(kind="workflow")
        api = WorkflowAPI(supervisor, handle, agent._authorized_tool_names(), {})
        assert await api.workflow("child") is None
        assert handle.record.status == "blocked"
        assert handle.record.progress["children"][0]["failed"] == 2
        assert handle.record.progress["children"][0]["dropped"] == 3


async def test_cache_uses_role_context_and_allows_non_hook_plugins(tmp_path):
    root = write_plugin(tmp_path / "plugins/static", "static", prompt="prompt.md")
    (root / "prompt.md").write_text("Plugin context")
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(
            tmp_path,
            store=store,
            roles={
                "pure": {"tools": [], "context": [], "persona": "Role-specific persona"}
            },
        )
        supervisor.profile.plugins = ["static"]
        calls = []
        supervisor.llm_factory = lambda *a: calls.append(a) or result()
        async with agent:
            await run_workflow(
                supervisor,
                agent._authorized_tool_names(),
                spec={"name": "cached", "steps": [step("a")]},
            )
            record = supervisor.journal.list()[0]
            (tmp_path / "role.md").write_text("Unused parent persona changed")
            (tmp_path / "workflow.md").write_text("Unused parent workflow changed")
            await resume_workflow(supervisor, record.id, agent._authorized_tool_names())
            assert len(calls) == 1
            (root / "prompt.md").write_text("Changed worker plugin context")
            await resume_workflow(supervisor, record.id, agent._authorized_tool_names())
            assert len(calls) == 2


async def test_headless_workflow_confirmation_uses_current_driver_callback(tmp_path):
    spec = {"name": "confirm", "steps": [{"agent": {"prompt": "do the task"}}]}
    parent = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="wf", name="workflow_run", arguments={"spec": spec})
                ]
            ),
            ScriptedTurn(text="finished"),
        ]
    )
    worker = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="shell",
                        name="run_shell",
                        arguments={"command": "printf test"},
                    )
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="result",
                        name="submit_result",
                        arguments={"status": "done", "result": "done"},
                    )
                ]
            ),
        ]
    )
    agent = Agent.from_profile(
        profile(tmp_path, tools=["workflow_run", "run_shell"]), llm=parent
    )
    agent.tool_ctx.delegation.llm_factory = lambda *a: worker
    prompts = []

    async def confirm(prompt):
        prompts.append(prompt)
        return not prompt.startswith("[worker")

    await run_headless(agent, ["start"], confirm=confirm)
    assert any(prompt.startswith("Run confirm") for prompt in prompts)
    assert any(prompt.startswith("[worker") for prompt in prompts)


async def test_cache_tracks_configured_memory_path(tmp_path):
    (tmp_path / "roles").mkdir()
    (tmp_path / "roles/pure.yaml").write_text(
        "tools: []\ncontext: [memory]\npersona: Worker persona\n"
    )
    memory = tmp_path / "custom-memory.md"
    memory.write_text("first memory")
    with SessionStore(tmp_path / "sessions.db") as store:
        p = profile(
            tmp_path,
            tools=["delegate", "memory"],
            tool_options={"memory": {"path": "custom-memory.md"}},
        )
        agent = Agent.from_profile(
            p, llm=FakeLLMClient([]), session_store=store, confirm=lambda _: yes()
        )
        supervisor = agent.tool_ctx.delegation
        calls = []
        supervisor.llm_factory = lambda *a: calls.append(a) or result()
        async with agent:
            await run_workflow(
                supervisor,
                agent._authorized_tool_names(),
                spec={"name": "memory", "steps": [step("a")]},
            )
            record = supervisor.journal.list()[0]
            (tmp_path / "memory.md").write_text("unrelated default path")
            await resume_workflow(supervisor, record.id, agent._authorized_tool_names())
            assert len(calls) == 1
            memory.write_text("changed actual memory")
            await resume_workflow(supervisor, record.id, agent._authorized_tool_names())
            assert len(calls) == 2


async def yes():
    return True


async def test_python_child_preserves_an_actual_worker_question(tmp_path):
    agent, supervisor = build(tmp_path)
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows/child.yaml").write_text(
        yaml.safe_dump(
            {"name": "child", "steps": [step("a"), step("consumer", "Use ${a}")]}
        )
    )
    supervisor.llm_factory = lambda *a: FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="blocked",
                        name="submit_result",
                        arguments={
                            "status": "blocked",
                            "question": "Which input should I use?",
                        },
                    )
                ]
            )
        ]
    )
    async with agent:
        handle = supervisor.new_run(kind="workflow")
        api = WorkflowAPI(supervisor, handle, agent._authorized_tool_names(), {})
        assert await api.workflow("child") is None
        assert handle.record.status == "blocked"
        child = handle.record.progress["children"][0]
        assert child["blocked"][0]["reason"] == "Which input should I use?"
        assert child["dropped"] == 1


async def test_shared_writer_wait_does_not_consume_reader_slot(tmp_path):
    agent, supervisor = build(
        tmp_path,
        roles={
            "writer": {"tools": ["write_file"], "context": []},
            "pure": {"tools": [], "context": []},
        },
    )
    supervisor.slots = asyncio.Semaphore(1)
    supervisor.llm_factory = lambda *a: result()
    async with agent:
        handle = supervisor.new_run(kind="workflow")
        snapshot = agent._authorized_tool_names()
        await supervisor.writer.acquire()
        writer = asyncio.create_task(
            supervisor.worker(
                handle, "write", "writer", supervisor.envelope("writer", snapshot)
            )
        )
        try:
            await asyncio.sleep(0)
            report = await asyncio.wait_for(
                supervisor.worker(
                    handle, "read", "pure", supervisor.envelope("pure", snapshot)
                ),
                2,
            )
            assert report.status == "completed" and not writer.done()
        finally:
            supervisor.writer.release()
            await writer
