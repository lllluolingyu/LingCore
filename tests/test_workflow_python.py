from __future__ import annotations

import pytest
from pydantic import ValidationError

from lingcore.delegation.catalog import prompt_catalog
from lingcore.delegation.validation import validate_definitions
from lingcore.errors import ToolError
from lingcore.sessions import SessionStore
from lingcore.workflows.api import plugin_workflows
from lingcore.workflows.runtime import resume_workflow, run_workflow
from tests.support.plugin_foundation import write_plugin
from tests.support.workflows import build, result

SOURCE = """from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel
from lingcore.workflows import workflow

@dataclass
class Label:
    text: str

class Args(BaseModel):
    prompt: str
    role: str = "pure"

@workflow(name="trusted-test", description="A typed operator workflow")
async def run(wf, args: Args):
    return await wf.agent(Label(args.prompt).text, role=args.role)
"""


def install(tmp_path):
    root = write_plugin(tmp_path / "plugins" / "trusted", "trusted", workflows="flows")
    (root / "flows").mkdir()
    path = root / "flows" / "run.py"
    path.write_text(SOURCE)
    return path


async def test_python_discovery_requires_consent_and_offline_validation_never_imports(
    tmp_path,
):
    path = install(tmp_path)
    path.write_text('raise RuntimeError("operator code executed")\n')
    agent, supervisor = build(tmp_path)
    async with agent:
        assert plugin_workflows(supervisor.profile) == {}
        supervisor.profile.plugins = ["trusted"]
        validate_definitions(supervisor.profile)
        with pytest.raises(RuntimeError, match="operator code executed"):
            plugin_workflows(supervisor.profile)
        path.write_text(SOURCE)
        first = plugin_workflows(supervisor.profile)
        assert (
            first["trusted-test"]
            is plugin_workflows(supervisor.profile)["trusted-test"]
        )
        card = prompt_catalog(supervisor.profile)
        assert "trusted-test" in card and '"prompt"' in card


async def test_python_typed_arguments_and_resume_keep_original_budget(tmp_path):
    install(tmp_path)
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        supervisor.profile.plugins = ["trusted"]
        supervisor.profile.delegation.limits.max_agents = 2
        calls = []
        supervisor.llm_factory = lambda *a: calls.append(a) or result()
        async with agent:
            with pytest.raises(ValidationError):
                await run_workflow(
                    supervisor,
                    agent._authorized_tool_names(),
                    name="trusted-test",
                    args={"prompt": 123},
                )
            assert not supervisor.journal.list()
            assert (
                await run_workflow(
                    supervisor,
                    agent._authorized_tool_names(),
                    name="trusted-test",
                    args={"prompt": "hello"},
                )
                == "same"
            )
            record = supervisor.journal.list()[0]
            assert (
                await resume_workflow(
                    supervisor, record.id, agent._authorized_tool_names()
                )
                == "same"
            )
            exhausted = await resume_workflow(
                supervisor, record.id, agent._authorized_tool_names()
            )
            assert exhausted["status"] == "budget_exceeded"
            assert len(calls) == 2
            assert len(supervisor.journal.list()) == 1
            assert supervisor.journal.get(record.id).agents == 2


async def test_python_resume_refuses_any_started_effecting_worker(tmp_path):
    install(tmp_path)
    with SessionStore(tmp_path / "sessions.db") as store:
        agent, supervisor = build(tmp_path, store=store)
        supervisor.profile.plugins = ["trusted"]
        supervisor.llm_factory = lambda *a: result()
        async with agent:
            await run_workflow(
                supervisor,
                agent._authorized_tool_names(),
                name="trusted-test",
                args={"prompt": "write", "role": "general"},
            )
            record = supervisor.journal.list()[0]
            with pytest.raises(ToolError, match="effecting worker"):
                await resume_workflow(
                    supervisor, record.id, agent._authorized_tool_names()
                )
