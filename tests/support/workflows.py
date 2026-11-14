from __future__ import annotations

import yaml

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.message import ToolCall
from tests.fakes import FakeLLMClient, ScriptedTurn


async def yes(prompt):
    return True


def build(tmp_path, *, store=None, roles=None):
    (tmp_path / "roles").mkdir(exist_ok=True)
    (tmp_path / "ws").mkdir(exist_ok=True)
    definitions = roles or {
        "pure": {"tools": [], "workspace": "scratch", "context": []}
    }
    for name, definition in definitions.items():
        (tmp_path / "roles" / f"{name}.yaml").write_text(yaml.safe_dump(definition))
    profile = AgentProfile.model_validate(
        {
            "name": "test",
            "llm": {"model": "fake"},
            "workspace": str(tmp_path / "ws"),
            "tools": ["delegate", "read_file", "write_file"],
            "initial_tools": ["delegate", "read_file", "write_file"],
        }
    )
    object.__setattr__(profile, "_source_dir", tmp_path)
    agent = Agent.from_profile(
        profile, llm=FakeLLMClient([]), session_store=store, confirm=yes
    )
    return agent, agent.tool_ctx.delegation


def result(value="same"):
    return FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="s",
                        name="submit_result",
                        arguments={"status": "done", "result": value},
                    )
                ]
            )
        ]
    )


def step(name, prompt="read", **options):
    return {"id": name, "agent": {"role": "pure", "prompt": prompt, **options}}
