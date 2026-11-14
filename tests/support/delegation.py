"""Small builders for delegation runtime seam tests."""

from __future__ import annotations

from pydantic import BaseModel

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from tests.fakes import FakeLLMClient


class EmptyArgs(BaseModel):
    pass


def build_agent(
    tmp_path, turns, *, extra_tools=(), terminal_tools=frozenset(), **kwargs
):
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "fake"},
            "workspace": str(tmp_path),
            "tools": [],
        }
    )
    llm = FakeLLMClient(turns)
    agent = Agent.from_profile(
        profile,
        llm=llm,
        extra_tools=extra_tools,
        terminal_tools=terminal_tools,
        **kwargs,
    )
    return agent, llm
