"""Fixtures and builders for agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

import lingcore.tools.builtin  # noqa: F401  (registers builtins)
from lingcore.agent import Agent
from lingcore.composer import StaticComposer
from lingcore.memory import WindowMemory
from lingcore.tools import ToolContext, ToolRegistry


class _EmptyArgs(BaseModel):
    pass


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    return tmp_path


def _agent(llm, workspace, tools=("read_file",), **kw) -> Agent:
    reg = ToolRegistry()
    from lingcore.tools import REGISTRY

    for name in tools:
        reg.register(REGISTRY.get(name))
    return Agent(
        llm=llm,
        tools=reg,
        tool_ctx=ToolContext(workspace=workspace),
        composer=StaticComposer("You are a coding agent."),
        memory=WindowMemory(model="gpt-4o"),
        **kw,
    )


async def _drain(agent, text):
    return [ev async for ev in agent.run(text)]


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr("lingcore.agent._backoff_seconds", lambda attempt: 0.0)
