"""Fixtures and builders for sessions tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.config import AgentProfile
from lingcore.message import Message, ToolCall, ToolResult
from lingcore.sessions import (
    SessionStore,
)

PROFILE_YAML = """
name: testprof
workspace: .
llm:
  model: gpt-4o
tools: []
"""


def _profile(tmp_path: Path, yaml_text: str = PROFILE_YAML) -> AgentProfile:
    d = tmp_path / "prof"
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(yaml_text, encoding="utf-8")
    return AgentProfile.load(d)


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    with SessionStore(tmp_path / "sessions.db") as s:
        yield s


def _tool_block() -> list[Message]:
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})
    return [
        Message.assistant(content="", tool_calls=[call]),
        Message.from_tool_result(
            ToolResult(call_id="c1", name="read_file", content="hello")
        ),
    ]
