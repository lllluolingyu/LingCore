"""Tests for config / profile loading and Agent.from_profile (M4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from tests.fakes import FakeLLMClient, ScriptedTurn

# A minimal profile that only uses tools built before M6.
FIXTURE = """
name: test-coding
workspace: ${TEST_WS:-.}
llm:
  model: ${TEST_MODEL:-test-model}
  base_url: http://localhost:11434/v1
  api_key_env: TEST_KEY
  sampling:
    temperature: 0.1
    top_p: 0.9
persona:
  system_prompt: "You are a test agent."
tools:
  - read_file
  - list_dir
memory:
  max_messages: 10
  max_tokens: 2000
loop:
  max_iters: 5
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "profile.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    prof = AgentProfile.load(_write(tmp_path, FIXTURE))
    assert prof.name == "test-coding"
    assert prof.llm.model == "test-model"  # default branch of ${TEST_MODEL:-...}
    assert prof.llm.sampling.as_kwargs() == {"temperature": 0.1, "top_p": 0.9}
    assert prof.tools == ["read_file", "list_dir"]
    assert prof.loop.max_iters == 5


def test_env_substitution_uses_set_value(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODEL", "from-env")
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    prof = AgentProfile.load(_write(tmp_path, FIXTURE))
    assert prof.llm.model == "from-env"


def test_missing_env_without_default_errors(tmp_path):
    text = """
llm:
  model: ${REQUIRED_BUT_UNSET}
  api_key_env: TEST_KEY
"""
    with pytest.raises(ConfigError, match="not set"):
        AgentProfile.load(_write(tmp_path, text))


def test_api_key_resolution_missing(tmp_path):
    prof = AgentProfile.load(_write(tmp_path, FIXTURE.replace("api_key_env: TEST_KEY", "api_key_env: NOPE_KEY")))
    with pytest.raises(ConfigError, match="not set in the environment"):
        prof.llm.resolve_api_key()


def test_no_api_key_env_returns_placeholder(tmp_path):
    text = """
llm:
  model: m
  base_url: http://localhost:11434/v1
"""
    prof = AgentProfile.load(_write(tmp_path, text))
    assert prof.llm.resolve_api_key() == "lingcore-no-key"


def test_typo_key_rejected(tmp_path):
    text = """
llm:
  model: m
  api_key_env: TEST_KEY
temperatuer: 0.5
"""
    with pytest.raises(ConfigError, match="invalid profile"):
        AgentProfile.load(_write(tmp_path, text))


def test_unknown_tool_rejected_at_assembly(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    text = FIXTURE.replace("  - list_dir", "  - list_dir\n  - ghost_tool")
    prof = AgentProfile.load(_write(tmp_path, text))
    llm = FakeLLMClient([ScriptedTurn(text="hi")])
    with pytest.raises(ConfigError, match="unknown tool"):
        Agent.from_profile(prof, llm=llm, base_dir=tmp_path)


def test_missing_profile_file():
    with pytest.raises(ConfigError, match="not found"):
        AgentProfile.load("/nonexistent/profile.yaml")


async def test_from_profile_builds_runnable_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    (tmp_path / "hello.txt").write_text("world", encoding="utf-8")
    monkeypatch.setenv("TEST_WS", str(tmp_path))
    prof = AgentProfile.load(_write(tmp_path, FIXTURE))

    llm = FakeLLMClient([ScriptedTurn(text="Done.")])
    agent = Agent.from_profile(prof, llm=llm, base_dir=tmp_path)

    assert agent.system_prompt == "You are a test agent."
    assert agent.max_iters == 5
    assert set(agent.tools.names()) == {"read_file", "list_dir"}
    assert agent.tool_ctx.workspace == tmp_path.resolve()

    events = [ev async for ev in agent.run("hi")]
    assert events[-1].__class__.__name__ == "Final"
