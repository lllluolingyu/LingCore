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


def test_source_dir_set_on_file_load(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    p = _write(tmp_path, FIXTURE)
    prof = AgentProfile.load(p)
    assert prof._source_dir == tmp_path.resolve()


def test_load_from_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(FIXTURE, encoding="utf-8")
    prof = AgentProfile.load(tmp_path)
    assert prof.name == "test-coding"
    assert prof._source_dir == tmp_path.resolve()


def test_load_directory_missing_config(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        AgentProfile.load(tmp_path)


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


# --------------------------------------------------------------------------- #
# Workspace resolution                                                         #
# --------------------------------------------------------------------------- #

def test_unset_workspace_defaults_to_profile_subdir(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    text = FIXTURE.replace("workspace: ${TEST_WS:-.}\n", "")
    prof = AgentProfile.load(_write(tmp_path, text))
    assert prof.workspace is None
    # The profile dir wins even when a base (the user's CWD) is supplied.
    assert prof.workspace_path(tmp_path / "elsewhere") == (tmp_path / "workspace").resolve()


def test_blank_workspace_expansion_means_default(tmp_path, monkeypatch):
    # The bundled-profile idiom: ${VAR:-} expands to "" when VAR is unset,
    # which must behave exactly like omitting the workspace key.
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    monkeypatch.delenv("TEST_WS", raising=False)
    text = FIXTURE.replace("${TEST_WS:-.}", "${TEST_WS:-}")
    prof = AgentProfile.load(_write(tmp_path, text))
    assert prof.workspace is None
    assert prof.workspace_path() == (tmp_path / "workspace").resolve()


def test_explicit_relative_workspace_resolves_against_base(tmp_path, monkeypatch):
    # Invariant 8: an explicit relative workspace follows the user's CWD
    # (passed as base), not the profile directory.
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    monkeypatch.delenv("TEST_WS", raising=False)
    prof = AgentProfile.load(_write(tmp_path, FIXTURE))  # workspace: "."
    assert prof.workspace == "."
    assert prof.workspace_path(tmp_path / "cwd") == (tmp_path / "cwd").resolve()


def test_default_workspace_without_source_dir_falls_back_to_base(tmp_path):
    prof = AgentProfile.model_validate({"llm": {"model": "m"}})
    assert prof.workspace_path(tmp_path) == (tmp_path / "workspace").resolve()


async def test_from_profile_creates_default_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    text = FIXTURE.replace("workspace: ${TEST_WS:-.}\n", "")
    prof = AgentProfile.load(_write(tmp_path, text))
    agent = Agent.from_profile(
        prof, llm=FakeLLMClient([ScriptedTurn(text="hi")]), base_dir=tmp_path / "cwd"
    )
    assert agent.tool_ctx.workspace == (tmp_path / "workspace").resolve()
    assert agent.tool_ctx.workspace.is_dir()  # auto-created on assembly
