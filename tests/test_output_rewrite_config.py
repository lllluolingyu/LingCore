"""Experimental rewrite configuration, scoped credentials, and diagnostics."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from lingcore.agent import Agent
from lingcore.config import AgentProfile, OutputRewriteCfg
from lingcore.doctor import diagnose_profile, required_environment_names
from lingcore.errors import ConfigError
from lingcore.profiles import initialize_profile
from lingcore.usage import TokenUsage
from tests.fakes import FakeLLMClient, ScriptedTurn


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": True},
        {"timeout_seconds": 0},
        {"timeout_seconds": 301},
        {"max_input_chars": 0},
        {"max_output_chars": 0},
        {"prompt": ""},
        {"enabeld": True},
        {"llm": {"model": "rewrite", "api_key": "literal-key"}},
    ],
)
def test_invalid_rewrite_configuration_is_rejected(options):
    with pytest.raises(ValidationError):
        OutputRewriteCfg.model_validate(options)


async def test_disabled_rewrite_resolves_no_credentials_or_client(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ABSENT_REWRITE_KEY", raising=False)
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "main"},
            "workspace": str(tmp_path),
            "output_rewrite": {
                "enabled": False,
                "llm": {"model": "rewrite", "api_key_env": "ABSENT_REWRITE_KEY"},
            },
        }
    )
    rewrite = FakeLLMClient([ScriptedTurn(text="should not run")])
    async with Agent.from_profile(
        profile,
        llm=FakeLLMClient([ScriptedTurn(text="original")]),
        rewrite_llm=rewrite,
    ) as agent:
        assert agent.output_rewriter is None
        assert (await _events(agent))[-1].content == "original"
    assert rewrite.calls == []
    assert required_environment_names(profile) == frozenset()


async def _events(agent):
    return [event async for event in agent.run("hello")]


@pytest.mark.parametrize("backend", ["openai", "openai-responses", "anthropic"])
async def test_rewrite_builds_an_independent_client(tmp_path, monkeypatch, backend):
    captured = []

    class Client(FakeLLMClient):
        def __init__(self, **kwargs):
            super().__init__([])
            self.usage_sink = kwargs["usage_sink"]
            captured.append(kwargs)

    monkeypatch.setattr("lingcore.llm.LLMClient", Client)
    monkeypatch.setattr("lingcore.llm_responses.ResponsesLLMClient", Client)
    monkeypatch.setattr("lingcore.llm_anthropic.AnthropicLLMClient", Client)
    (tmp_path / "config.yaml").write_text(
        """
llm:
  model: main
  base_url: https://main.example/v1
  api_key_env: MAIN_KEY
output_rewrite:
  enabled: true
  llm:
    model: ${REWRITE_MODEL}
    backend: ${REWRITE_BACKEND}
    base_url: ${REWRITE_URL}
    api_key_env: EDITOR_KEY
"""
    )
    (tmp_path / ".env").write_text(
        f"MAIN_KEY=main-profile-secret\nEDITOR_KEY=editor-profile-secret\n"
        f"REWRITE_MODEL=editor\nREWRITE_BACKEND={backend}\n"
        "REWRITE_URL=https://editor.example/v1\n"
    )
    monkeypatch.setenv("EDITOR_KEY", "ambient-editor-secret")
    monkeypatch.delenv("REWRITE_MODEL", raising=False)
    profile = AgentProfile.load(tmp_path)
    assert profile.output_rewrite.llm.resolve_api_key() == "editor-profile-secret"
    assert "REWRITE_MODEL" not in os.environ
    assert os.environ["EDITOR_KEY"] == "ambient-editor-secret"
    async with Agent.from_profile(profile) as agent:
        assert agent.output_rewriter.client is not agent.llm
        captured[0]["usage_sink"](TokenUsage("main"))
        captured[1]["usage_sink"](TokenUsage("editor"))
        assert [event.purpose for event in agent.drain_usage()] == [
            "main",
            "output_rewrite",
        ]
    assert [cfg["model"] for cfg in captured] == ["main", "editor"]
    assert [cfg["base_url"] for cfg in captured] == [
        "https://main.example/v1",
        "https://editor.example/v1",
    ]
    assert [cfg["api_key"] for cfg in captured] == [
        "main-profile-secret",
        "editor-profile-secret",
    ]
    if backend == "anthropic":
        assert captured[1]["prompt_caching"] is True


def test_doctor_checks_rewrite_credentials_without_printing_values(
    tmp_path, monkeypatch
):
    (tmp_path / "config.yaml").write_text(
        "llm: {model: main}\noutput_rewrite:\n  enabled: true\n"
        "  llm: {model: editor, api_key_env: EDITOR_KEY}\n"
    )
    monkeypatch.delenv("EDITOR_KEY", raising=False)
    profile = AgentProfile.load(tmp_path)
    assert required_environment_names(profile) == {"EDITOR_KEY"}
    report = diagnose_profile(profile)
    assert any("EDITOR_KEY is missing" in item.message for item in report.errors)
    assert any("experimental" in item.message for item in report.findings)
    (tmp_path / ".env").write_text("EDITOR_KEY=\n")
    monkeypatch.setenv("EDITOR_KEY", "ambient-secret")
    profile = AgentProfile.load(tmp_path)
    with pytest.raises(ConfigError, match="EDITOR_KEY"):
        Agent.from_profile(profile, llm=FakeLLMClient([]))
    assert "ambient-secret" not in repr(diagnose_profile(profile))


def test_only_daily_enables_rewriting_and_environment_can_disable_it(
    tmp_path, monkeypatch
):
    for name in ("MODEL", "BASE_URL", "API_KEY_ENV", "WORKSPACE", "REWRITE_ENABLED"):
        monkeypatch.delenv(f"LINGCORE_{name}", raising=False)
    for name in ("coding", "frontend", "daily", "teaching"):
        destination = initialize_profile(name, destination=tmp_path / name)
        profile = AgentProfile.load(destination)
        assert profile.output_rewrite.enabled is (name == "daily")
    daily = tmp_path / "daily"
    (daily / ".env").write_text(
        "LINGCORE_MODEL=main-model\nLINGCORE_BASE_URL=https://main.example/v1\n"
        "LINGCORE_API_KEY_ENV=MAIN_KEY\nLINGCORE_REWRITE_MODEL=editor\n"
        "LINGCORE_REWRITE_BASE_URL=https://editor.example/v1\n"
        "LINGCORE_REWRITE_API_KEY_ENV=EDITOR_KEY\nLINGCORE_REWRITE_ENABLED=false\n"
    )
    profile = AgentProfile.load(daily)
    assert profile.output_rewrite.enabled is False
    assert profile.output_rewrite.llm.model == "editor"
    assert profile.output_rewrite.llm.base_url == "https://editor.example/v1"
    assert profile.output_rewrite.llm.api_key_env == "EDITOR_KEY"
    assert profile.llm.model == "main-model"
    assert profile.llm.base_url == "https://main.example/v1"
    assert profile.llm.api_key_env == "MAIN_KEY"
