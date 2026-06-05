"""Declarative agent profiles.

A profile is the whole reason LingCore exists: the same runtime becomes a
coding agent, a role-play agent, or a psych consultant purely by loading a
different YAML file. This module defines the validated shape of that file and
``Agent.from_profile``'s backing assembly logic.

Secrets never live in the profile. ``llm.api_key_env`` names an environment
variable; the key is read from the process environment at load time. String
fields support ``${VAR}`` and ``${VAR:-default}`` expansion so endpoints and
model names can come from the environment too.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lingcore.errors import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in strings."""
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(
                f"environment variable {name!r} referenced in profile is not set "
                "and has no default"
            )

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class SamplingCfg(BaseModel):
    """Sampling params passed through to the API.

    ``extra="allow"`` so any provider-specific knob (top_p, presence_penalty,
    reasoning_effort, ...) flows through without a schema change.
    """

    model_config = ConfigDict(extra="allow")

    temperature: float | None = None
    max_tokens: int | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class LLMCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = None
    sampling: SamplingCfg = Field(default_factory=SamplingCfg)

    def resolve_api_key(self) -> str:
        """Read the API key from the named env var.

        Returns a harmless placeholder when no env var is named — local
        OpenAI-compatible servers (Ollama, vLLM) ignore the key but the SDK
        requires a non-empty string.
        """
        if not self.api_key_env:
            return "lingcore-no-key"
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ConfigError(
                f"api_key_env names {self.api_key_env!r} but that variable is "
                "not set in the environment"
            )
        return key


class PersonaCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str = "You are a helpful assistant."


class MemoryCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_messages: int = 40
    max_tokens: int = 12_000


class LoopCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iters: int = 25
    parallel_tools: bool = True


class GuardrailCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: str = "noop"


class AgentProfile(BaseModel):
    """A complete, validated agent definition loaded from YAML.

    ``extra="forbid"`` makes a typo'd top-level key a loud ConfigError instead
    of a silently ignored setting — important DX for a config-driven framework.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "agent"
    workspace: str = "."
    llm: LLMCfg
    persona: PersonaCfg = Field(default_factory=PersonaCfg)
    tools: list[str] = Field(default_factory=list)
    tool_options: dict[str, Any] = Field(default_factory=dict)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    loop: LoopCfg = Field(default_factory=LoopCfg)
    guardrail: GuardrailCfg = Field(default_factory=GuardrailCfg)

    @classmethod
    def load(cls, path: str | Path) -> "AgentProfile":
        """Load and validate a profile YAML, expanding ${ENV} references."""
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"profile not found: {path}")
        try:
            raw = yaml.safe_load(p.read_text("utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {path}: {e}") from None
        if not isinstance(raw, dict):
            raise ConfigError(f"profile {path} must be a mapping at the top level")
        expanded = _expand(raw)
        try:
            return cls.model_validate(expanded)
        except Exception as e:
            # Pydantic's ValidationError is verbose but precise; wrap its
            # message so callers see a single ConfigError type.
            raise ConfigError(f"invalid profile {path}: {e}") from None

    def workspace_path(self, base: Path | None = None) -> Path:
        """Resolve the workspace, relative to ``base`` (e.g. the profile dir)."""
        ws = Path(self.workspace).expanduser()
        if not ws.is_absolute() and base is not None:
            ws = base / ws
        return ws.resolve()
