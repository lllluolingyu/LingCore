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
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    # Transient-failure retries when opening the stream (429 / 5xx / connection
    # errors). Delegated to the OpenAI SDK, which honors Retry-After and the
    # x-should-retry hint. 0 disables retrying. Raise it for an unstable
    # endpoint; lower it for a local server you'd rather fail fast against.
    max_retries: int = Field(default=10, ge=0)
    # Re-request budget for a response that fails *mid-stream*: a connection
    # drop, a stall past `timeout`, or a stream that ends without a finish
    # reason (truncation). max_retries never covers these — the SDK only
    # retries opening the request — so the agent loop recovers instead: it
    # discards the partial turn (never committed to memory), emits a
    # StreamRetry event so the frontend can mark the rupture, and asks again,
    # at most this many times per request. 0 disables mid-stream recovery.
    stream_retries: int = Field(default=3, ge=0)
    # Per-request inactivity timeout (seconds): httpx's read window — the
    # longest gap with no bytes received, including the wait for the first
    # token, before an attempt fails. NOT a wall-clock cap on the full streamed
    # response (which runs as long as tokens keep arriving), and across
    # max_retries attempts + backoff the total wait can still reach minutes; it
    # just stops one *stalled* attempt from hanging on the SDK's 600s default.
    # Raise it if a slow model legitimately pauses between tokens.
    timeout: float = Field(default=120.0, gt=0)

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
    include: list[str] = Field(default_factory=list)


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


class SessionsCfg(BaseModel):
    """Session-history persistence settings.

    The DB path resolves exactly like the memory tool's file (invariant 12):
    relative paths are confined to the profile directory, absolute paths
    require ``allow_absolute_path: true``, and a path inside the installed
    package tree disables persistence gracefully.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    path: str = "sessions.db"
    allow_absolute_path: bool = False


class AgentProfile(BaseModel):
    """A complete, validated agent definition loaded from YAML.

    ``extra="forbid"`` makes a typo'd top-level key a loud ConfigError instead
    of a silently ignored setting — important DX for a config-driven framework.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "agent"
    # Where the agent's file tools (and shell) operate. Unset (the default)
    # resolves to a ``workspace/`` subdirectory of the profile directory —
    # per-profile scratch space alongside sessions.db and memory.md — so a
    # bare launch never adopts whatever directory the user happened to be in.
    workspace: str | None = None
    llm: LLMCfg
    persona: PersonaCfg = Field(default_factory=PersonaCfg)
    tools: list[str] = Field(default_factory=list)
    # Skills the profile *statically engages*: their bundled tool code is loaded
    # and their instructions are injected as a prompt layer (always-on). Distinct
    # from the model-invoked ``activate_skill`` tool (dynamic). A profile may use
    # either, both, or neither.
    skills: list[str] = Field(default_factory=list)
    tool_options: dict[str, Any] = Field(default_factory=dict)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    loop: LoopCfg = Field(default_factory=LoopCfg)
    guardrail: GuardrailCfg = Field(default_factory=GuardrailCfg)
    sessions: SessionsCfg = Field(default_factory=SessionsCfg)

    @field_validator("workspace")
    @classmethod
    def _blank_workspace_is_unset(cls, v: str | None) -> str | None:
        """Treat a blank workspace exactly like an omitted one.

        Bundled profiles write ``workspace: ${LINGCORE_WORKSPACE:-}`` — the
        expansion yields ``""`` when the env var is unset, which must mean
        "use the per-profile default", not "the empty path".
        """
        if v is not None and not v.strip():
            return None
        return v

    @classmethod
    def load(cls, path: str | Path) -> "AgentProfile":
        """Load and validate a profile YAML, expanding ${ENV} references.

        ``path`` may be a directory (``config.yaml`` inside is used) or a
        direct path to any YAML file.  The resolved source directory is stored
        in ``_source_dir`` for later use (e.g. resolving prompt layer files).
        """
        p = Path(path)
        if p.is_dir():
            p = p / "config.yaml"
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
            profile = cls.model_validate(expanded)
        except Exception as e:
            # Pydantic's ValidationError is verbose but precise; wrap its
            # message so callers see a single ConfigError type.
            raise ConfigError(f"invalid profile {path}: {e}") from None
        # Store the directory so callers can resolve sibling files (prompt
        # layers, memory.md, skills/) without knowing the load path.
        object.__setattr__(profile, "_source_dir", p.parent.resolve())
        return profile

    def workspace_path(self, base: Path | None = None) -> Path:
        """Resolve the workspace directory.

        Unset (``None``/blank) defaults to a ``workspace/`` subdirectory of
        the profile directory (falling back to ``base``, then the CWD, for a
        profile constructed without a source file). An *explicit* relative
        path resolves against ``base`` — the user's CWD, not the profile dir
        (invariant 8); absolute paths are taken as-is.
        """
        if self.workspace is None:
            root = getattr(self, "_source_dir", None) or base or Path.cwd()
            return (root / "workspace").resolve()
        ws = Path(self.workspace).expanduser()
        if not ws.is_absolute() and base is not None:
            ws = base / ws
        return ws.resolve()
