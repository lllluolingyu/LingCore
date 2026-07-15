"""Declarative agent profiles.

A profile is the whole reason LingCore exists: the same runtime becomes a
coding agent, a role-play agent, or a psych consultant purely by loading a
different YAML file. This module defines the validated shape of that file and
``Agent.from_profile``'s backing assembly logic.

Secrets never live in profile YAML. ``llm.api_key_env`` names an environment
variable; its value may be exported by the parent process or kept in the
selected profile directory's optional ``.env`` file. The selected profile's
``.env`` wins, and LingCore never searches the CWD or parent directories for another
``.env``. String fields support ``${VAR}`` and ``${VAR:-default}`` expansion so
endpoints and model names can come from the same environment too.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lingcore.errors import ConfigError
from lingcore.media_types import FALLBACK_TEXT_MAX_CHARS, NativeModality
from lingcore.modality import DEFAULT_PDF_MAX_CHARS

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _load_profile_env(profile_file: Path) -> dict[str, str]:
    """Parse only ``<profile dir>/.env`` without mutating ``os.environ``.

    An explicit path is important here: ``python-dotenv`` can discover files by
    walking parent directories when no path is supplied, which would make a
    launch depend on the caller's CWD and could import variables from an
    unrelated checkout. Profile selection is the trust decision, so its exact
    sibling file is the only implicit source.
    """
    env_path = profile_file.parent / ".env"
    if not env_path.is_file():
        return {}
    try:
        # interpolate=False: this file's primary payload is secrets, and an
        # opaque token may legally contain ``${``. Composition belongs to the
        # YAML's own ${VAR} expansion, not to dotenv rewriting values.
        parsed = dotenv_values(
            dotenv_path=env_path,
            encoding="utf-8",
            interpolate=False,
        )
    except (OSError, UnicodeError) as e:
        raise ConfigError(
            f"could not read profile environment file {env_path}: {e}"
        ) from None
    # A bare key (``NAME`` with no '=') has value None in python-dotenv and is
    # intentionally ignored, matching load_dotenv's behavior.
    return {name: value for name, value in parsed.items() if value is not None}


def _expand(value: Any, environment: Mapping[str, str] | None = None) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in strings."""
    env = os.environ if environment is None else environment
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            raise ConfigError(
                f"environment variable {name!r} referenced in profile is not set "
                "and has no default"
            )

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, env) for v in value]
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
    # Attachment modalities the model natively accepts as content parts
    # ("image" -> image_url parts, "file" -> file/PDF parts); text is always
    # on. Defaults to both — today's behavior. Declare fewer for a model that
    # rejects media parts: unsupported attachments then degrade to text via
    # the profile's ``media_fallback`` section instead of erroring mid-turn.
    modalities: list[NativeModality] = Field(
        default_factory=lambda: ["image", "file"]
    )
    # Send an OpenAI ``prompt_cache_key`` (the session id) with every request so
    # same-session traffic routes to the same warm cache node — the routing lever
    # that lifts the realized prompt-cache hit rate on top of a stable prefix.
    # Off by default: a strict OpenAI-compatible server (Ollama/vLLM/proxy) may
    # reject the unknown body field. Flip it on for an endpoint that honors it.
    send_prompt_cache_key: bool = False

    @field_validator("modalities")
    @classmethod
    def _dedup_modalities(cls, v: list[NativeModality]) -> list[NativeModality]:
        return list(dict.fromkeys(v))

    def resolve_api_key(
        self, environment: Mapping[str, str] | None = None
    ) -> str:
        """Read the API key from the named env var.

        Returns a harmless placeholder when no env var is named — local
        OpenAI-compatible servers (Ollama, vLLM) ignore the key but the SDK
        requires a non-empty string.
        """
        if not self.api_key_env:
            return "lingcore-no-key"
        profile_env = (
            environment
            if environment is not None
            else getattr(self, "_profile_env", {})
        )
        if self.api_key_env in profile_env:
            key = profile_env.get(self.api_key_env)
        else:
            key = os.environ.get(self.api_key_env)
        if not key:
            raise ConfigError(
                f"api_key_env names {self.api_key_env!r} but that variable is "
                "not set in the profile .env or process environment"
            )
        return key


class PersonaCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str = "You are a helpful assistant."
    include: list[str] = Field(default_factory=list)


class MediaFallbackCfg(BaseModel):
    """Text fallbacks for attachment modalities the main model lacks.

    Consulted only for kinds missing from ``llm.modalities``. PDFs degrade to
    extracted text (needs the optional ``lingcore[pdf]`` extra — pymupdf);
    images degrade to a description from the secondary vision model declared
    in ``image`` (an ordinary ``llm`` block, so its key still comes from the
    environment via ``api_key_env``, never the YAML). When no fallback is
    available, the model receives a short note saying what it can't see.
    """

    model_config = ConfigDict(extra="forbid")

    pdf: Literal["markdown", "none"] = "markdown"
    pdf_max_chars: int = Field(
        default=DEFAULT_PDF_MAX_CHARS, ge=200, le=FALLBACK_TEXT_MAX_CHARS
    )
    image: LLMCfg | None = None
    image_prompt: str = (
        "Describe this image in detail for a text-only assistant. "
        "Transcribe any visible text verbatim."
    )
    image_max_chars: int = Field(default=4_000, ge=200, le=FALLBACK_TEXT_MAX_CHARS)

    @model_validator(mode="after")
    def _vision_model_must_see_images(self) -> "MediaFallbackCfg":
        if self.image is not None and "image" not in self.image.modalities:
            raise ValueError(
                "media_fallback.image names a vision model whose modalities "
                "exclude 'image' — it could never describe anything"
            )
        return self


class CompactionCfg(BaseModel):
    """Summarize-then-evict policy for a nearly-full window.

    When the working set reaches ``compact_at_ratio`` of ``max_tokens``, the
    oldest history is summarized by the model into one compact note while the
    most recent ``keep_recent_ratio`` of context is kept verbatim. If the result
    still exceeds the hard cap, ``WindowMemory``'s eviction floor takes over.
    Off by default — compaction costs one summarizer call; enable it per profile.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    compact_at_ratio: float = Field(default=0.85, gt=0, le=1.0)
    keep_recent_ratio: float = Field(default=0.35, gt=0, lt=1.0)
    max_summary_chars: int = Field(default=4_000, ge=200)

    @model_validator(mode="after")
    def _recent_below_trigger(self) -> "CompactionCfg":
        if self.keep_recent_ratio >= self.compact_at_ratio:
            raise ValueError(
                "compaction.keep_recent_ratio must be < compact_at_ratio, else "
                "compaction could never shrink the window"
            )
        return self


class MemoryCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_messages: int = 40
    max_tokens: int = 12_000
    # Fraction of ``max_tokens`` the window keeps after an eviction. Eviction is
    # chunked down to this low-water mark (hysteresis) only when a hard cap is
    # breached, so the rendered prefix stays byte-stable across many turns
    # instead of sliding every request — the key to a high prompt-cache hit
    # rate. ``1.0`` reproduces the legacy trim-to-the-cap-every-render behavior.
    evict_to_ratio: float = Field(default=0.5, gt=0, le=1.0)
    compaction: CompactionCfg = Field(default_factory=CompactionCfg)


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
    # Tools enabled WITHOUT any skill active — the "initially enabled" set.
    # ``None`` (default) means all of ``tools`` are initially enabled (today's
    # behavior). Set it to a subset to gate the rest behind skill activation
    # (progressive disclosure / least privilege): ``tools`` stays the hard
    # ceiling, a skill can unlock any ceiling tool it requests, but a tool that
    # is neither initial nor granted by an active skill is neither advertised
    # nor dispatchable.
    initial_tools: list[str] | None = None
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
    media_fallback: MediaFallbackCfg = Field(default_factory=MediaFallbackCfg)

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

    @model_validator(mode="after")
    def _initial_tools_within_ceiling(self) -> "AgentProfile":
        """initial_tools must be a subset of tools (the hard ceiling)."""
        if self.initial_tools is not None:
            extra = [t for t in self.initial_tools if t not in self.tools]
            if extra:
                raise ValueError(
                    f"initial_tools {extra} are not listed in tools (the ceiling)"
                )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "AgentProfile":
        """Load a profile's ``.env`` and YAML, then expand ${ENV} references.

        ``path`` may be a directory (``config.yaml`` inside is used) or a
        direct path to any YAML file. Only ``.env`` beside that selected YAML is
        considered; its values take precedence over exported variables. The resolved
        source directory is stored in ``_source_dir`` for later use (e.g.
        resolving prompt layer files).
        """
        p = Path(path)
        if p.is_dir():
            p = p / "config.yaml"
        if not p.is_file():
            raise ConfigError(f"profile not found: {path}")
        # Must precede YAML parsing/expansion: .env values may supply `${VAR}`
        # fields as well as credentials resolved later by providers and tools.
        profile_env = _load_profile_env(p)
        try:
            raw = yaml.safe_load(p.read_text("utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {path}: {e}") from None
        if not isinstance(raw, dict):
            raise ConfigError(f"profile {path} must be a mapping at the top level")
        # The explicitly selected profile overrides ambient variables inherited
        # from the shell, without copying either source into process globals.
        # An empty profile value still blocks the exported variable but counts
        # as unset, so a ${VAR:-default} expansion falls back to its default
        # (invariant 4) instead of injecting an empty string.
        effective_env = {**os.environ, **profile_env}
        for name, value in profile_env.items():
            if not value:
                effective_env.pop(name, None)
        expanded = _expand(raw, effective_env)
        try:
            profile = cls.model_validate(expanded)
        except Exception as e:
            # Pydantic's ValidationError is verbose but precise; wrap its
            # message so callers see a single ConfigError type.
            raise ConfigError(f"invalid profile {path}: {e}") from None
        # Store the directory so callers can resolve sibling files (prompt
        # layers, memory.md, skills/) without knowing the load path.
        object.__setattr__(profile, "_source_dir", p.parent.resolve())
        object.__setattr__(profile, "_profile_env", profile_env)
        # Keep the convenient ``profile.llm.resolve_api_key()`` API working
        # while Agent.from_profile also passes the profile environment
        # explicitly at the composition boundary.
        object.__setattr__(profile.llm, "_profile_env", profile_env)
        if profile.media_fallback.image is not None:
            object.__setattr__(
                profile.media_fallback.image, "_profile_env", profile_env
            )
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
