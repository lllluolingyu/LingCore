"""Knowledge options, data models, and provider validation."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, cast

from lingcore.errors import ConfigError, ToolError
from lingcore.knowledge import (
    EmbeddingProvider,
    RerankingProvider,
    build_embedding_provider,
    build_reranking_provider,
    embedding_enabled,
)
from lingcore.tools import ToolContext
from lingcore.tools.builtin._offload import RUNTIME_DIRNAME
from lingcore.tools.builtin._options import int_option

_MAX_READ_BYTES = 256 * 1024
_MAX_INDEX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_HITS = 50
_GREP_LINE_CHARS = 200
# Catastrophic regex backtracking is bounded per match and across the scan;
# the tool runs this synchronous pass in a worker thread so the event loop (and
# Stop handling) stays responsive while the regex engine is busy.
_DEFAULT_GREP_TIME_BUDGET_MS = 2_000
_GREP_MATCH_TIMEOUT_MS = 250
_DEFAULT_SOURCES = ["**/*"]
_DEFAULT_CHUNK_CHARS = 1_600
_DEFAULT_CHUNK_OVERLAP_LINES = 2
_DEFAULT_EXCERPT_CHARS = 1_000
_DEFAULT_CONTEXT_CHARS = 8_000
_DEFAULT_CANDIDATE_MULTIPLIER = 4
_DEFAULT_RRF_K = 60
_DEFAULT_INDEX_PATH = f"{RUNTIME_DIRNAME}/knowledge.sqlite3"
_MAX_INDEX_BYTES = 512 * 1024 * 1024
_SCHEMA_VERSION = 1

# Injection keys are public so alternate composition roots and tests can supply
# provider implementations without placing live Python objects inside YAML.
EMBEDDING_PROVIDER_KEY = "_knowledge_embedding_provider"
RERANKING_PROVIDER_KEY = "_knowledge_reranking_provider"


@dataclass(frozen=True, slots=True)
class _ChunkDraft:
    chunk_key: str
    ordinal: int
    page: int | None
    line_start: int
    line_end: int
    text_hash: str
    text: str


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    path: str
    content_hash: str
    size: int
    mtime_ns: int
    chunks: tuple[_ChunkDraft, ...]


@dataclass(frozen=True, slots=True)
class _SearchHit:
    chunk_id: int
    path: str
    ordinal: int
    page: int | None
    line_start: int
    line_end: int
    text: str
    score: float
    semantic_score: float | None = None


@dataclass(frozen=True, slots=True)
class _StaleState:
    changed: frozenset[str]
    deleted: frozenset[str]
    new: frozenset[str]

    @property
    def count(self) -> int:
        return len(self.changed | self.deleted | self.new)

    @property
    def excluded(self) -> frozenset[str]:
        return self.changed | self.deleted


def _confined(base: Path, path: Path) -> bool:
    return path == base or path.is_relative_to(base)


def _knowledge_options(ctx: ToolContext) -> dict[str, Any]:
    raw = ctx.options.get("knowledge", {})
    if not isinstance(raw, Mapping):
        raise ConfigError("tool_options.knowledge must be a mapping")
    return dict(raw)


def _backend(options: Mapping[str, Any]) -> Literal["grep", "index", "hybrid"]:
    value = options.get("backend", "grep")
    if value not in {"grep", "index", "hybrid"}:
        raise ConfigError(
            "tool_options.knowledge.backend must be grep, index, or hybrid"
        )
    return cast(Literal["grep", "index", "hybrid"], value)


def _sources(options: Mapping[str, Any]) -> list[str]:
    raw = options.get("sources", _DEFAULT_SOURCES)
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or not value.strip() for value in raw)
    ):
        raise ConfigError(
            "tool_options.knowledge.sources must be a non-empty list of globs"
        )
    return [value.strip() for value in raw]


def _int_option(
    options: Mapping[str, Any],
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    option_path: str = "tool_options.knowledge",
) -> int:
    return int_option(
        options,
        name,
        default,
        minimum=minimum,
        maximum=maximum,
        option_path=option_path,
    )


def _float_option(
    options: Mapping[str, Any], name: str, default: float, *, minimum: float = 0.0
) -> float:
    raw = options.get(name, default)
    if isinstance(raw, bool):
        raise ConfigError(f"tool_options.knowledge.{name} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"tool_options.knowledge.{name} must be a number") from None
    if not math.isfinite(value) or value < minimum:
        raise ConfigError(f"tool_options.knowledge.{name} must be >= {minimum}")
    return value


def _normalize_vector(values: Iterable[Any]) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ToolError("embedding provider returned a non-numeric vector") from None
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ToolError("embedding provider returned an empty or non-finite vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ToolError("embedding provider returned a zero-length vector")
    return [value / norm for value in vector]


def _pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(payload: bytes, dimensions: int) -> tuple[float, ...] | None:
    if dimensions <= 0 or len(payload) != dimensions * 4:
        return None
    try:
        return struct.unpack(f"<{dimensions}f", payload)
    except struct.error:
        return None


def _provider_identity(provider: Any) -> str:
    identity = getattr(provider, "identity", None)
    if not isinstance(identity, str) or not identity.strip():
        raise ConfigError("knowledge provider identity must be a non-empty string")
    return identity.strip()


def _embedder(ctx: ToolContext, options: Mapping[str, Any]) -> EmbeddingProvider:
    if not embedding_enabled(options):
        raise ConfigError(
            "embedding retrieval is disabled by default; set "
            "tool_options.knowledge.embedding.enabled: true to opt in"
        )
    provider = ctx.options.get(EMBEDDING_PROVIDER_KEY)
    if provider is None:
        provider = build_embedding_provider(options, environment=ctx.environment)
        ctx.options[EMBEDDING_PROVIDER_KEY] = provider
    if not isinstance(provider, EmbeddingProvider):
        raise ConfigError("injected knowledge embedder does not implement embed()")
    _provider_identity(provider)
    return provider


def _reranker(ctx: ToolContext, options: Mapping[str, Any]) -> RerankingProvider:
    provider = ctx.options.get(RERANKING_PROVIDER_KEY)
    if provider is None:
        provider = build_reranking_provider(options, environment=ctx.environment)
        ctx.options[RERANKING_PROVIDER_KEY] = provider
    if not isinstance(provider, RerankingProvider):
        raise ConfigError("injected knowledge reranker does not implement rerank()")
    _provider_identity(provider)
    return provider
