"""Provider seams and HTTP adapters for knowledge retrieval.

The knowledge tool owns chunking and its local SQLite index.  This module owns
the remote model boundary: small protocols that alternate embedding/reranking
providers can implement, plus the SiliconFlow implementation used by the
bundled configuration example.

Secrets are never accepted as configuration values.  Provider builders read a
key from the environment variable *named* by ``api_key_env`` only when the
corresponding opt-in feature is actually used.
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

import httpx

from lingcore.errors import ConfigError, ToolError

EmbeddingInput: TypeAlias = str | dict[str, str]

DEFAULT_PROVIDER = "siliconflow"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-8B"

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_ALLOWED_EMBEDDING_KEYS = frozenset({
    "enabled",
    "provider",
    "base_url",
    "api_key_env",
    "model",
    "dimensions",
    "batch_size",
    "timeout",
    "max_retries",
})
_ALLOWED_RERANKER_KEYS = frozenset({
    "enabled",
    "provider",
    "base_url",
    "api_key_env",
    "model",
    "timeout",
    "max_retries",
    "candidate_count",
})


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-independent async embedding boundary."""

    @property
    def identity(self) -> str:
        """Stable model/config identity used to invalidate stored vectors."""

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[list[float]]:
        """Return one vector per input, in input order."""


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


@runtime_checkable
class RerankingProvider(Protocol):
    """Provider-independent async reranking boundary."""

    @property
    def identity(self) -> str: ...

    async def rerank(
        self,
        query: EmbeddingInput,
        documents: Sequence[EmbeddingInput],
        *,
        top_n: int,
    ) -> list[RerankResult]:
        """Return ranked input indices with relevance scores."""


def _require_mapping(value: Any, *, option: str) -> dict[str, Any]:
    """Normalize ``false``/``true`` shorthand and validate an option block."""
    if value is None or value is False:
        return {"enabled": False}
    if value is True:
        return {"enabled": True}
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"tool_options.knowledge.{option} must be a boolean or mapping"
        )
    result = dict(value)
    enabled = result.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError(
            f"tool_options.knowledge.{option}.enabled must be true or false"
        )
    return result


def embedding_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized embedding block (disabled when omitted)."""
    cfg = _require_mapping(options.get("embedding"), option="embedding")
    unknown = set(cfg) - _ALLOWED_EMBEDDING_KEYS
    if unknown:
        raise ConfigError(
            "unknown tool_options.knowledge.embedding option(s): "
            + ", ".join(sorted(unknown))
        )
    return cfg


def reranker_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized reranker block (disabled when omitted)."""
    cfg = _require_mapping(options.get("reranker"), option="reranker")
    unknown = set(cfg) - _ALLOWED_RERANKER_KEYS
    if unknown:
        raise ConfigError(
            "unknown tool_options.knowledge.reranker option(s): "
            + ", ".join(sorted(unknown))
        )
    return cfg


def embedding_enabled(options: Mapping[str, Any]) -> bool:
    """Whether semantic retrieval was explicitly opted into."""
    return bool(embedding_options(options).get("enabled", False))


def reranker_enabled(options: Mapping[str, Any]) -> bool:
    """Whether the optional second-stage reranker was explicitly opted into."""
    return bool(reranker_options(options).get("enabled", False))


def _positive_int(
    cfg: Mapping[str, Any], name: str, default: int, *, maximum: int | None = None
) -> int:
    raw = cfg.get(name, default)
    if isinstance(raw, bool):
        raise ConfigError(f"knowledge provider option {name!r} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"knowledge provider option {name!r} must be an integer"
        ) from None
    if value < 1 or (maximum is not None and value > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ConfigError(
            f"knowledge provider option {name!r} must be >= 1{suffix}"
        )
    return value


def _non_negative_int(cfg: Mapping[str, Any], name: str, default: int) -> int:
    raw = cfg.get(name, default)
    if isinstance(raw, bool):
        raise ConfigError(f"knowledge provider option {name!r} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"knowledge provider option {name!r} must be an integer"
        ) from None
    if value < 0:
        raise ConfigError(f"knowledge provider option {name!r} must be >= 0")
    return value


def _positive_float(cfg: Mapping[str, Any], name: str, default: float) -> float:
    raw = cfg.get(name, default)
    if isinstance(raw, bool):
        raise ConfigError(f"knowledge provider option {name!r} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"knowledge provider option {name!r} must be a number"
        ) from None
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"knowledge provider option {name!r} must be > 0")
    return value


def _required_text(cfg: Mapping[str, Any], name: str, default: str) -> str:
    raw = cfg.get(name, default)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"knowledge provider option {name!r} must be non-empty")
    return raw.strip()


def _resolve_api_key(
    cfg: Mapping[str, Any], environment: Mapping[str, str] | None = None
) -> str:
    # Explicit key material is forbidden even though tool_options is otherwise
    # intentionally open-ended.  Keeping this check here prevents a tempting
    # provider-specific shortcut from violating the profile secret invariant.
    if "api_key" in cfg:
        raise ConfigError(
            "knowledge provider secrets must not live in profile YAML; "
            "set api_key_env to the name of an environment variable"
        )
    env_name = _required_text(cfg, "api_key_env", DEFAULT_API_KEY_ENV)
    if environment is not None and env_name in environment:
        key = (environment or {}).get(env_name)
    else:
        key = os.environ.get(env_name)
    if not key:
        raise ConfigError(
            f"knowledge provider api_key_env names {env_name!r} but that "
            "variable is not set in the profile .env or process environment"
        )
    return key


def embedding_identity(options: Mapping[str, Any]) -> str | None:
    """Return the configured vector fingerprint without resolving a secret."""
    cfg = embedding_options(options)
    if not cfg.get("enabled", False):
        return None
    provider = _required_text(cfg, "provider", DEFAULT_PROVIDER)
    if provider != DEFAULT_PROVIDER:
        raise ConfigError(f"unknown knowledge embedding provider: {provider!r}")
    base_url = _required_text(cfg, "base_url", DEFAULT_BASE_URL).rstrip("/")
    model = _required_text(cfg, "model", DEFAULT_EMBEDDING_MODEL)
    dimensions = cfg.get("dimensions")
    if dimensions is not None:
        dimensions = _positive_int(cfg, "dimensions", 1, maximum=65_536)
    return f"{provider}:{base_url}:{model}:dimensions={dimensions or 'native'}"


@dataclass(slots=True)
class _SiliconFlowHTTP:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 60.0
    max_retries: int = 2

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        break
                else:
                    if response.status_code < 400:
                        try:
                            body = response.json()
                        except ValueError:
                            raise ToolError(
                                f"knowledge provider returned invalid JSON for /{endpoint}"
                            ) from None
                        if not isinstance(body, dict):
                            raise ToolError(
                                f"knowledge provider returned an invalid /{endpoint} response"
                            )
                        return body
                    if (
                        response.status_code not in _RETRYABLE_STATUS
                        or attempt >= self.max_retries
                    ):
                        raise ToolError(
                            "knowledge provider request failed: "
                            f"/{endpoint} returned HTTP {response.status_code}"
                        )
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                # Small capped exponential delay.  Provider calls are opt-in;
                # retries make transient 429/5xx failures tolerable without
                # turning a single tool invocation into a minutes-long wait.
                await asyncio.sleep(min(2.0, 0.25 * (2**attempt)))
        kind = type(last_error).__name__ if last_error is not None else "error"
        raise ToolError(f"knowledge provider request failed: {kind}")


@dataclass(slots=True)
class SiliconFlowEmbeddingProvider(_SiliconFlowHTTP):
    """SiliconFlow ``POST /embeddings`` adapter."""

    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int | None = None
    batch_size: int = 32

    @property
    def identity(self) -> str:
        return (
            f"{DEFAULT_PROVIDER}:{self.base_url.rstrip('/')}:{self.model}:"
            f"dimensions={self.dimensions or 'native'}"
        )

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[list[float]]:
        if not inputs:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = list(inputs[start : start + self.batch_size])
            payload: dict[str, Any] = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
            }
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions
            body = await self._post("embeddings", payload)
            raw_data = body.get("data")
            if not isinstance(raw_data, list) or len(raw_data) != len(batch):
                raise ToolError(
                    "knowledge embedding provider returned the wrong vector count"
                )
            try:
                ordered = sorted(raw_data, key=lambda item: int(item["index"]))
            except (KeyError, TypeError, ValueError):
                raise ToolError(
                    "knowledge embedding provider returned invalid vector indices"
                ) from None
            for expected, item in enumerate(ordered):
                if not isinstance(item, Mapping) or int(item.get("index", -1)) != expected:
                    raise ToolError(
                        "knowledge embedding provider returned non-contiguous indices"
                    )
                raw_vector = item.get("embedding")
                if not isinstance(raw_vector, list) or not raw_vector:
                    raise ToolError(
                        "knowledge embedding provider returned an empty vector"
                    )
                try:
                    vector = [float(value) for value in raw_vector]
                except (TypeError, ValueError):
                    raise ToolError(
                        "knowledge embedding provider returned a non-numeric vector"
                    ) from None
                if any(not math.isfinite(value) for value in vector):
                    raise ToolError(
                        "knowledge embedding provider returned a non-finite vector"
                    )
                vectors.append(vector)
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ToolError(
                "knowledge embedding provider returned inconsistent vector dimensions"
            )
        return vectors


@dataclass(slots=True)
class SiliconFlowRerankingProvider(_SiliconFlowHTTP):
    """SiliconFlow ``POST /rerank`` adapter."""

    model: str = DEFAULT_RERANKER_MODEL

    @property
    def identity(self) -> str:
        return f"{DEFAULT_PROVIDER}:{self.base_url.rstrip('/')}:{self.model}"

    async def rerank(
        self,
        query: EmbeddingInput,
        documents: Sequence[EmbeddingInput],
        *,
        top_n: int,
    ) -> list[RerankResult]:
        if not documents:
            return []
        top_n = min(max(1, top_n), len(documents))
        body = await self._post(
            "rerank",
            {
                "model": self.model,
                "query": query,
                "documents": list(documents),
                "top_n": top_n,
                "return_documents": False,
            },
        )
        raw_results = body.get("results")
        if not isinstance(raw_results, list):
            raise ToolError("knowledge reranker returned an invalid result list")
        results: list[RerankResult] = []
        seen: set[int] = set()
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                raise ToolError("knowledge reranker returned an invalid result")
            try:
                index = int(raw["index"])
                score = float(raw["relevance_score"])
            except (KeyError, TypeError, ValueError):
                raise ToolError("knowledge reranker returned an invalid result") from None
            if (
                index < 0
                or index >= len(documents)
                or index in seen
                or not math.isfinite(score)
            ):
                raise ToolError("knowledge reranker returned an invalid result")
            seen.add(index)
            results.append(RerankResult(index=index, score=score))
        return sorted(results, key=lambda result: (-result.score, result.index))[
            :top_n
        ]


def build_embedding_provider(
    options: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> EmbeddingProvider:
    """Build the configured embedder after the explicit opt-in gate."""
    cfg = embedding_options(options)
    if not cfg.get("enabled", False):
        raise ConfigError(
            "embedding retrieval is disabled; set "
            "tool_options.knowledge.embedding.enabled: true to opt in"
        )
    provider = _required_text(cfg, "provider", DEFAULT_PROVIDER)
    if provider != DEFAULT_PROVIDER:
        raise ConfigError(f"unknown knowledge embedding provider: {provider!r}")
    dimensions: int | None = None
    if cfg.get("dimensions") is not None:
        dimensions = _positive_int(cfg, "dimensions", 1, maximum=65_536)
    return SiliconFlowEmbeddingProvider(
        api_key=_resolve_api_key(cfg, environment),
        base_url=_required_text(cfg, "base_url", DEFAULT_BASE_URL),
        timeout=_positive_float(cfg, "timeout", 60.0),
        max_retries=_non_negative_int(cfg, "max_retries", 2),
        model=_required_text(cfg, "model", DEFAULT_EMBEDDING_MODEL),
        dimensions=dimensions,
        batch_size=_positive_int(cfg, "batch_size", 32, maximum=256),
    )


def build_reranking_provider(
    options: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> RerankingProvider:
    """Build the configured reranker after its explicit opt-in gate."""
    cfg = reranker_options(options)
    if not cfg.get("enabled", False):
        raise ConfigError(
            "knowledge reranking is disabled; set "
            "tool_options.knowledge.reranker.enabled: true to opt in"
        )
    provider = _required_text(cfg, "provider", DEFAULT_PROVIDER)
    if provider != DEFAULT_PROVIDER:
        raise ConfigError(f"unknown knowledge reranker provider: {provider!r}")
    return SiliconFlowRerankingProvider(
        api_key=_resolve_api_key(cfg, environment),
        base_url=_required_text(cfg, "base_url", DEFAULT_BASE_URL),
        timeout=_positive_float(cfg, "timeout", 60.0),
        max_retries=_non_negative_int(cfg, "max_retries", 2),
        model=_required_text(cfg, "model", DEFAULT_RERANKER_MODEL),
    )
