"""Provider-seam and SiliconFlow wire-contract tests for knowledge retrieval."""

from __future__ import annotations

import pytest

from lingcore.errors import ConfigError
from lingcore.knowledge import (
    SiliconFlowEmbeddingProvider,
    SiliconFlowRerankingProvider,
    build_embedding_provider,
    embedding_enabled,
)


class StubEmbeddingProvider(SiliconFlowEmbeddingProvider):
    def __init__(self, **kwargs):
        super().__init__(api_key="test-key", **kwargs)
        self.requests: list[tuple[str, dict]] = []

    async def _post(self, endpoint: str, payload: dict) -> dict:
        self.requests.append((endpoint, payload))
        # Deliberately reverse provider order: the adapter must restore indices.
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }


class StubRerankingProvider(SiliconFlowRerankingProvider):
    def __init__(self, **kwargs):
        super().__init__(api_key="test-key", **kwargs)
        self.requests: list[tuple[str, dict]] = []

    async def _post(self, endpoint: str, payload: dict) -> dict:
        self.requests.append((endpoint, payload))
        return {
            "results": [
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.9},
            ]
        }


def test_embedding_is_opt_in_by_default():
    assert embedding_enabled({}) is False
    assert embedding_enabled({"embedding": False}) is False
    assert embedding_enabled({"embedding": {}}) is False
    assert embedding_enabled({"embedding": True}) is True


def test_provider_key_is_resolved_only_after_opt_in(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="not set"):
        build_embedding_provider({"embedding": True})


def test_provider_accepts_profile_scoped_environment(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "process-secret")

    provider = build_embedding_provider(
        {"embedding": True},
        environment={"SILICONFLOW_API_KEY": "profile-secret"},
    )

    assert provider.api_key == "profile-secret"


def test_provider_rejects_unknown_embedding_options():
    with pytest.raises(ConfigError, match="unknown"):
        embedding_enabled({"embedding": {"enabled": True, "typo": 1}})


async def test_siliconflow_embedding_wire_shape_and_order():
    provider = StubEmbeddingProvider(
        model="Qwen/Qwen3-VL-Embedding-8B",
        dimensions=768,
        batch_size=2,
    )
    vectors = await provider.embed(["first", {"text": "second"}])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    endpoint, payload = provider.requests[0]
    assert endpoint == "embeddings"
    assert payload == {
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "input": ["first", {"text": "second"}],
        "encoding_format": "float",
        "dimensions": 768,
    }


async def test_siliconflow_reranker_wire_shape():
    provider = StubRerankingProvider(model="Qwen/Qwen3-VL-Reranker-8B")
    results = await provider.rerank("query", ["one", "two"], top_n=2)
    assert [(result.index, result.score) for result in results] == [
        (1, 0.9),
        (0, 0.2),
    ]
    endpoint, payload = provider.requests[0]
    assert endpoint == "rerank"
    assert payload == {
        "model": "Qwen/Qwen3-VL-Reranker-8B",
        "query": "query",
        "documents": ["one", "two"],
        "top_n": 2,
        "return_documents": False,
    }
