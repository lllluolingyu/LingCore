"""Tests for grep, semantic, and hybrid knowledge retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import pytest

from lingcore.errors import ConfigError, ToolError
from lingcore.knowledge import EmbeddingInput, RerankResult
from lingcore.paths import ConfinedDirectory
from lingcore.tools import ToolContext
from lingcore.tools.builtin.knowledge import (
    EMBEDDING_PROVIDER_KEY,
    RERANKING_PROVIDER_KEY,
    knowledge,
)


class FakeEmbedder:
    identity = "fake:concepts:v1"

    def __init__(self) -> None:
        self.calls: list[list[EmbeddingInput]] = []

    @staticmethod
    def _vector(value: EmbeddingInput) -> list[float]:
        text = value if isinstance(value, str) else str(value)
        lowered = text.lower()
        if "cat" in lowered or "feline" in lowered:
            return [1.0, 0.0, 0.0]
        if "dog" in lowered or "canine" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[list[float]]:
        self.calls.append(list(inputs))
        return [self._vector(value) for value in inputs]


class FakeReranker:
    identity = "fake:reranker:v1"

    def __init__(self) -> None:
        self.calls: list[tuple[EmbeddingInput, list[EmbeddingInput], int]] = []

    async def rerank(
        self,
        query: EmbeddingInput,
        documents: Sequence[EmbeddingInput],
        *,
        top_n: int,
    ) -> list[RerankResult]:
        self.calls.append((query, list(documents), top_n))
        return [
            RerankResult(index=index, score=1.0 - (rank * 0.1))
            for rank, index in enumerate(reversed(range(len(documents))))
        ][:top_n]


def _ctx(
    tmp_path: Path,
    opts: dict | None = None,
    *,
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
) -> ToolContext:
    options: dict = {"knowledge": opts or {}}
    if embedder is not None:
        options[EMBEDDING_PROVIDER_KEY] = embedder
    if reranker is not None:
        options[RERANKING_PROVIDER_KEY] = reranker
    return ToolContext(workspace=tmp_path, options=options)


def _indexed_opts(backend: str = "index", **extra) -> dict:
    return {
        "backend": backend,
        "embedding": {"enabled": True},
        "chunk_chars": 200,
        **extra,
    }


async def test_query_finds_substring(tmp_path):
    (tmp_path / "a.txt").write_text("the quick brown fox\nlazy dog", encoding="utf-8")
    ctx = _ctx(tmp_path)
    result = await knowledge(knowledge.args_model(action="query", query="fox"), ctx)
    assert "a.txt:1" in result and "fox" in result


async def test_query_regex(tmp_path):
    (tmp_path / "a.txt").write_text("error: 404\nok: 200", encoding="utf-8")
    ctx = _ctx(tmp_path)
    result = await knowledge(knowledge.args_model(action="query", query=r"\d{3}"), ctx)
    assert "404" in result


async def test_query_no_matches(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    ctx = _ctx(tmp_path)
    result = await knowledge(knowledge.args_model(action="query", query="zzz"), ctx)
    assert "no matches" in result


async def test_query_requires_query(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(ToolError, match="requires a query"):
        await knowledge(knowledge.args_model(action="query"), ctx)


async def test_status(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")
    ctx = _ctx(tmp_path)
    result = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "source files=2" in result
    assert "embedding=false" in result


async def test_source_escape_rejected(tmp_path):
    ctx = _ctx(tmp_path, {"sources": ["../*.txt"]})
    with pytest.raises(ToolError, match="escapes workspace"):
        await knowledge(knowledge.args_model(action="query", query="x"), ctx)


async def test_index_backend_requires_explicit_embedding_opt_in(tmp_path):
    ctx = _ctx(tmp_path, {"backend": "index"})
    with pytest.raises(ConfigError, match="disabled by default"):
        await knowledge(knowledge.args_model(action="query", query="x"), ctx)
    assert not (tmp_path / ".lingcore" / "knowledge.sqlite3").exists()


async def test_grep_index_is_noop(tmp_path):
    ctx = _ctx(tmp_path, {"backend": "grep"})
    result = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "no index" in result


async def test_sources_glob_scoping(tmp_path):
    (tmp_path / "keep.md").write_text("target", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("target", encoding="utf-8")
    ctx = _ctx(tmp_path, {"sources": ["*.md"]})
    result = await knowledge(knowledge.args_model(action="query", query="target"), ctx)
    assert "keep.md" in result and "skip.txt" not in result


async def test_semantic_index_query_returns_ranked_citations(tmp_path):
    (tmp_path / "cats.txt").write_text(
        "Felines purr and climb trees.", encoding="utf-8"
    )
    (tmp_path / "dogs.txt").write_text(
        "Canines bark and enjoy walks.", encoding="utf-8"
    )
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)

    indexed = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "new=2" in indexed
    assert "embedded=2" in indexed

    result = await knowledge(
        knowledge.args_model(action="query", query="dog behavior"), ctx
    )
    assert result.startswith("[1] dogs.txt:1")
    assert "Canines bark" in result
    assert "score=" in result

    status = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "indexed files=2" in status
    assert "embedded chunks=2" in status
    assert "stale files=0" in status


async def test_incremental_index_skips_unchanged_and_updates_changed_file(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A feline note.", encoding="utf-8")
    b.write_text("A canine note.", encoding="utf-8")
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)

    await knowledge(knowledge.args_model(action="index"), ctx)
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 2

    embedder.calls.clear()
    unchanged = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "unchanged=2" in unchanged
    assert "embedded=0" in unchanged
    assert embedder.calls == []

    a.write_text("A feline note with a changed fact.", encoding="utf-8")
    changed = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "updated=1" in changed
    assert "unchanged=1" in changed
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == ["A feline note with a changed fact."]


async def test_full_index_removes_deleted_sources(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("feline", encoding="utf-8")
    b.write_text("canine", encoding="utf-8")
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)
    await knowledge(knowledge.args_model(action="index"), ctx)

    b.unlink()
    report = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "deleted=1" in report
    status = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "indexed files=1" in status
    assert "chunks=1" in status


async def test_content_hash_detects_stale_file_with_preserved_size_and_mtime(tmp_path):
    path = tmp_path / "facts.txt"
    path.write_text("feline", encoding="utf-8")
    original = path.stat()
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)
    await knowledge(knowledge.args_model(action="index"), ctx)

    path.write_text("canine", encoding="utf-8")  # same byte length
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    status = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "stale files=1" in status


async def test_renamed_unchanged_content_reuses_its_vector(tmp_path):
    old = tmp_path / "old.txt"
    old.write_text("feline", encoding="utf-8")
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)
    await knowledge(knowledge.args_model(action="index"), ctx)

    old.rename(tmp_path / "new.txt")
    embedder.calls.clear()
    report = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "new=1" in report and "deleted=1" in report
    assert "reused_vectors=1" in report
    assert embedder.calls == []


async def test_skipped_binary_source_does_not_leave_index_perpetually_stale(tmp_path):
    (tmp_path / "notes.txt").write_text("feline", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\x00\xff\x00")
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)
    report = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "skipped=1" in report
    status = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "stale files=0" in status


async def test_skipped_corrupt_pdf_does_not_leave_index_perpetually_stale(tmp_path):
    pytest.importorskip("fitz")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.7\nnot a valid PDF")
    embedder = FakeEmbedder()
    ctx = _ctx(
        tmp_path,
        _indexed_opts(sources=["*.pdf"]),
        embedder=embedder,
    )

    report = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "skipped=1" in report and "invalid PDF" in report
    status = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "stale files=0" in status
    result = await knowledge(
        knowledge.args_model(action="query", query="anything"), ctx
    )
    assert result == "(no matches)"


async def test_partial_index_only_updates_selected_paths(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("feline one", encoding="utf-8")
    b.write_text("canine one", encoding="utf-8")
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)
    await knowledge(knowledge.args_model(action="index"), ctx)

    a.write_text("feline two", encoding="utf-8")
    b.write_text("canine two", encoding="utf-8")
    embedder.calls.clear()
    await knowledge(
        knowledge.args_model(action="index", paths=["a.txt"]), ctx
    )
    assert embedder.calls == [["feline two"]]
    status = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "stale files=1" in status


async def test_partial_index_removes_a_selected_missing_path(tmp_path):
    path = tmp_path / "gone.txt"
    path.write_text("feline", encoding="utf-8")
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)
    await knowledge(knowledge.args_model(action="index"), ctx)
    path.unlink()

    report = await knowledge(
        knowledge.args_model(action="index", paths=["gone.txt"]), ctx
    )
    assert "deleted=1" in report
    status = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "indexed files=0" in status


async def test_stale_content_is_excluded_until_reindexed(tmp_path):
    path = tmp_path / "facts.txt"
    path.write_text("Old feline fact.", encoding="utf-8")
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts(), embedder=embedder)
    await knowledge(knowledge.args_model(action="index"), ctx)

    path.write_text("New feline fact that is not indexed.", encoding="utf-8")
    result = await knowledge(
        knowledge.args_model(action="query", query="cat"), ctx
    )
    assert result.startswith("[index stale: changed=1")
    assert "Old feline fact" not in result
    assert result.endswith("(no matches)")


async def test_hybrid_combines_full_text_and_semantic_candidates(tmp_path):
    (tmp_path / "literal.txt").write_text(
        "A cat literal in an otherwise canine document.", encoding="utf-8"
    )
    (tmp_path / "semantic.txt").write_text(
        "Feline behavior without the literal query token.", encoding="utf-8"
    )
    embedder = FakeEmbedder()
    ctx = _ctx(tmp_path, _indexed_opts("hybrid"), embedder=embedder)
    await knowledge(knowledge.args_model(action="index"), ctx)

    result = await knowledge(
        knowledge.args_model(action="query", query="cat"), ctx
    )
    assert "literal.txt:1" in result
    assert "semantic.txt:1" in result


async def test_optional_reranker_controls_final_order(tmp_path):
    (tmp_path / "first.txt").write_text("cat first", encoding="utf-8")
    (tmp_path / "second.txt").write_text("feline second", encoding="utf-8")
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    opts = _indexed_opts(
        reranker={"enabled": True, "candidate_count": 2}, max_hits=2
    )
    ctx = _ctx(tmp_path, opts, embedder=embedder, reranker=reranker)
    await knowledge(knowledge.args_model(action="index"), ctx)

    ctx.options["knowledge"]["reranker"]["candidate_count"] = 0
    with pytest.raises(
        ConfigError, match=r"tool_options\.knowledge\.reranker\.candidate_count"
    ):
        await knowledge(knowledge.args_model(action="query", query="cat"), ctx)
    ctx.options["knowledge"]["reranker"]["candidate_count"] = 2

    result = await knowledge(
        knowledge.args_model(action="query", query="cat"), ctx
    )
    assert result.startswith("[1] second.txt:1")
    assert reranker.calls[0][0] == "cat"
    assert reranker.calls[0][2] == 2


async def test_pdf_index_preserves_page_and_line_citation(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "A feline fact on a PDF page")
    pdf.save(tmp_path / "notes.pdf")
    pdf.close()
    embedder = FakeEmbedder()
    ctx = _ctx(
        tmp_path,
        _indexed_opts(sources=["*.pdf"]),
        embedder=embedder,
    )
    await knowledge(knowledge.args_model(action="index"), ctx)

    result = await knowledge(
        knowledge.args_model(action="query", query="cat"), ctx
    )
    assert "notes.pdf#page=1:lines=1" in result


async def test_index_path_and_subset_escapes_are_rejected(tmp_path):
    embedder = FakeEmbedder()
    ctx = _ctx(
        tmp_path,
        _indexed_opts(index_path="../outside.sqlite3"),
        embedder=embedder,
    )
    with pytest.raises(ToolError, match="escapes workspace"):
        await knowledge(knowledge.args_model(action="index"), ctx)

    safe = _ctx(tmp_path, _indexed_opts(), embedder=embedder)
    with pytest.raises(ToolError, match="escapes workspace"):
        await knowledge(
            knowledge.args_model(action="index", paths=["../secret.txt"]), safe
        )


async def test_index_final_symlink_is_never_followed(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("feline", encoding="utf-8")
    runtime = workspace / ".lingcore"
    runtime.mkdir()
    outside = tmp_path / "outside.sqlite3"
    try:
        (runtime / "knowledge.sqlite3").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported on this platform")
    ctx = _ctx(workspace, _indexed_opts(), embedder=FakeEmbedder())

    with pytest.raises(ToolError, match="must not be a symlink"):
        await knowledge(knowledge.args_model(action="index"), ctx)
    assert not outside.exists()


async def test_index_parent_swap_fails_closed(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("feline", encoding="utf-8")
    runtime = workspace / ".lingcore"
    held = tmp_path / "runtime-held"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = ConfinedDirectory.open_exclusive
    swapped = False

    def swap_then_open(self, name, mode=0o644):
        nonlocal swapped
        if name.endswith(".part") and not swapped:
            runtime.rename(held)
            runtime.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(self, name, mode)

    monkeypatch.setattr(ConfinedDirectory, "open_exclusive", swap_then_open)
    ctx = _ctx(workspace, _indexed_opts(), embedder=FakeEmbedder())

    with pytest.raises(ToolError, match="cannot safely write knowledge index"):
        await knowledge(knowledge.args_model(action="index"), ctx)
    assert not list(outside.iterdir())
    assert not list(held.iterdir())


async def test_source_parent_swap_cannot_redirect_index_read(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    docs = workspace / "docs"
    docs.mkdir(parents=True)
    (docs / "note.txt").write_text("feline inside", encoding="utf-8")
    held = tmp_path / "docs-held"
    outside = tmp_path / "outside-docs"
    outside.mkdir()
    (outside / "note.txt").write_text("canine secret outside", encoding="utf-8")
    real_read = ConfinedDirectory.read_regular_with_stat
    swapped = False

    def swap_then_read(self, name, *, max_bytes):
        nonlocal swapped
        if name == "note.txt" and not swapped:
            docs.rename(held)
            docs.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_read(self, name, max_bytes=max_bytes)

    monkeypatch.setattr(
        ConfinedDirectory, "read_regular_with_stat", swap_then_read
    )
    embedder = FakeEmbedder()
    ctx = _ctx(
        workspace,
        _indexed_opts(sources=["docs/*.txt"]),
        embedder=embedder,
    )

    report = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "skipped=1" in report
    assert embedder.calls == []
