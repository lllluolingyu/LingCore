"""Tests for the knowledge builtin tool (grep backend)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.errors import ConfigError, ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.knowledge import knowledge


def _ctx(tmp_path: Path, opts: dict | None = None) -> ToolContext:
    return ToolContext(workspace=tmp_path, options={"knowledge": opts or {}})


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


async def test_source_escape_rejected(tmp_path):
    ctx = _ctx(tmp_path, {"sources": ["../*.txt"]})
    with pytest.raises(ToolError, match="escapes workspace"):
        await knowledge(knowledge.args_model(action="query", query="x"), ctx)


async def test_index_backend_requires_numpy_or_unimplemented(tmp_path):
    ctx = _ctx(tmp_path, {"backend": "index"})
    # Either numpy is missing (ConfigError about numpy) or it's the
    # not-implemented ConfigError — both are ConfigError.
    with pytest.raises(ConfigError):
        await knowledge(knowledge.args_model(action="query", query="x"), ctx)


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
