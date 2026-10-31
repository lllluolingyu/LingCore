"""Fixtures and builders for tools tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lingcore.tools import ToolContext


class _IntArgs(BaseModel):
    x: int


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return ToolContext(workspace=tmp_path)
