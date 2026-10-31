"""Fixtures and builders for plugin foundation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lingcore.config import AgentProfile


def write_plugin(root: Path, name: str, **fields):
    root.mkdir(parents=True, exist_ok=True)
    (root / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "1.0.0", "api": 1, **fields})
    )
    return root


def profile(root: Path, **fields):
    result = AgentProfile(llm={"model": "test"}, **fields)
    result._source_dir = root
    return result


def load_plugins_for(root: Path, **fields):
    from lingcore.plugins.loader import load_plugins

    return load_plugins(profile(root, **fields))


@pytest.fixture
def isolated_tool_catalog(monkeypatch):
    from lingcore.tools import REGISTRY

    monkeypatch.setattr(REGISTRY, "_tools", dict(REGISTRY._tools))
    return REGISTRY


def _write_external_function_plugin(tmp_path, monkeypatch):
    """Register an imported callable, whose __module__ does not identify plugin ownership."""
    from lingcore.plugins import discovery

    package = tmp_path / "provenance_external_package"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from pydantic import BaseModel\n"
        "from lingcore.tools import ToolContext\n"
        'class Args(BaseModel):\n    text: str = "hello"\n'
        "async def external_echo(args: Args, ctx: ToolContext) -> str:\n"
        "    return args.text\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    root = write_plugin(
        tmp_path / "plugins" / "provenance-test",
        "provenance-test",
        module="tools.py",
        provides=["provenance_test_echo"],
    )
    (root / "tools.py").write_text(
        "from provenance_external_package import external_echo\n"
        "from lingcore.tools import tool\n"
        'echo = tool(name="provenance_test_echo", description="echo")(external_echo)\n'
    )
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    return root


LOCAL_TOOL = """from __future__ import annotations
from pydantic import BaseModel
from lingcore.tools import ToolContext, tool


class A(BaseModel):
    pass


@tool(name="shadowdemo_run")
async def run(args: A, ctx: ToolContext) -> str:
    \"\"\"Local replacement.\"\"\"
    return "local"
"""
