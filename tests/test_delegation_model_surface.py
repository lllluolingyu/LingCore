from __future__ import annotations

from pathlib import Path

from lingcore.superagent.tools import executor_tools, router_tools
from lingcore.tools import REGISTRY
from lingcore.tools.builtin import delegate, workflow  # noqa: F401


def test_shipped_delegation_model_surface_stays_bounded():
    names = (
        "delegate",
        "workflow_run",
        "workflow_resume",
        "workflow_status",
        "workflow_result",
        "workflow_stop",
    )
    tools = [REGISTRY.get(name) for name in names]
    tools += list(router_tools(None)) + list(
        executor_tools(None, "ticket", "assignment")
    )
    descriptions = sum(len(tool.description) for tool in tools)
    personas = sum(
        len(path.read_text())
        for profile in ("coding", "team")
        for path in Path(f"profiles/{profile}/roles").glob("*.md")
    )
    skill = Path(
        "lingcore/bundled_plugins/workflows/skills/workflows/skill.md"
    ).read_text()
    assert descriptions < 5000
    assert personas < 5000
    assert len(skill) < 6000
    assert descriptions + personas + len(skill) < 12000
