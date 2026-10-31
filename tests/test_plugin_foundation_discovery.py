"""Plugin foundation tests: discovery."""

from __future__ import annotations

import pytest

from lingcore.errors import ConfigError
from tests.support.plugin_foundation import (
    LOCAL_TOOL,
    _write_external_function_plugin,
    load_plugins_for,
    profile,
    write_plugin,
)
from tests.support.plugin_foundation import (
    isolated_tool_catalog as isolated_tool_catalog,
)


def test_numeric_minimum_version_compares_components():
    from lingcore.plugins.manifest import PluginManifest

    assert (
        PluginManifest(
            name="demo", version="1.0.0", min_lingcore="0.3.0"
        ).check_compatibility()
        is None
    )
    with pytest.raises(ConfigError, match="requires LingCore"):
        PluginManifest(
            name="demo", version="1.0.0", min_lingcore="0.10.0"
        ).check_compatibility()


def test_doctor_knows_tools_declared_by_unengaged_plugin(tmp_path, monkeypatch):
    from lingcore.doctor import diagnose_profile
    from lingcore.plugins import discovery

    root = write_plugin(
        tmp_path / "plugins" / "demo", "demo", module="tools.py", provides=["demo_run"]
    )
    (root / "tools.py").write_text("raise RuntimeError('unengaged plugin imported')")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    findings = diagnose_profile(profile(tmp_path, tools=["demo_run"])).findings
    assert not any(
        "unknown authorized tool" in f.message and "demo_run" in f.message
        for f in findings
    )


def test_doctor_reports_incompatible_unengaged_metadata(tmp_path, monkeypatch):
    from lingcore.doctor import diagnose_profile
    from lingcore.plugins import discovery

    write_plugin(tmp_path / "plugins" / "demo", "demo", min_lingcore="99.0.0")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    findings = diagnose_profile(profile(tmp_path)).errors
    assert any("requires LingCore" in f.message for f in findings)


def test_auto_engaged_bundled_components_do_not_enable_prompt_commands_hooks(
    tmp_path, monkeypatch
):
    from lingcore.plugins import discovery
    from lingcore.plugins.loader import load_plugins

    root = write_plugin(
        tmp_path / "bundled" / "demo",
        "demo",
        module="tools.py",
        hooks="Hooks",
        prompt="prompt.md",
    )
    (root / "tools.py").write_text(
        "from lingcore.plugins import PluginHooks\nclass Hooks(PluginHooks):\n    pass\n"
    )
    (root / "prompt.md").write_text("explicit context")
    commands = root / "commands"
    commands.mkdir()
    (commands / "check.md").write_text("---\ndescription: check\n---\ncheck it")
    skill = root / "skills" / "helper"
    skill.mkdir(parents=True)
    (skill / "skill.md").write_text("---\nname: helper\n---\nhelp")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", root.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    auto = load_plugins(profile(tmp_path, skills=["helper"]))
    assert (
        not auto.prompt_layers
        and not auto.hook_factories
        and not auto.commands.commands
    )
    explicit = load_plugins(profile(tmp_path, plugins=["demo"]))
    assert explicit.prompt_layers == ["explicit context"]
    assert len(explicit.hook_factories) == 1
    assert explicit.commands.resolve("/demo:check", reserved=set()).text == "check it"


def test_atomic_module_rolls_back_undeclared_plugin_tool(tmp_path):
    from lingcore.skills import _load_tool_module
    from lingcore.tools import REGISTRY

    source = tmp_path / "tools.py"
    source.write_text(
        'from pydantic import BaseModel\nfrom lingcore.tools import tool, ToolContext\nclass Args(BaseModel):\n    pass\n@tool(name="plugin_wrong", description="wrong")\nasync def wrong(args: Args, ctx: ToolContext) -> str:\n    return "bad"\n'
    )
    before = dict(REGISTRY._tools)
    with pytest.raises(ConfigError, match="undeclared"):
        _load_tool_module("demo_rollback", source, (), prefix="demo")
    assert REGISTRY._tools == before


def test_registered_module_tools_tracks_imported_function_object(
    tmp_path, monkeypatch, isolated_tool_catalog
):
    from lingcore.plugins.loader import load_plugins
    from lingcore.skills import registered_module_tools

    _write_external_function_plugin(tmp_path, monkeypatch)
    loaded = load_plugins(profile(tmp_path, plugins=["provenance-test"]))
    registered = isolated_tool_catalog.get("provenance_test_echo")
    assert registered.fn.__module__ == "provenance_external_package"
    assert "provenance_test_echo" in registered_module_tools()
    assert loaded.tool_names == frozenset({"provenance_test_echo"})


@pytest.mark.parametrize("remove_plugin", [False, True])
async def test_cached_plugin_tool_requires_current_engagement(
    tmp_path, monkeypatch, isolated_tool_catalog, remove_plugin
):
    from lingcore.agent import Agent
    from tests.fakes import FakeLLMClient

    root = _write_external_function_plugin(tmp_path, monkeypatch)
    enabled = profile(
        tmp_path, plugins=["provenance-test"], tools=["provenance_test_echo"]
    )
    first = Agent.from_profile(enabled, llm=FakeLLMClient([]))
    assert "provenance_test_echo" in first.tools.names()
    await first.aclose()
    if remove_plugin:
        root.rename(tmp_path / "retired-plugin")
    disabled = profile(tmp_path, tools=["provenance_test_echo"])
    with pytest.raises(ConfigError, match="engag"):
        Agent.from_profile(disabled, llm=FakeLLMClient([]))


def test_cached_enabled_module_keeps_provenance_and_tool_identity(
    tmp_path, monkeypatch, isolated_tool_catalog
):
    from lingcore.plugins.loader import load_plugins
    from lingcore.skills import registered_module_tools

    _write_external_function_plugin(tmp_path, monkeypatch)
    enabled = profile(tmp_path, plugins=["provenance-test"])
    first = load_plugins(enabled)
    original = isolated_tool_catalog.get("provenance_test_echo")
    second = load_plugins(enabled)
    assert second.tool_names == first.tool_names == frozenset({"provenance_test_echo"})
    assert isolated_tool_catalog.get("provenance_test_echo") is original
    assert "provenance_test_echo" in registered_module_tools()


def test_module_provenance_does_not_restrict_builtin_or_manual_tools(
    tmp_path, monkeypatch, isolated_tool_catalog
):
    from pydantic import BaseModel

    from lingcore.agent import Agent
    from lingcore.plugins.loader import load_plugins
    from lingcore.skills import registered_module_tools
    from lingcore.tools import Tool
    from tests.fakes import FakeLLMClient

    _write_external_function_plugin(tmp_path, monkeypatch)
    load_plugins(profile(tmp_path, plugins=["provenance-test"]))
    original = isolated_tool_catalog.get("provenance_test_echo")
    replacement = Tool("provenance_test_echo", "manual tool", BaseModel, original.fn)
    isolated_tool_catalog.register(replacement)
    assert "provenance_test_echo" not in registered_module_tools()
    assert "read_file" not in registered_module_tools()
    agent = Agent.from_profile(
        profile(tmp_path, tools=["read_file", "provenance_test_echo"]),
        llm=FakeLLMClient([]),
    )
    assert agent.tools.get("provenance_test_echo") is replacement
    assert agent.tools.get("read_file") is isolated_tool_catalog.get("read_file")


def test_failed_module_does_not_add_provenance(tmp_path, isolated_tool_catalog):
    from lingcore.skills import _load_tool_module, registered_module_tools

    source = tmp_path / "failed.py"
    source.write_text(
        "from pydantic import BaseModel\n"
        "from lingcore.tools import tool, ToolContext\n"
        "class Args(BaseModel):\n    pass\n"
        '@tool(name="failed_provenance_tool")\n'
        "async def registered(args: Args, ctx: ToolContext) -> str:\n"
        '    return "bad"\n'
        'raise RuntimeError("failure")\n'
    )
    before = registered_module_tools()
    with pytest.raises(ConfigError, match="failed to import"):
        _load_tool_module("failed-provenance", source, ("failed_provenance_tool",))
    assert "failed_provenance_tool" not in isolated_tool_catalog.names()
    assert registered_module_tools() == before


def test_local_skill_with_code_shadows_bundled_plugin_tools(tmp_path, monkeypatch):
    from lingcore.plugins import discovery
    from lingcore.plugins.loader import load_plugins

    root = write_plugin(
        tmp_path / "bundled" / "shadowdemo",
        "shadowdemo",
        module="tools.py",
        provides=["shadowdemo_run"],
    )
    (root / "tools.py").write_text("raise RuntimeError('bundled code executed')")
    bundled_skill = root / "skills" / "shadowdemo"
    bundled_skill.mkdir(parents=True)
    (bundled_skill / "skill.md").write_text(
        "---\nname: shadowdemo\nrequested_tools: [shadowdemo_run]\n---\nbundled"
    )
    local = tmp_path / "skills" / "shadowdemo"
    local.mkdir(parents=True)
    (local / "skill.md").write_text(
        "---\nname: shadowdemo\nrequested_tools: [shadowdemo_run]\n"
        "provides: [shadowdemo_run]\nmodule: tools.py\n---\nlocal"
    )
    (local / "tools.py").write_text(LOCAL_TOOL)
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", root.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    p = profile(tmp_path, skills=["shadowdemo"], tools=["shadowdemo_run"])
    assert "shadowdemo" not in discovery.engaged_plugins(
        p, discovery.discover_plugins(None)
    )
    loaded = load_plugins(p)
    assert loaded.skills["shadowdemo"].instructions == "local"
    assert "shadowdemo_run" in loaded.tool_names


def test_plugin_skill_cannot_replace_core_skill(tmp_path, monkeypatch):
    from lingcore.doctor import diagnose_profile
    from lingcore.plugins import discovery

    root = write_plugin(tmp_path / "bundled" / "demo", "demo")
    skill = root / "skills" / "code-review"
    skill.mkdir(parents=True)
    (skill / "skill.md").write_text("---\nname: code-review\n---\nimpostor")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", root.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    with pytest.raises(ConfigError, match="core skill"):
        load_plugins_for(tmp_path)
    report = diagnose_profile(profile(tmp_path))
    assert any("core skill" in f.message for f in report.errors)
