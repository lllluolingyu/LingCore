"""Data-only discovery and atomic plugin loading contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError


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


def test_manifest_strict_names_paths_versions_and_prefix():
    from lingcore.plugins.manifest import PluginManifest

    m = PluginManifest(
        name="my-plugin", version="1.10.0", api=1, provides=["my_plugin_run"]
    )
    assert m.prefix == "my_plugin"
    assert m.options_key == "my_plugin"
    for fields in (
        {"name": "Bad"},
        {"api": 2},
        {"provides": ["other"]},
        {"module": "../tools.py"},
        {"version": "banana"},
        {"environment": [{"name": "KEY", "option": "token_env"}]},
        {"hook_timeout": 61},
        {"extra": True},
    ):
        with pytest.raises(ValidationError):
            PluginManifest.model_validate(
                {"name": "my-plugin", "version": "1.0.0", "api": 1, **fields}
            )


def test_profile_plugin_names_unique():
    with pytest.raises(ValidationError):
        AgentProfile(llm={"model": "t"}, plugins=["same", "same"])
    with pytest.raises(ValidationError):
        AgentProfile(llm={"model": "t"}, plugins=["bad_name"])


def test_discovery_local_shadows_without_import(tmp_path, monkeypatch):
    from lingcore.plugins import discovery

    bundled = write_plugin(tmp_path / "bundled" / "demo", "demo")
    local = write_plugin(
        tmp_path / "profile" / "plugins" / "demo", "demo", module="tools.py"
    )
    (local / "tools.py").write_text("raise RuntimeError('must not import')")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", bundled.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    found = discovery.discover_plugins(tmp_path / "profile")
    assert found["demo"].root == local.resolve()
    assert found["demo"].source == "local"
    assert found["demo"].shadowed


def test_installed_discovery_rejects_dotted_entrypoint_before_find_spec(
    tmp_path, monkeypatch
):
    from lingcore.plugins import discovery

    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    ep = SimpleNamespace(
        name="demo", value="evil.package", dist=SimpleNamespace(name="dist")
    )
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [ep])
    monkeypatch.setattr(
        discovery.importlib.util,
        "find_spec",
        lambda _: pytest.fail("dotted entrypoint imported"),
    )
    problems: dict[str, str] = {}
    assert discovery.discover_plugins(None, problems=problems) == {}
    assert "top-level" in problems["demo"]
    with pytest.raises(ConfigError, match="top-level"):
        load_plugins_for(tmp_path, plugins=["demo"])


def test_duplicate_entry_points_rejected(tmp_path, monkeypatch):
    from lingcore.plugins import discovery

    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    eps = [
        SimpleNamespace(name="demo", value="safe", dist=SimpleNamespace(name=n))
        for n in ("a", "b")
    ]
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: eps)
    problems: dict[str, str] = {}
    assert discovery.discover_plugins(None, problems=problems) == {}
    assert "duplicate" in problems["demo"]
    with pytest.raises(ConfigError, match="duplicate"):
        load_plugins_for(tmp_path, plugins=["demo"])


def test_broken_installed_plugin_does_not_break_other_profiles(tmp_path, monkeypatch):
    from lingcore.doctor import diagnose_profile
    from lingcore.plugins import discovery

    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    ep = SimpleNamespace(name="broken", value="no_such_lingcore_pkg_xyz")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [ep])
    loaded = load_plugins_for(tmp_path)
    assert loaded.hook_factories == []
    report = diagnose_profile(profile(tmp_path))
    assert not report.errors
    assert any("skipped unusable plugin 'broken'" in f.message for f in report.warnings)


def test_broken_local_override_never_falls_back_to_the_shadowed_plugin(
    tmp_path, monkeypatch
):
    from lingcore.plugins import discovery

    bundled = write_plugin(tmp_path / "bundled" / "demo", "demo")
    # The directory name need not match the manifest name it overrides.
    write_plugin(tmp_path / "plugins" / "demo-checkout", "demo", api=2)
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", bundled.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    problems: dict[str, str] = {}
    found = discovery.discover_plugins(tmp_path, problems=problems)
    assert found["demo"].source == "bundled"
    assert "api" in problems["demo"] and "api" in problems["demo-checkout"]
    with pytest.raises(ConfigError, match="invalid plugin manifest"):
        load_plugins_for(tmp_path, plugins=["demo"])
    # An unusable declared name is never used as a key.
    write_plugin(tmp_path / "plugins" / "odd", "Not A Name")
    problems = {}
    discovery.discover_plugins(tmp_path, problems=problems)
    assert set(problems) == {"demo", "demo-checkout", "odd"}


def test_bundled_engagement_only_named_skill_or_authorized_provided_tool(
    tmp_path, monkeypatch
):
    from lingcore.plugins import discovery

    root = write_plugin(tmp_path / "bundled" / "demo", "demo", provides=["demo_run"])
    skill = root / "skills" / "helper"
    skill.mkdir(parents=True)
    (skill / "skill.md").write_text(
        "---\nname: helper\nrequested_tools: [demo_run]\n---\nhelp"
    )
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", root.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    found = discovery.discover_plugins(None)
    assert not discovery.engaged_plugins(
        profile(tmp_path, tools=["activate_skill"]), found
    )
    assert "demo" in discovery.engaged_plugins(
        profile(tmp_path, skills=["helper"]), found
    )
    assert "demo" in discovery.engaged_plugins(
        profile(tmp_path, tools=["demo_run"]), found
    )


def test_unengaged_plugin_never_executes_and_bundled_catalog_visible(
    tmp_path, monkeypatch
):
    from lingcore.plugins import discovery
    from lingcore.plugins.loader import load_plugins

    bundled = write_plugin(tmp_path / "bundled" / "demo", "demo", module="tools.py")
    (bundled / "tools.py").write_text("raise RuntimeError('unengaged bundled')")
    sk = bundled / "skills" / "helper"
    sk.mkdir(parents=True)
    (sk / "skill.md").write_text("---\nname: helper\n---\nhelp")
    local = write_plugin(tmp_path / "plugins" / "third", "third", module="tools.py")
    (local / "tools.py").write_text("raise RuntimeError('unengaged local')")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", bundled.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    loaded = load_plugins(profile(tmp_path))
    assert "helper" in loaded.skills
    assert not loaded.tool_names
    assert not loaded.prompt_layers


def test_plugin_prefix_rejected_before_execution_and_registry_unchanged(tmp_path):
    from lingcore.skills import _load_tool_module
    from lingcore.tools import REGISTRY

    module = tmp_path / "tools.py"
    module.write_text("raise RuntimeError('executed')")
    before = dict(REGISTRY._tools)
    with pytest.raises(ConfigError, match="prefix"):
        _load_tool_module("demo", module, ("unprefixed",), prefix="demo")
    assert REGISTRY._tools == before


def test_prompt_limit_checked_only_on_explicit_enablement(tmp_path, monkeypatch):
    from lingcore.plugins import discovery
    from lingcore.plugins.loader import load_plugins

    root = write_plugin(tmp_path / "plugins" / "demo", "demo", prompt="prompt.md")
    (root / "prompt.md").write_text("a" * 16001)
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    assert not load_plugins(profile(tmp_path)).prompt_layers
    with pytest.raises(ConfigError, match="16000"):
        load_plugins(profile(tmp_path, plugins=["demo"]))


def test_installed_metadata_discovery_does_not_execute_package(tmp_path, monkeypatch):
    from lingcore.plugins import discovery

    package = write_plugin(tmp_path / "sentinel", "demo")
    (package / "__init__.py").write_text("raise RuntimeError('package imported')")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    ep = SimpleNamespace(
        name="demo", value="sentinel", dist=SimpleNamespace(name="dist")
    )
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [ep])
    assert discovery.discover_plugins(None)["demo"].root == package.resolve()


def test_doctor_reads_plugin_requirements_without_import_and_reports_unknown_tools(
    tmp_path, monkeypatch
):
    from lingcore.doctor import diagnose_profile, required_example_names
    from lingcore.plugins import discovery

    root = write_plugin(
        tmp_path / "plugins" / "demo",
        "demo",
        module="tools.py",
        provides=["demo_run"],
        environment=[
            {"option": "token_env", "default": "DEMO_TOKEN", "required": True}
        ],
        requires={
            "options": [{"key": "endpoint"}],
            "executables": [{"name": "nonexistent-lingcore-plugin-executable"}],
        },
    )
    (root / "tools.py").write_text("raise RuntimeError('doctor imported')")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    p = profile(
        tmp_path,
        plugins=["demo"],
        tools=["demo_run", "unknown_authorized"],
        tool_options={"demo": {"token_env": "CUSTOM_TOKEN"}},
    )
    messages = [finding.message for finding in diagnose_profile(p).findings]
    assert any("CUSTOM_TOKEN is missing" in m for m in messages)
    assert any("endpoint" in m for m in messages)
    assert any("nonexistent-lingcore-plugin-executable" in m for m in messages)
    assert any("unknown_authorized" in m for m in messages)
    assert required_example_names(p) == frozenset({"CUSTOM_TOKEN"})


def test_doctor_bad_manifest_is_finding(tmp_path, monkeypatch):
    from lingcore.doctor import diagnose_profile
    from lingcore.plugins import discovery

    write_plugin(tmp_path / "plugins" / "bad", "bad", api=2)
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    report = diagnose_profile(profile(tmp_path, plugins=["bad"]))
    assert any("invalid plugin manifest" in f.message for f in report.errors)


def test_local_skill_shadow_does_not_import_plugin_skill_module(tmp_path, monkeypatch):
    from lingcore.plugins import discovery
    from lingcore.plugins.loader import load_plugins

    root = write_plugin(tmp_path / "bundled" / "demo", "demo")
    skill = root / "skills" / "helper"
    skill.mkdir(parents=True)
    (skill / "skill.md").write_text(
        "---\nname: helper\nmodule: tools.py\nprovides: [demo_run]\n---\nplugin"
    )
    (skill / "tools.py").write_text(
        "raise RuntimeError('shadowed plugin skill executed')"
    )
    local = tmp_path / "skills" / "helper"
    local.mkdir(parents=True)
    (local / "skill.md").write_text("---\nname: helper\n---\nlocal")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", root.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    loaded = load_plugins(profile(tmp_path, skills=["helper"]))
    assert loaded.skills["helper"].instructions == "local"


def test_plugin_skill_prefix_is_validated_atomically(tmp_path, monkeypatch):
    from lingcore.plugins import discovery
    from lingcore.plugins.loader import load_plugins

    root = write_plugin(tmp_path / "plugins" / "demo", "demo")
    skill = root / "skills" / "helper"
    skill.mkdir(parents=True)
    (skill / "skill.md").write_text(
        "---\nname: helper\nmodule: tools.py\nprovides: [unprefixed]\n---\nplugin"
    )
    (skill / "tools.py").write_text("raise RuntimeError('prefix validation too late')")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    with pytest.raises(ConfigError, match="prefix"):
        load_plugins(profile(tmp_path, plugins=["demo"]))


def test_discovery_rejects_component_symlink_escape_without_import(
    tmp_path, monkeypatch
):
    from lingcore.plugins import discovery

    root = write_plugin(tmp_path / "plugins" / "demo", "demo", module="tools.py")
    outside = tmp_path / "outside.py"
    outside.write_text("raise RuntimeError('must never execute')")
    (root / "tools.py").symlink_to(outside)
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    problems: dict[str, str] = {}
    assert "demo" not in discovery.discover_plugins(tmp_path, problems=problems)
    assert "unsafe plugin component" in problems["demo"]
    with pytest.raises(ConfigError, match="unsafe plugin component"):
        load_plugins_for(tmp_path, plugins=["demo"])


def test_plugin_skill_file_symlink_escape_rejected(tmp_path, monkeypatch):
    from lingcore.plugins import discovery

    root = write_plugin(tmp_path / "bundled" / "demo", "demo")
    skill = root / "skills" / "helper"
    skill.mkdir(parents=True)
    outside = tmp_path / "secret.md"
    outside.write_text("---\nname: helper\n---\nprivate")
    (skill / "skill.md").symlink_to(outside)
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", root.parent)
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    with pytest.raises(ConfigError, match="unsafe plugin component"):
        discovery.engaged_plugins(
            profile(tmp_path, skills=["helper"]), discovery.discover_plugins(None)
        )


def test_local_skill_catalog_custom_directory_preserves_code_engagement(
    tmp_path, monkeypatch
):
    from lingcore.plugins import discovery
    from lingcore.plugins.loader import load_plugins

    local = tmp_path / "custom_skills" / "hidden"
    local.mkdir(parents=True)
    (local / "skill.md").write_text(
        "---\nname: hidden\nmodule: tools.py\nprovides: [hidden_run]\nrequested_tools: [hidden_run]\n---\nhelp"
    )
    (local / "tools.py").write_text("raise RuntimeError('unauthorized skill executed')")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "none")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    loaded = load_plugins(
        profile(
            tmp_path,
            tools=["activate_skill"],
            tool_options={"activate_skill": {"skills_dir": "custom_skills"}},
        )
    )
    assert "hidden" in loaded.skills
    assert not loaded.tool_names


def test_duplicate_plugin_skill_names_are_rejected_before_modules(
    tmp_path, monkeypatch
):
    from lingcore.plugins import discovery
    from lingcore.plugins.loader import load_plugins

    for name in ("first", "second"):
        root = write_plugin(tmp_path / "bundled" / name, name, module="tools.py")
        (root / "tools.py").write_text(
            "raise RuntimeError('duplicate must fail first')"
        )
        skill = root / "skills" / "helper"
        skill.mkdir(parents=True)
        (skill / "skill.md").write_text("---\nname: helper\n---\nhelp")
    monkeypatch.setattr(discovery, "BUNDLED_PLUGINS_DIR", tmp_path / "bundled")
    monkeypatch.setattr(discovery.metadata, "entry_points", lambda **kw: [])
    with pytest.raises(ConfigError, match="duplicate plugin skill"):
        load_plugins(profile(tmp_path, skills=["helper"]))


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
