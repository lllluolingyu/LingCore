"""Plugin foundation tests: core behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from tests.support.plugin_foundation import (
    isolated_tool_catalog as isolated_tool_catalog,
)
from tests.support.plugin_foundation import load_plugins_for, profile, write_plugin


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
