"""Engagement-gated tool imports and explicitly enabled plugin components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.plugins.commands import CommandCatalog, load_commands
from lingcore.plugins.discovery import (
    DiscoveredPlugin,
    discover_plugins,
    engaged_plugins,
    local_skills,
    plugin_skills,
    require_healthy,
)
from lingcore.plugins.hooks import HookFactory, PluginHooks
from lingcore.plugins.manifest import component_path
from lingcore.skills import Skill, _load_tool_module, load_skill_tools, load_skills


@dataclass(frozen=True)
class LoadedPlugins:
    skill_dirs: list[Path]
    skills: dict[str, Skill]
    tool_names: frozenset[str]
    prompt_layers: list[str]
    commands: CommandCatalog
    hook_factories: list[HookFactory]


def load_plugins(
    profile: AgentProfile, discovered: dict[str, DiscoveredPlugin] | None = None
) -> LoadedPlugins:
    source_dir: Path | None = getattr(profile, "_source_dir", None)
    problems: dict[str, str] = {}
    found = (
        discover_plugins(source_dir, problems=problems)
        if discovered is None
        else discovered
    )
    require_healthy(profile.plugins, problems)
    engagement = engaged_plugins(profile, found)
    # A broken local/installed plugin may have been meant to shadow an engaged
    # one; never fall back silently to the code it was replacing.
    require_healthy(engagement, problems)
    skill_dirs = [Path(__file__).resolve().parent.parent / "skills"]
    skills = load_skills(skill_dirs)
    core = set(skills)
    owners: dict[str, str] = {}
    for name, plugin in found.items():
        if plugin.source != "bundled" and name not in engagement:
            continue
        catalog = plugin_skills(plugin)
        for skill_name, skill in catalog.items():
            if skill_name in core:
                raise ConfigError(
                    f"plugin {name!r} skill {skill_name!r} collides with a core skill"
                )
            if skill_name in owners:
                raise ConfigError(
                    f"duplicate plugin skill {skill_name!r}: {owners[skill_name]!r} and {name!r}"
                )
            owners[skill_name] = name
            skills[skill_name] = skill
        if plugin.manifest.skills is not None:
            skill_dirs.append(component_path(plugin.root, plugin.manifest.skills))
    if source_dir is not None:
        skill_options = profile.tool_options.get("activate_skill", {})
        skill_dirs.append(source_dir / skill_options.get("skills_dir", "skills"))
        local = local_skills(profile)
        for name in local:
            owners.pop(name, None)
        skills.update(local)

    tool_names: set[str] = set()
    prompt_layers: list[str] = []
    commands = CommandCatalog()
    factories: list[HookFactory] = []
    for name in engagement:
        plugin = found[name]
        manifest = plugin.manifest
        manifest.check_compatibility()
        module = None
        if manifest.module is not None:
            module = _load_tool_module(
                name,
                component_path(plugin.root, manifest.module),
                manifest.provides,
                prefix=manifest.prefix,
            )
            tool_names.update(manifest.provides)
        for skill_name, owner in owners.items():
            if owner != name:
                continue
            skill = skills[skill_name]
            if skill.module is not None:
                assert skill.source_dir is not None
                _load_tool_module(
                    skill.name,
                    component_path(skill.source_dir, skill.module),
                    skill.provides,
                    prefix=manifest.prefix,
                )
                tool_names.update(skill.provides)
        if name not in profile.plugins:
            continue
        if manifest.prompt is not None:
            try:
                text = component_path(plugin.root, manifest.prompt).read_text("utf-8")
            except (OSError, UnicodeError) as exc:
                raise ConfigError(
                    f"cannot read plugin {name!r} prompt: {type(exc).__name__}"
                ) from None
            if len(text) > 16000:
                raise ConfigError(f"plugin {name!r} prompt exceeds 16000 characters")
            prompt_layers.append(text)
        if manifest.commands is not None:
            commands.extend(
                load_commands(
                    component_path(plugin.root, manifest.commands), namespace=name
                )
            )
        if manifest.hooks is not None:
            hooks = getattr(module, manifest.hooks, None)
            if not isinstance(hooks, type) or not issubclass(hooks, PluginHooks):
                raise ConfigError(
                    f"plugin {name!r} hooks must name a PluginHooks subclass"
                )
            assert manifest.options_key is not None
            factories.append(
                HookFactory(
                    name=name,
                    root=plugin.root,
                    hooks=hooks,
                    options_key=manifest.options_key,
                    on_hook_error=manifest.on_hook_error,
                    hook_timeout=manifest.hook_timeout,
                )
            )

    # Legacy/core and profile-local skills retain their original engagement rules.
    legacy = {
        name: skill
        for name, skill in skills.items()
        if name not in owners
        and skill.module is not None
        and (name in profile.skills or set(skill.provides) & set(profile.tools))
    }
    tool_names.update(load_skill_tools(legacy))
    if source_dir is not None:
        commands.extend(load_commands(source_dir / "commands"))
    return LoadedPlugins(
        skill_dirs, skills, frozenset(tool_names), prompt_layers, commands, factories
    )
