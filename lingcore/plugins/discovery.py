"""Read plugin metadata without importing any plugin package code."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

import yaml

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.plugins.manifest import (
    PLUGIN_NAME,
    PluginManifest,
    component_path,
    read_manifest,
)
from lingcore.skills import Skill, load_skills

BUNDLED_PLUGINS_DIR = Path(__file__).resolve().parent.parent / "bundled_plugins"


@dataclass(frozen=True)
class DiscoveredPlugin:
    manifest: PluginManifest
    root: Path
    source: Literal["bundled", "installed", "local"]
    shadowed: tuple[DiscoveredPlugin, ...] = ()


def plugin_skills(plugin: DiscoveredPlugin) -> dict[str, Skill]:
    if plugin.manifest.skills is None:
        return {}
    directory = component_path(plugin.root, plugin.manifest.skills)
    if directory.is_dir():
        for subdirectory in sorted(directory.iterdir()):
            if not subdirectory.is_dir():
                continue
            relative = subdirectory.relative_to(plugin.root)
            component_path(plugin.root, relative.as_posix())
            component_path(plugin.root, (relative / "skill.md").as_posix())
    skills = load_skills([directory])
    for skill in skills.values():
        if (
            skill.source_dir is not None
            and not skill.source_dir.resolve().is_relative_to(plugin.root.resolve())
        ):
            raise ConfigError(
                f"plugin {plugin.manifest.name!r} skill path escapes plugin root"
            )
    return skills


def local_skills(profile: AgentProfile) -> dict[str, Skill]:
    """The profile's own ``activate_skill.skills_dir`` catalog (default ``skills``)."""
    source_dir: Path | None = getattr(profile, "_source_dir", None)
    if source_dir is None:
        return {}
    options = profile.tool_options.get("activate_skill", {})
    relative = (
        options.get("skills_dir", "skills")
        if isinstance(options, Mapping)
        else "skills"
    )
    return load_skills([source_dir / relative])


def require_healthy(names: Iterable[str], problems: Mapping[str, str]) -> None:
    """Raise the recorded discovery error for any plugin the profile relies on."""
    for name in names:
        if name in problems:
            raise ConfigError(problems[name])


def discover_plugins(
    profile_dir: Path | None, *, problems: dict[str, str] | None = None
) -> dict[str, DiscoveredPlugin]:
    """Read every visible manifest without importing plugin code.

    Bundled manifests ship with LingCore and must be valid. A broken installed
    entry point or profile-local plugin is skipped and recorded in ``problems``
    (keyed by entry-point name, or by a local plugin's directory name and
    declared manifest name), so one bad ``pip install`` cannot break profiles
    that never enable it, while a broken local override never silently falls
    back to the plugin it was meant to shadow. Callers raise the recorded error
    with :func:`require_healthy` for plugins a profile actually enables or
    engages.
    """
    found: dict[str, DiscoveredPlugin] = {}
    problems = {} if problems is None else problems

    def add(
        root: Path,
        source: Literal["bundled", "installed", "local"],
        expected: str | None = None,
    ) -> None:
        manifest = read_manifest(root)
        if expected is not None and manifest.name != expected:
            raise ConfigError(
                f"plugin entry point {expected!r} does not match manifest name {manifest.name!r}"
            )
        previous = found.get(manifest.name)
        shadowed = (previous, *previous.shadowed) if previous is not None else ()
        found[manifest.name] = DiscoveredPlugin(
            manifest, root.resolve(), source, shadowed
        )

    if BUNDLED_PLUGINS_DIR.is_dir():
        for root in sorted(BUNDLED_PLUGINS_DIR.iterdir()):
            if root.is_dir() and (root / "plugin.yaml").is_file():
                add(root, "bundled")
    entries = list(metadata.entry_points(group="lingcore.plugins"))
    counts: dict[str, int] = {}
    for ep in entries:
        counts[ep.name] = counts.get(ep.name, 0) + 1
    for ep in sorted(entries, key=lambda entry: entry.name):
        if counts[ep.name] > 1:
            problems[ep.name] = f"duplicate installed plugin entry point {ep.name!r}"
            continue
        try:
            add(_entry_point_root(ep.name, ep.value), "installed", ep.name)
        except ConfigError as exc:
            problems[ep.name] = str(exc)
    if profile_dir is not None:
        local = profile_dir / "plugins"
        if local.is_dir():
            for root in sorted(local.iterdir()):
                if root.is_dir() and (root / "plugin.yaml").is_file():
                    try:
                        if not root.resolve().is_relative_to(local.resolve()):
                            raise ConfigError(
                                "profile-local plugin path escapes plugins directory"
                            )
                        add(root, "local")
                    except ConfigError as exc:
                        problems[root.name] = str(exc)
                        declared = _declared_name(root)
                        if declared is not None:
                            problems[declared] = str(exc)
    return found


def _declared_name(root: Path) -> str | None:
    """The ``name`` an unusable local manifest declares, if it is a valid name.

    Only well-formed names are returned, so diagnostics never echo arbitrary
    manifest values.
    """
    try:
        raw = yaml.safe_load((root / "plugin.yaml").read_text("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    name = raw.get("name") if isinstance(raw, dict) else None
    if isinstance(name, str) and PLUGIN_NAME.fullmatch(name):
        return name
    return None


def _entry_point_root(name: str, value: str) -> Path:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ConfigError(
            f"plugin entry point {name!r} must name a single top-level package"
        )
    try:
        spec = importlib.util.find_spec(value)
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is None or spec.submodule_search_locations is None:
        raise ConfigError(f"plugin entry point {name!r} does not resolve to a package")
    roots = list(spec.submodule_search_locations)
    if len(roots) != 1:
        raise ConfigError(
            f"plugin entry point {name!r} must resolve to one package directory"
        )
    return Path(roots[0])


def engaged_plugins(
    profile: AgentProfile, discovered: dict[str, DiscoveredPlugin]
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for name in profile.plugins:
        if name not in discovered:
            raise ConfigError(f"unknown enabled plugin {name!r}")
        reasons[name] = "explicitly enabled"
    # A profile-local skill shadows a bundled one of the same name, including
    # its tools: neither may auto-engage the bundled code it replaces.
    local = local_skills(profile)
    local_provides = {tool for skill in local.values() for tool in skill.provides}
    for name, plugin in discovered.items():
        if name in reasons or plugin.source != "bundled":
            continue
        skills = {
            skill_name: skill
            for skill_name, skill in plugin_skills(plugin).items()
            if skill_name not in local
        }
        requested = set(profile.skills) & set(skills)
        provided = set(plugin.manifest.provides)
        for skill in skills.values():
            provided.update(skill.provides)
        provided -= local_provides
        if requested:
            reasons[name] = "bundled skill: " + ", ".join(sorted(requested))
        elif provided & set(profile.tools):
            reasons[name] = "authorized bundled tool: " + ", ".join(
                sorted(provided & set(profile.tools))
            )
    return reasons
