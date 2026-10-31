"""Offline plugin discovery, declared requirements, and configuration diagnostics."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path

from lingcore._doctor_models import DoctorFinding
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.outer_agents import OUTER_AGENTS
from lingcore.plugins.discovery import DiscoveredPlugin


def _add_requirement(
    requirements: dict[str, set[str]], name: str | None, consumer: str
) -> None:
    if name is None or not name.strip():
        return
    requirements.setdefault(name.strip(), set()).add(consumer)


def _option_value(
    options: Mapping[str, object], key: str, default: object = None
) -> object:
    value: object = options
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _plugin_discovery(
    profile: AgentProfile,
) -> tuple[dict[str, DiscoveredPlugin], dict[str, str], dict[str, str]]:
    """Discover like the loader: broken plugins the profile relies on are errors."""
    from lingcore.plugins.discovery import (
        discover_plugins,
        engaged_plugins,
        require_healthy,
    )

    problems: dict[str, str] = {}
    discovered = discover_plugins(
        getattr(profile, "_source_dir", None), problems=problems
    )
    require_healthy(profile.plugins, problems)
    engagement = engaged_plugins(profile, discovered)
    require_healthy(engagement, problems)
    return discovered, engagement, problems


def _plugin_environment_requirements(
    profile: AgentProfile,
    requirements: dict[str, set[str]],
    example_names: set[str],
    errors: list[str],
) -> None:
    from lingcore.plugins.manifest import EnvironmentName

    try:
        discovered, engagement, _ = _plugin_discovery(profile)
    except ConfigError as exc:
        errors.append(str(exc))
        return
    for name in engagement:
        manifest = discovered[name].manifest
        raw_options = profile.tool_options.get(
            manifest.options_key or manifest.prefix, {}
        )
        if not isinstance(raw_options, Mapping):
            errors.append(f"tool_options.{manifest.options_key} must be a mapping")
            continue
        for entry in manifest.environment:
            if isinstance(entry, EnvironmentName):
                env_name: object = entry.name
                consumer = f"plugin {name}"
            else:
                env_name = _option_value(raw_options, entry.option, entry.default)
                consumer = f"tool_options.{manifest.options_key}.{entry.option}"
            if not isinstance(env_name, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", env_name
            ):
                errors.append(f"{consumer} must name an environment variable")
                continue
            example_names.add(env_name)
            if entry.required:
                _add_requirement(requirements, env_name, consumer)


def _builtin_tool_names() -> set[str]:
    """Read builtin declarations without importing their implementation."""
    names: set[str] = set()
    root = Path(__file__).resolve().parent / "tools" / "builtin"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text("utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if (
                    not isinstance(decorator, ast.Call)
                    or not isinstance(decorator.func, ast.Name)
                    or decorator.func.id != "tool"
                ):
                    continue
                name = node.name
                for keyword in decorator.keywords:
                    if (
                        keyword.arg == "name"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        name = keyword.value.value
                names.add(name)
    return names


def _plugin_findings(profile: AgentProfile) -> list[DoctorFinding]:
    from lingcore.plugins.discovery import plugin_skills
    from lingcore.plugins.manifest import component_path
    from lingcore.skills import load_skills

    findings: list[DoctorFinding] = []
    try:
        discovered, engagement, problems = _plugin_discovery(profile)
    except ConfigError:
        # The environment pass reports discovery errors once.
        return findings
    for name, problem in sorted(problems.items()):
        findings.append(
            DoctorFinding("warning", f"skipped unusable plugin {name!r}: {problem}")
        )
    known = _builtin_tool_names()
    core_skills = set(load_skills([Path(__file__).resolve().parent / "skills"]))
    skill_owners: dict[str, str] = {}
    source_dir: Path | None = getattr(profile, "_source_dir", None)
    for name, plugin in discovered.items():
        if plugin.shadowed:
            findings.append(
                DoctorFinding(
                    "warning",
                    f"plugin {name!r} from {plugin.source} shadows "
                    + ", ".join(previous.source for previous in plugin.shadowed),
                )
            )
        try:
            skills = plugin_skills(plugin)
            manifest = plugin.manifest
            known.update(manifest.provides)
            for skill in skills.values():
                known.update(skill.provides)
            manifest.check_compatibility()
            if name not in engagement and plugin.source != "bundled":
                continue
            for skill_name in skills:
                if skill_name in core_skills:
                    findings.append(
                        DoctorFinding(
                            "error",
                            f"plugin {name!r} skill {skill_name!r} collides with a core skill",
                        )
                    )
                if skill_name in skill_owners:
                    findings.append(
                        DoctorFinding("error", f"duplicate plugin skill {skill_name!r}")
                    )
                skill_owners[skill_name] = name
            if name not in engagement:
                continue
            options = profile.tool_options.get(
                manifest.options_key or manifest.prefix, {}
            )
            if not isinstance(options, Mapping):
                continue
            for requirement in manifest.requires.options:
                value = _option_value(options, requirement.key)
                if (
                    value is None
                    or value == ""
                    or (isinstance(value, str) and not value.strip())
                ):
                    findings.append(
                        DoctorFinding(
                            "error",
                            requirement.hint
                            or f"plugin {name!r} requires tool_options.{manifest.options_key}.{requirement.key}",
                        )
                    )
            # Outer-agent-specific option validation retains its established diagnostics.
            outer_keys = {spec.tool for spec in OUTER_AGENTS}
            for executable in manifest.requires.executables:
                if (
                    manifest.options_key in outer_keys
                    and manifest.options_key in profile.tools
                ):
                    continue
                configured = (
                    _option_value(options, executable.option, "")
                    if executable.option
                    else ""
                )
                if configured is not None and not isinstance(configured, str):
                    findings.append(
                        DoctorFinding(
                            "error",
                            f"plugin {name!r} executable option must be a string",
                        )
                    )
                    continue
                path = configured or executable.name
                expanded = Path(path).expanduser()
                if "/" in path:
                    candidate = (
                        expanded
                        if expanded.is_absolute()
                        else profile.workspace_path() / expanded
                    )
                    exists = candidate.is_file() and os.access(candidate, os.X_OK)
                else:
                    exists = shutil.which(path) is not None
                if not exists:
                    findings.append(
                        DoctorFinding(
                            "error" if configured else "warning",
                            f"plugin {name!r} executable {executable.name!r} is missing",
                        )
                    )
            for module in manifest.requires.modules:
                # find_spec on a top-level name locates it without importing.
                if importlib.util.find_spec(module.name) is None:
                    findings.append(
                        DoctorFinding(
                            "error",
                            module.hint
                            or f"plugin {name!r} requires Python module {module.name!r}",
                        )
                    )
            if manifest.hooks is not None and name not in profile.plugins:
                findings.append(
                    DoctorFinding(
                        "warning",
                        f"plugin {name!r} is engaged without explicit consent, so its "
                        f"hooks do not run; add {name!r} to plugins: if its tools "
                        "depend on them",
                    )
                )
            if name in profile.plugins:
                for field in ("module", "prompt"):
                    relative = getattr(manifest, field)
                    if (
                        relative is not None
                        and not component_path(plugin.root, relative).is_file()
                    ):
                        findings.append(
                            DoctorFinding(
                                "error", f"plugin {name!r} {field} file is missing"
                            )
                        )
                if manifest.prompt is not None:
                    prompt_path = component_path(plugin.root, manifest.prompt)
                    if (
                        prompt_path.is_file()
                        and len(prompt_path.read_text("utf-8")) > 16000
                    ):
                        findings.append(
                            DoctorFinding(
                                "error",
                                f"plugin {name!r} prompt exceeds 16000 characters",
                            )
                        )
            provided = set(manifest.provides)
            for skill in skills.values():
                provided.update(skill.provides)
            reachable_skill = any(
                skill.name in profile.skills
                or (
                    "activate_skill" in profile.tools
                    and (
                        not skill.requested_tools
                        or set(skill.requested_tools) & set(profile.tools)
                    )
                )
                for skill in skills.values()
            )
            explicit_components = name in profile.plugins and (
                manifest.hooks is not None
                or manifest.prompt is not None
                or (
                    manifest.commands is not None
                    and component_path(plugin.root, manifest.commands).is_dir()
                )
            )
            if (
                not provided.intersection(profile.tools)
                and not reachable_skill
                and not explicit_components
            ):
                findings.append(
                    DoctorFinding(
                        "warning",
                        f"plugin {name!r} is engaged but has no reachable tools, skills or enabled components",
                    )
                )
        except (ConfigError, OSError, UnicodeError) as exc:
            message = (
                str(exc)
                if isinstance(exc, ConfigError)
                else f"cannot read plugin {name!r} data: {type(exc).__name__}"
            )
            findings.append(DoctorFinding("error", message))
    try:
        dirs = [Path(__file__).resolve().parent / "skills"]
        if source_dir is not None:
            skill_options = profile.tool_options.get("activate_skill", {})
            relative = (
                skill_options.get("skills_dir", "skills")
                if isinstance(skill_options, Mapping)
                else "skills"
            )
            dirs.append(source_dir / relative)
        for skill in load_skills(dirs).values():
            if skill.name in profile.skills or set(skill.provides).intersection(
                profile.tools
            ):
                known.update(skill.provides)
    except ConfigError as exc:
        findings.append(DoctorFinding("error", str(exc)))
    for name in sorted(set(profile.tools) - known):
        findings.append(DoctorFinding("error", f"unknown authorized tool {name!r}"))
    return findings
