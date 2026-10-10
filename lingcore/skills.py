"""Skill loading and runtime state.

A *skill* is a named bundle of (instruction body + requested tools + a
description), and optionally a Python module that *ships its own tool code*.
Skills are engaged either statically (a profile's ``skills:`` list) or
model-invoked via the ``activate_skill`` tool.

Security model: a skill never *grants* tools beyond the profile.  The effective
tools when a skill is active are ``profile_tools ∩ skill.requested_tools`` —
the profile's ``tools:`` list is a hard ceiling a skill can never exceed.  A
skill that ships code (``module:`` + ``provides:``) registers its tools into the
global ``REGISTRY`` at load, but *registration is not authorization*: such a
tool is only reachable if its name is also listed in the profile's ``tools:``
(see ``load_skill_tools`` and invariant 13).  ``SkillState`` is shared (by
reference) between the agent and the ``activate_skill`` tool; the tool mutates
``active`` and the agent reads it on the next loop iteration.  The base
``ToolRegistry`` subset is never mutated at runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import yaml

from lingcore.errors import ConfigError

if TYPE_CHECKING:
    from lingcore.tools import Tool

# Keep the registered objects alive: removing a plugin or its module cache must
# not erase the provenance of a tool still present in the global catalog. Tool
# objects are mutable/unhashable, so identity keys avoid equality or callable-
# module heuristics (plugins can register functions imported from another package).
_MODULE_TOOLS: dict[int, Tool] = {}


def registered_module_tools() -> frozenset[str]:
    """Names currently owned by successful skill/plugin module registrations.

    Callers must intersect these names with the current profile's explicitly
    loaded contribution set before authorizing a global-catalog subset. Manual
    replacement objects and builtin tools have independent provenance.
    """
    from lingcore.tools import REGISTRY

    return frozenset(
        name
        for name, registered in REGISTRY._tools.items()
        if _MODULE_TOOLS.get(id(registered)) is registered
    )


# Builtin tools whose grant always needs confirmation. This baseline is
# name-based because the builtins are core; a tool shipped by a skill (or any
# third party) declares its own risk with ``@tool(high_risk=True)`` instead, and
# ``Agent.from_profile`` folds those declarations into ``SkillState`` — the
# permission model never has to learn a skill's tool names.
DEFAULT_HIGH_RISK_TOOLS = frozenset(
    {"run_shell", "write_file", "patch_file", "edit_file"}
)

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.S)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    requested_tools: tuple[str, ...]
    instructions: str
    # Code-shipping skills declare a module + the tool names it registers.
    provides: tuple[str, ...] = ()
    module: str | None = None
    # The directory holding this skill's ``skill.md`` — used to locate ``module``.
    source_dir: Path | None = None


def _parse_skill(text: str, *, source: str, source_dir: Path | None = None) -> Skill:
    m = _FRONTMATTER.match(text)
    if not m:
        raise ConfigError(f"skill {source!r} is missing YAML frontmatter")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid frontmatter in skill {source!r}: {e}") from None
    if not isinstance(meta, dict) or "name" not in meta:
        raise ConfigError(f"skill {source!r} frontmatter must define a name")
    provides = tuple(meta.get("provides", []))
    module = str(meta["module"]) if meta.get("module") else None
    # provides and module are two halves of one feature: a skill cannot ship
    # tools without code, and code with nothing declared is untracked.
    if provides and module is None:
        raise ConfigError(
            f"skill {source!r} declares provides but no module to register them"
        )
    if module is not None and not provides:
        raise ConfigError(f"skill {source!r} sets module but declares no provides")
    return Skill(
        name=str(meta["name"]),
        description=str(meta.get("description", "")),
        requested_tools=tuple(meta.get("requested_tools", [])),
        instructions=m.group(2).strip(),
        provides=provides,
        module=module,
        source_dir=source_dir,
    )


def load_skills(dirs: list[Path]) -> dict[str, Skill]:
    """Load every ``<dir>/<skill>/skill.md`` from the given directories.

    Later directories override earlier ones on name collision, so a
    profile-local skill can shadow a bundled one (including its shipped code,
    since each ``Skill`` carries its own ``source_dir``).
    """
    skills: dict[str, Skill] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            sf = sub / "skill.md"
            if sf.is_file():
                skill = _parse_skill(
                    sf.read_text("utf-8"), source=str(sf), source_dir=sub
                )
                skills[skill.name] = skill
    return skills


def _load_tool_module(
    owner: str, mod_path: Path, provides: Iterable[str], *, prefix: str | None = None
) -> ModuleType:
    """Execute one path-identified module and atomically validate its catalog."""
    from lingcore.tools import REGISTRY

    declared = frozenset(provides)
    if prefix is not None:
        invalid = sorted(
            name
            for name in declared
            if not (
                re.fullmatch(r"[a-z0-9_]+", name)
                and (name == prefix or name.startswith(prefix + "_"))
            )
        )
        if invalid:
            raise ConfigError(
                f"{owner!r} tool names {invalid} violate plugin prefix {prefix!r}"
            )
    mod_path = mod_path.resolve()
    if not mod_path.is_file():
        raise ConfigError(f"skill {owner!r} module not found: {mod_path}")
    path_tag = hashlib.sha1(str(mod_path).encode("utf-8")).hexdigest()[:8]
    mod_name = f"lingcore_skill_tools.{owner}.{mod_path.stem}_{path_tag}"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        missing = sorted(declared - set(REGISTRY.names()))
        if missing:
            raise ConfigError(f"skill {owner!r} did not register: {missing}")
        registered = getattr(cached, "__lingcore_provides__", declared)
        if registered != declared:
            raise ConfigError(f"{owner!r} module provides changed after import")
        return cached
    before_tools = dict(REGISTRY._tools)
    before = set(before_tools)
    collisions = sorted(declared & before)
    if collisions:
        raise ConfigError(
            f"skill {owner!r} provides {collisions} which collide(s) with an already-registered tool"
        )
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot load skill module: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
        undeclared = sorted((set(REGISTRY.names()) - before) - declared)
        if undeclared:
            raise ConfigError(
                f"skill {owner!r} module registered undeclared tools {undeclared}; add them to provides or remove them"
            )
        clobbered = sorted(
            name
            for name in before
            if REGISTRY._tools.get(name) is not before_tools[name]
        )
        if clobbered:
            raise ConfigError(
                f"skill {owner!r} module overwrote existing tool(s) {clobbered}; a skill may only register names it declares in provides"
            )
        missing = sorted(declared - set(REGISTRY.names()))
        if missing:
            raise ConfigError(
                f"skill {owner!r} declares provides={sorted(declared)} but did not register: {missing}"
            )
        module.__lingcore_provides__ = declared  # type: ignore[attr-defined]
    except BaseException as exc:
        REGISTRY._tools.clear()
        REGISTRY._tools.update(before_tools)
        sys.modules.pop(mod_name, None)
        if isinstance(exc, ConfigError) or not isinstance(exc, Exception):
            raise
        raise ConfigError(
            f"failed to import skill module for {owner!r} ({mod_path}): {exc!r}"
        ) from exc
    # Publish provenance only after every registration contract check succeeds.
    # Rollback paths leave both the catalog and provenance unchanged.
    for name in declared:
        registered = REGISTRY._tools[name]
        _MODULE_TOOLS[id(registered)] = registered
    return module


def load_skill_tools(skills: dict[str, Skill]) -> frozenset[str]:
    """Import code-shipping skill modules, preserving atomic registration."""
    from lingcore.paths import PathEscapeError, resolve_confined

    contributed: set[str] = set()
    for skill in skills.values():
        if skill.module is None:
            continue
        if skill.source_dir is None:
            raise ConfigError(
                f"skill {skill.name!r} declares a module but has no source_dir"
            )
        try:
            if Path(skill.module).is_absolute():
                raise PathEscapeError("absolute module path")
            mod_path = resolve_confined(skill.source_dir, skill.module)
        except PathEscapeError:
            raise ConfigError(
                f"skill {skill.name!r} module path escapes skill directory"
            ) from None
        _load_tool_module(skill.name, mod_path, skill.provides)
        contributed.update(skill.provides)
    return frozenset(contributed)


@dataclass
class SkillState:
    """Mutable runtime state shared between the agent and ``activate_skill``."""

    skills: dict[str, Skill]
    profile_tools: frozenset[str]
    active: list[str] = field(default_factory=list)
    allow_concurrent: bool = False
    high_risk_tools: frozenset[str] = DEFAULT_HIGH_RISK_TOOLS
    # High-risk grants covered by the confirmation that activated each skill.
    # This travels with durable session state so a changed profile/skill cannot
    # silently widen an old activation into newly dangerous capabilities.
    approved_high_risk: dict[str, frozenset[str]] = field(default_factory=dict)

    def effective_tools(self, skill: Skill) -> frozenset[str]:
        """Tools a skill actually gets: profile ceiling ∩ requested."""
        return self.profile_tools & frozenset(skill.requested_tools)

    def active_effective_tools(self) -> frozenset[str]:
        """Union active grants, never exceeding recorded risky consent."""
        out: frozenset[str] = frozenset()
        for name in self.active:
            skill = self.skills.get(name)
            if skill:
                effective = self.effective_tools(skill)
                risky = effective & self.high_risk_tools
                approved = self.approved_high_risk.get(name, frozenset())
                out |= (effective - risky) | (risky & approved)
        return out

    def instruction_map(self) -> dict[str, str]:
        return {name: s.instructions for name, s in self.skills.items()}
