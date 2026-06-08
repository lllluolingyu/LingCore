"""Skill loading and runtime state.

A *skill* is a named bundle of (instruction body + requested tools + a
description).  Skills are model-invoked via the ``activate_skill`` tool.

Security model (see plan §4): a skill never *grants* tools.  The effective
tools when a skill is active are ``profile_tools ∩ skill.requested_tools`` —
the profile's ``tools:`` list is a hard ceiling a skill can never exceed.
``SkillState`` is shared (by reference) between the agent and the
``activate_skill`` tool; the tool mutates ``active`` and the agent reads it on
the next loop iteration.  The base ``ToolRegistry`` is never mutated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lingcore.errors import ConfigError

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


def _parse_skill(text: str, *, source: str) -> Skill:
    m = _FRONTMATTER.match(text)
    if not m:
        raise ConfigError(f"skill {source!r} is missing YAML frontmatter")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid frontmatter in skill {source!r}: {e}") from None
    if not isinstance(meta, dict) or "name" not in meta:
        raise ConfigError(f"skill {source!r} frontmatter must define a name")
    return Skill(
        name=str(meta["name"]),
        description=str(meta.get("description", "")),
        requested_tools=tuple(meta.get("requested_tools", [])),
        instructions=m.group(2).strip(),
    )


def load_skills(dirs: list[Path]) -> dict[str, Skill]:
    """Load every ``<dir>/<skill>/skill.md`` from the given directories.

    Later directories override earlier ones on name collision, so a
    profile-local skill can shadow a bundled one.
    """
    skills: dict[str, Skill] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            sf = sub / "skill.md"
            if sf.is_file():
                skill = _parse_skill(sf.read_text("utf-8"), source=str(sf))
                skills[skill.name] = skill
    return skills


@dataclass
class SkillState:
    """Mutable runtime state shared between the agent and ``activate_skill``."""

    skills: dict[str, Skill]
    profile_tools: frozenset[str]
    active: list[str] = field(default_factory=list)
    allow_concurrent: bool = False
    high_risk_tools: frozenset[str] = DEFAULT_HIGH_RISK_TOOLS

    def effective_tools(self, skill: Skill) -> frozenset[str]:
        """Tools a skill actually gets: profile ceiling ∩ requested."""
        return self.profile_tools & frozenset(skill.requested_tools)

    def active_effective_tools(self) -> frozenset[str]:
        """Union of effective tools across all active skills."""
        out: frozenset[str] = frozenset()
        for name in self.active:
            skill = self.skills.get(name)
            if skill:
                out |= self.effective_tools(skill)
        return out

    def instruction_map(self) -> dict[str, str]:
        return {name: s.instructions for name, s in self.skills.items()}
