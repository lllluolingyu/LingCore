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
        raise ConfigError(
            f"skill {source!r} sets module but declares no provides"
        )
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


def load_skill_tools(skills: dict[str, Skill]) -> frozenset[str]:
    """Import the tool module of every skill that ships one, registering its
    ``@tool`` functions into the global ``REGISTRY``.

    Returns the set of tool names newly contributed by these skills.  This is
    code execution, so it fails loud: a broken module, a ``provides`` name the
    module never registered, an undeclared tool, or a name that collides with an
    existing tool all raise ``ConfigError`` (never a silent swallow — invariant
    5 / 13).

    Registration targets the process-global ``REGISTRY`` (the ``@tool``
    decorator's default and the same catalog builtins use); isolation between
    sessions is enforced one layer up by ``REGISTRY.subset(profile.tools)`` and
    ``SkillState`` — a registered-but-unauthorized tool is unreachable.

    Idempotent: a module already imported this process (tracked by its synthetic
    name in ``sys.modules``) is not re-executed, so repeated ``from_profile``
    calls in one process are safe.
    """
    from lingcore.tools import REGISTRY

    reg = REGISTRY
    newly: set[str] = set()
    for skill in skills.values():
        if skill.module is None:
            continue
        if skill.source_dir is None:
            raise ConfigError(
                f"skill {skill.name!r} declares a module but has no source_dir"
            )
        mod_path = (skill.source_dir / skill.module).resolve()
        if not mod_path.is_file():
            raise ConfigError(
                f"skill {skill.name!r} module not found: {mod_path}"
            )
        # Synthetic module name keyed by the *resolved path* (not just
        # name+stem) so a profile-local skill that shadows a bundled one — same
        # skill name, same filename, different file — gets its own sys.modules
        # entry and is actually executed, instead of silently aliasing the
        # first-loaded module. Same path twice still hashes equal → idempotent.
        path_tag = hashlib.sha1(str(mod_path).encode("utf-8")).hexdigest()[:8]
        mod_name = f"lingcore_skill_tools.{skill.name}.{mod_path.stem}_{path_tag}"
        if mod_name not in sys.modules:
            # Snapshot the global catalog so a module that fails its contract
            # (below) can be rolled back atomically — registration is a side
            # effect of import and must not leak a half-loaded skill.
            before_tools = dict(reg._tools)
            before = set(before_tools)
            # Refuse a declared name that already exists *before* executing —
            # shadowing a builtin or a previously-loaded skill is never allowed,
            # and checking up front means a colliding @tool never overwrites the
            # incumbent (register() is last-wins).
            collisions = sorted(set(skill.provides) & before)
            if collisions:
                raise ConfigError(
                    f"skill {skill.name!r} provides {collisions} which "
                    f"collide(s) with an already-registered tool"
                )
            spec = importlib.util.spec_from_file_location(mod_name, mod_path)
            if spec is None or spec.loader is None:
                raise ConfigError(f"cannot load skill module: {mod_path}")
            module = importlib.util.module_from_spec(spec)
            # Register in sys.modules *before* exec so the @tool decorator's
            # get_type_hints resolves the module's own arg models, and so
            # tracebacks/repr are sane.
            sys.modules[mod_name] = module
            try:
                spec.loader.exec_module(module)
                # The module must register exactly what it declared: no surprise
                # tools outside `provides` …
                undeclared = sorted((set(reg.names()) - before) - set(skill.provides))
                if undeclared:
                    raise ConfigError(
                        f"skill {skill.name!r} module registered undeclared tools "
                        f"{undeclared}; add them to provides or remove them"
                    )
                # … and no overwriting an incumbent. A name *in* provides that
                # already exists is caught pre-exec above; a name *not* in
                # provides that clobbers a builtin would slip past the
                # name-set diff, so compare object identity to catch it.
                clobbered = sorted(
                    n for n in before if reg._tools.get(n) is not before_tools[n]
                )
                if clobbered:
                    raise ConfigError(
                        f"skill {skill.name!r} module overwrote existing tool(s) "
                        f"{clobbered}; a skill may only register names it declares "
                        f"in provides"
                    )
            except Exception as e:
                # Restore the catalog and sys.modules to exactly their prior
                # state — a failed load leaves no trace, so a fixed retry runs
                # cleanly. (REGISTRY has no transaction API; restore in place to
                # preserve the singleton's identity.)
                reg._tools.clear()
                reg._tools.update(before_tools)
                sys.modules.pop(mod_name, None)
                if isinstance(e, ConfigError):
                    raise
                raise ConfigError(
                    f"failed to import skill module for {skill.name!r} "
                    f"({mod_path}): {e!r}"
                ) from e
        # Validate the declared contract regardless of import vs. cache hit.
        missing = [t for t in skill.provides if t not in reg.names()]
        if missing:
            raise ConfigError(
                f"skill {skill.name!r} declares provides={list(skill.provides)} "
                f"but did not register: {missing}"
            )
        newly |= set(skill.provides)
    return frozenset(newly)


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
