"""Tests for skill loading, the permission model, and activate_skill."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.errors import ConfigError, ToolError
from lingcore.skills import Skill, SkillState, load_skills
from lingcore.tools import ToolContext
from lingcore.tools.builtin.skill import SKILL_STATE_KEY, activate_skill


def _skill(name="review", tools=("read_file", "run_shell")) -> Skill:
    return Skill(
        name=name,
        description=f"{name} desc",
        requested_tools=tuple(tools),
        instructions=f"instructions for {name}",
    )


def _state(profile_tools, skills=None, **kw) -> SkillState:
    skills = skills or {"review": _skill()}
    return SkillState(
        skills=skills,
        profile_tools=frozenset(profile_tools),
        **kw,
    )


def _ctx(state, confirm=None) -> ToolContext:
    return ToolContext(
        workspace=Path("/tmp"),
        confirm=confirm,
        options={SKILL_STATE_KEY: state},
    )


# --------------------------------------------------------------------------- #
# Loading & frontmatter                                                       #
# --------------------------------------------------------------------------- #

def test_load_skills_from_dir(tmp_path):
    d = tmp_path / "skills" / "greet"
    d.mkdir(parents=True)
    (d / "skill.md").write_text(
        "---\nname: greet\ndescription: say hi\nrequested_tools:\n  - read_file\n---\nGreet warmly.",
        encoding="utf-8",
    )
    skills = load_skills([tmp_path / "skills"])
    assert "greet" in skills
    assert skills["greet"].requested_tools == ("read_file",)
    assert "Greet warmly" in skills["greet"].instructions


def test_load_skills_missing_frontmatter(tmp_path):
    d = tmp_path / "skills" / "bad"
    d.mkdir(parents=True)
    (d / "skill.md").write_text("no frontmatter here", encoding="utf-8")
    with pytest.raises(ConfigError, match="frontmatter"):
        load_skills([tmp_path / "skills"])


def test_load_bundled_code_review_skill():
    bundled = Path(__file__).parent.parent / "lingcore" / "skills"
    skills = load_skills([bundled])
    assert "code-review" in skills


def test_prompt_only_skill_has_no_shipped_code():
    # A prompt-only skill keeps the new code-shipping fields at their defaults,
    # so existing skills are unaffected by invariant 13 (additive change).
    bundled = Path(__file__).parent.parent / "lingcore" / "skills"
    skills = load_skills([bundled])
    assert skills["code-review"].provides == ()
    assert skills["code-review"].module is None


# --------------------------------------------------------------------------- #
# Permission model (§8 invariants)                                            #
# --------------------------------------------------------------------------- #

def test_effective_tools_is_intersection():
    state = _state(profile_tools={"read_file", "search"})
    skill = _skill(tools=("read_file", "run_shell"))
    # run_shell requested but not in profile → excluded.
    assert state.effective_tools(skill) == frozenset({"read_file"})


def test_skill_cannot_grant_disallowed_tool():
    state = _state(profile_tools={"read_file"})
    skill = _skill(tools=("run_shell",))
    assert "run_shell" not in state.effective_tools(skill)


# --------------------------------------------------------------------------- #
# activate_skill behaviour                                                     #
# --------------------------------------------------------------------------- #

async def test_activate_unknown_skill():
    state = _state(profile_tools={"read_file"})
    with pytest.raises(ToolError, match="unknown skill"):
        await activate_skill(activate_skill.args_model(name="nope"), _ctx(state))


async def test_activate_then_deactivate():
    state = _state(profile_tools={"read_file"})
    await activate_skill(activate_skill.args_model(name="review"), _ctx(state))
    assert state.active == ["review"]
    await activate_skill(
        activate_skill.args_model(name="review", active=False), _ctx(state)
    )
    assert state.active == []


async def test_non_concurrent_replaces_active():
    skills = {"a": _skill("a", ("read_file",)), "b": _skill("b", ("read_file",))}
    state = _state(profile_tools={"read_file"}, skills=skills, allow_concurrent=False)
    await activate_skill(activate_skill.args_model(name="a"), _ctx(state))
    await activate_skill(activate_skill.args_model(name="b"), _ctx(state))
    assert state.active == ["b"]


async def test_concurrent_allows_multiple():
    skills = {"a": _skill("a", ("read_file",)), "b": _skill("b", ("read_file",))}
    state = _state(profile_tools={"read_file"}, skills=skills, allow_concurrent=True)
    await activate_skill(activate_skill.args_model(name="a"), _ctx(state))
    await activate_skill(activate_skill.args_model(name="b"), _ctx(state))
    assert set(state.active) == {"a", "b"}


async def test_high_risk_skill_requires_confirmation():
    state = _state(profile_tools={"read_file", "run_shell"})
    calls = []

    async def confirm(prompt):
        calls.append(prompt)
        return False

    with pytest.raises(ToolError, match="declined"):
        await activate_skill(
            activate_skill.args_model(name="review"), _ctx(state, confirm=confirm)
        )
    assert calls and "run_shell" in calls[0]
    assert state.active == []


async def test_high_risk_skill_confirmed():
    state = _state(profile_tools={"read_file", "run_shell"})

    async def confirm(prompt):
        return True

    await activate_skill(
        activate_skill.args_model(name="review"), _ctx(state, confirm=confirm)
    )
    assert state.active == ["review"]


async def test_high_risk_skill_refused_without_confirm_handler():
    # No confirmation handler (confirm=None) + a high-risk skill must REFUSE
    # activation, not silently activate (mirrors run_shell's no-handler refusal).
    state = _state(profile_tools={"read_file", "run_shell"})
    with pytest.raises(ToolError, match="no confirmation handler"):
        await activate_skill(
            activate_skill.args_model(name="review"), _ctx(state, confirm=None)
        )
    assert state.active == []


async def test_low_risk_skill_no_confirmation():
    state = _state(profile_tools={"read_file"}, skills={"review": _skill(tools=("read_file",))})

    async def confirm(prompt):
        raise AssertionError("should not be called for low-risk skill")

    await activate_skill(
        activate_skill.args_model(name="review"), _ctx(state, confirm=confirm)
    )
    assert state.active == ["review"]
