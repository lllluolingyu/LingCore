"""activate_skill — model-invoked skill activation.

The model calls this to turn a named skill on or off.  Activation:
  1. validates the skill exists;
  2. computes effective tools = profile ∩ skill.requested (never exceeds profile);
  3. if any effective tool is high-risk, requires confirmation via ctx.confirm;
  4. mutates the shared ``SkillState.active`` list — the agent reads it on the
     *next* loop iteration, so instructions and tools take effect then.

The base ToolRegistry is never mutated; activation only changes which schemas
the agent computes per iteration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool

# Reserved options key the agent uses to share live SkillState with this tool.
SKILL_STATE_KEY = "_skill_state"


class SkillArgs(BaseModel):
    name: str = Field(description="Name of the skill to activate or deactivate.")
    active: bool = Field(default=True, description="True to activate, False to deactivate.")


@tool(description=(
    "Activate or deactivate a named skill. An active skill's instructions are "
    "added to the system prompt and its requested tools (limited to those the "
    "profile already allows) become available on the next step."
))
async def activate_skill(args: SkillArgs, ctx: ToolContext) -> str:
    state = ctx.options.get(SKILL_STATE_KEY)
    if state is None:
        raise ToolError("skills are not configured for this agent")

    skill = state.skills.get(args.name)
    if skill is None:
        available = ", ".join(sorted(state.skills)) or "(none)"
        raise ToolError(f"unknown skill {args.name!r}; available: {available}")

    if not args.active:
        if args.name in state.active:
            state.active.remove(args.name)
            state.approved_high_risk.pop(args.name, None)
            return f"deactivated skill {args.name!r}"
        return f"skill {args.name!r} was not active"

    if args.name in state.active:
        return f"skill {args.name!r} already active"

    effective = state.effective_tools(skill)
    risky = effective & state.high_risk_tools
    if risky:
        # A skill granting high-risk tools requires human consent. If no
        # confirmation handler is wired to this frontend, refuse rather than
        # activate silently (mirrors run_shell's no-handler refusal).
        if ctx.confirm is None:
            raise ToolError(
                f"skill {args.name!r} requests high-risk tools "
                f"({', '.join(sorted(risky))}) but no confirmation handler is "
                "available on this frontend; activation refused"
            )
        approved = await ctx.confirm(
            f"Activate skill {args.name!r}? It requests high-risk tools: "
            f"{', '.join(sorted(risky))}"
        )
        if not approved:
            raise ToolError(f"user declined activation of skill {args.name!r}")

    if not state.allow_concurrent:
        state.active.clear()
        state.approved_high_risk.clear()
    state.active.append(args.name)
    state.approved_high_risk[args.name] = frozenset(risky)

    granted = ", ".join(sorted(effective)) or "(none)"
    return f"activated skill {args.name!r}; tools available: {granted}"
