"""Read world context or save an explicitly approved runtime snapshot."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool
from lingcore.world import WORLD_STATE_KEY, WorldFile


class WorldArgs(BaseModel):
    action: Literal["read", "update"]
    content: str | None = Field(
        default=None,
        description=(
            "For update: concise Markdown with observed system, resources, "
            "toolchains and installed packages, plus inspection limitations. "
            "Replaces the previous snapshot; omit secrets and section markers."
        ),
    )


@tool(
    description=(
        "Read world context or save a runtime environment snapshot after "
        "inspection with run_shell. Saving requires user confirmation and "
        "never modifies the authored world.md. "
        "The next model request sees the update. No workspace path is needed."
    ),
    high_risk=True,
)
async def world(args: WorldArgs, ctx: ToolContext) -> str:
    state = ctx.options.get(WORLD_STATE_KEY)
    if not isinstance(state, WorldFile):
        raise ToolError("world tool requires a profile directory and Agent assembly")
    if args.action == "read":
        return state.read() or "(world.md is empty)"
    if args.content is None:
        raise ToolError("world update requires content")
    content = state.validate_content(args.content)
    if ctx.confirm is None:
        raise ToolError("saving runtime world context requires a confirmation handler")
    revision = state.revision()
    approved = await ctx.confirm(
        f"Save this runtime environment snapshot to {state.path}?\n\n"
        "This text will become system-prompt context in later sessions. "
        "It describes observations, not permissions.\n\n" + content
    )
    if not approved:
        raise ToolError("runtime world snapshot update denied")
    await state.update(content, ctx.workspace, expected_revision=revision)
    return f"Updated {state.path}; environment context applies next request."
