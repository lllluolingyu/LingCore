"""Tool shipped by the bundled Claude Code collaboration skill."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from lingcore.outer_agents import (
    confirm_outer_agent_write,
    conversation_lock,
    load_conversation_session,
    run_outer_agent,
    save_conversation_session,
)
from lingcore.tools import ToolContext, tool


class ClaudeCodeAgentArgs(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=100_000,
        description=(
            "Task or follow-up question to send to the external Claude Code agent."
        ),
    )
    mode: Literal["consult", "implement"] = Field(
        default="consult",
        description=(
            "consult is read-only; implement may edit the workspace and requires "
            "fresh user confirmation"
        ),
    )
    conversation: str = Field(
        default="default",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description=(
            "Logical conversation name. Reusing it continues the prior Claude "
            "Code session in this LingCore chat."
        ),
    )
    restart: bool = Field(
        default=False,
        description="Start this named conversation afresh instead of resuming it.",
    )


@tool(
    name="claude_code_agent",
    description=(
        "Start or continue a named conversation with an external Claude Code CLI "
        "agent. Use only when the user asks to involve Claude Code. Consultation "
        "is read-only; implementation requires confirmation."
    ),
)
async def claude_code_agent(args: ClaudeCodeAgentArgs, ctx: ToolContext) -> str:
    write = args.mode == "implement"
    if write:
        await confirm_outer_agent_write(ctx, "Claude Code")
    action = (
        "Implement the requested change in the shared workspace. Stay within the "
        "stated scope, verify your work, and report changed files and checks run."
        if write
        else "Analyze the request without modifying files. Return concise findings with file references."
    )
    prompt = (
        "You are an external Claude Code collaborator called by a LingCore agent. "
        f"{action} Do not delegate to another agent.\n\n"
        f"Task from the orchestrating agent:\n{args.prompt.strip()}"
    )
    async with conversation_lock(ctx, "claude-code", args.conversation):
        existing = load_conversation_session(ctx, "claude-code", args.conversation)
        session_id = None if args.restart else existing
        active_session = session_id or str(uuid.uuid4())
        arguments = [
            "--print",
            "--output-format",
            "text",
            "--no-chrome",
            "--permission-mode",
            "acceptEdits" if write else "plan",
        ]
        if session_id is None:
            arguments.extend(["--session-id", active_session])
        else:
            arguments.extend(["--resume", session_id])
        if not write:
            arguments.append("--restricted")
        result = await run_outer_agent(
            program="claude",
            arguments=arguments,
            prompt=prompt,
            ctx=ctx,
            option_key="claude_code_agent",
            label="Claude Code",
            output_source="claude-code-agent",
        )
        save_conversation_session(ctx, "claude-code", args.conversation, active_session)
        state = "restarted" if args.restart else "resumed" if existing else "started"
        return result.replace(
            "Claude Code response:",
            f"Claude Code conversation {args.conversation!r} ({state}):",
            1,
        )
