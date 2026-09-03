"""Tool shipped by the bundled Claude Code collaboration skill.

Only the Claude-specific parts live here: the session id is minted client-side
(``--session-id``) on the first turn and resumed with ``--resume`` afterwards,
and ``--print`` already yields plain text. Everything else — argument schema,
prompt brief, alias persistence, supervised execution — is shared in
``lingcore.outer_agents``.
"""

from __future__ import annotations

import uuid

from lingcore.outer_agents import (
    CLAUDE_CODE,
    OuterAgentArgs,
    confirm_outer_agent_write,
    conversation_lock,
    frame_prompt,
    load_conversation_session,
    render_outer_agent_reply,
    run_outer_agent,
    save_conversation_session,
    turn_header,
)
from lingcore.tools import ToolContext, tool


@tool(
    name=CLAUDE_CODE.tool,
    description=(
        "Start or continue a named conversation with an external Claude Code CLI "
        "agent. Use only when the user asks to involve Claude Code. Consultation "
        "is read-only; implementation requires confirmation."
    ),
    high_risk=True,
)
async def claude_code_agent(args: OuterAgentArgs, ctx: ToolContext) -> str:
    write = args.mode == "implement"
    if write:
        await confirm_outer_agent_write(ctx, CLAUDE_CODE)
    prompt = frame_prompt(CLAUDE_CODE, args)
    async with conversation_lock(ctx, CLAUDE_CODE, args.conversation):
        existing = load_conversation_session(ctx, CLAUDE_CODE, args.conversation)
        session_id = None if args.restart else existing
        # A fresh conversation gets a client-minted id so a failed first turn
        # never has to be reconciled with whatever Claude may have stored.
        active_session = session_id or str(uuid.uuid4())
        arguments = [
            "--print",
            "--output-format",
            "text",
            "--no-chrome",
            # Non-interactive runs otherwise inherit project/user hooks,
            # plugins, skills, MCP servers, and permission allow-rules. Keep
            # authentication and built-in file tools, but suppress those
            # customizations and confine file access to the workspace.
            "--safe-mode",
            "--restricted",
            "--strict-mcp-config",
            "--permission-mode",
            "acceptEdits" if write else "plan",
        ]
        if session_id is None:
            arguments.extend(["--session-id", active_session])
        else:
            arguments.extend(["--resume", session_id])
        output = await run_outer_agent(
            CLAUDE_CODE, arguments=arguments, prompt=prompt, ctx=ctx
        )
        save_conversation_session(ctx, CLAUDE_CODE, args.conversation, active_session)
        return render_outer_agent_reply(
            CLAUDE_CODE,
            ctx,
            output,
            header=turn_header(CLAUDE_CODE, args, existing=existing),
        )
