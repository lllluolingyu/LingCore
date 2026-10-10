"""Tool shipped by the bundled Codex collaboration skill.

Only the Codex-specific parts live here: the durable thread id is minted by
Codex and read back from its ``--json`` event stream, and follow-ups go through
``codex exec resume <id>``. Everything else — argument schema, prompt brief,
alias persistence, supervised execution — is shared in
``lingcore.outer_agents``.
"""

from __future__ import annotations

import json
from dataclasses import replace

from lingcore.errors import ToolError
from lingcore.outer_agents import (
    CODEX,
    OuterAgentArgs,
    confirm_outer_agent_write,
    conversation_lock,
    frame_prompt,
    load_conversation_session,
    normalize_external_session_id,
    render_outer_agent_reply,
    run_outer_agent,
    save_conversation_session,
    turn_header,
)
from lingcore.tools import ToolContext, tool


def _text_content(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_text_content(item) for item in value]
        joined = "".join(part for part in parts if part)
        return joined or None
    if isinstance(value, dict):
        for key in ("text", "content", "message"):
            if key in value:
                found = _text_content(value[key])
                if found:
                    return found
    return None


def _codex_jsonl(raw: str) -> tuple[str | None, str]:
    """Extract the durable thread id and final assistant text from Codex JSONL."""
    session_id: str | None = None
    messages: list[str] = []
    diagnostics: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            diagnostics.append(line)
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", "")).lower().replace("_", ".")
        for container in (event, event.get("payload"), event.get("session")):
            if not isinstance(container, dict):
                continue
            for key in ("thread_id", "session_id"):
                candidate = container.get(key)
                if isinstance(candidate, str):
                    try:
                        session_id = normalize_external_session_id(candidate)
                    except ToolError:
                        pass
        if event_type in {"item.completed", "item.complete"}:
            item = event.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).lower().replace("_", ".")
                if item_type in {"agent.message", "assistant.message"}:
                    text = _text_content(item)
                    if text:
                        messages.append(text)
        if event_type in {"turn.completed", "task.complete"}:
            text = _text_content(event.get("last_agent_message"))
            if text:
                messages.append(text)
    rendered = messages[-1].strip() if messages else "\n".join(diagnostics).strip()
    if not rendered:
        rendered = "(Codex completed without an assistant message)"
    return session_id, rendered


@tool(
    name=CODEX.tool,
    description=(
        "Start or continue a named conversation with an external Codex CLI agent. "
        "Use only when the user asks to involve Codex. Consultation is read-only; "
        "implementation requires confirmation."
    ),
    high_risk=True,
)
async def codex_agent(args: OuterAgentArgs, ctx: ToolContext) -> str:
    write = args.mode == "implement"
    if write:
        await confirm_outer_agent_write(ctx, CODEX)
    prompt = frame_prompt(CODEX, args)
    async with conversation_lock(ctx, CODEX, args.conversation):
        existing = load_conversation_session(ctx, CODEX, args.conversation)
        session_id = None if args.restart else existing
        arguments = [
            "exec",
            # Headless exec normally denies escalation, but an inherited
            # ``approvals_reviewer=auto_review`` re-enables it and can defeat
            # even an explicit read-only sandbox. Pin the human reviewer; exec
            # has no interactive approver, so its deny-by-default policy holds.
            "-c",
            'approvals_reviewer="user"',
            "--json",
            "--color",
            "never",
            "--sandbox",
            "workspace-write" if write else "read-only",
            "--cd",
            str(ctx.workspace),
            "--skip-git-repo-check",
        ]
        if session_id is not None:
            arguments.extend(["resume", session_id])
        arguments.append("-")
        output = await run_outer_agent(
            CODEX, arguments=arguments, prompt=prompt, ctx=ctx
        )
        returned_session, message = _codex_jsonl(output.text)
        # Codex mints the thread id; a resumed turn may omit it, so fall back to
        # the id we resumed. Only a brand-new thread with no id is unusable.
        active_session = returned_session or session_id
        if active_session is None:
            raise ToolError("Codex completed but did not report a resumable thread id")
        save_conversation_session(ctx, CODEX, args.conversation, active_session)
        return render_outer_agent_reply(
            CODEX,
            ctx,
            replace(output, text=message),
            header=turn_header(CODEX, args, existing=existing),
        )
