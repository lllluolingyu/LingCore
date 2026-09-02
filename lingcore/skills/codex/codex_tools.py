"""Tool shipped by the bundled Codex collaboration skill."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.outer_agents import (
    confirm_outer_agent_write,
    conversation_lock,
    load_conversation_session,
    normalize_external_session_id,
    run_outer_agent,
    save_conversation_session,
)
from lingcore.tools import ToolContext, tool


class CodexAgentArgs(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=100_000,
        description="Task or follow-up question to send to the external Codex agent.",
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
            "Logical conversation name. Reusing it continues the prior Codex "
            "session in this LingCore chat."
        ),
    )
    restart: bool = Field(
        default=False,
        description="Start this named conversation afresh instead of resuming it.",
    )


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
    name="codex_agent",
    description=(
        "Start or continue a named conversation with an external Codex CLI agent. "
        "Use only when the user asks to involve Codex. Consultation is read-only; "
        "implementation requires confirmation."
    ),
)
async def codex_agent(args: CodexAgentArgs, ctx: ToolContext) -> str:
    write = args.mode == "implement"
    if write:
        await confirm_outer_agent_write(ctx, "Codex")
    action = (
        "Implement the requested change in the shared workspace. Stay within the "
        "stated scope, verify your work, and report changed files and checks run."
        if write
        else "Analyze the request without modifying files. Return concise findings with file references."
    )
    prompt = (
        "You are an external Codex collaborator called by a LingCore agent. "
        f"{action} Do not delegate to another agent.\n\n"
        f"Task from the orchestrating agent:\n{args.prompt.strip()}"
    )
    async with conversation_lock(ctx, "codex", args.conversation):
        existing = load_conversation_session(ctx, "codex", args.conversation)
        session_id = None if args.restart else existing
        arguments = [
            "exec",
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
        returned_session: str | None = None

        def transform(raw: str) -> str:
            nonlocal returned_session
            returned_session, text = _codex_jsonl(raw)
            if session_id is None and returned_session is None:
                raise ToolError(
                    "Codex completed but did not report a resumable thread id"
                )
            return text

        result = await run_outer_agent(
            program="codex",
            arguments=arguments,
            prompt=prompt,
            ctx=ctx,
            option_key="codex_agent",
            label="Codex",
            output_source="codex-agent",
            transform_output=transform,
        )
        active_session = returned_session or session_id
        assert active_session is not None
        save_conversation_session(ctx, "codex", args.conversation, active_session)
        state = "restarted" if args.restart else "resumed" if existing else "started"
        return result.replace(
            "Codex response:",
            f"Codex conversation {args.conversation!r} ({state}):",
            1,
        )
