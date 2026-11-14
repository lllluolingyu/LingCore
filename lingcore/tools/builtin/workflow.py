"""Small model surface for running, inspecting and recovering workflows."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lingcore._agent.dispatch import dispatch_authorization
from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin._offload import offload_text


class RunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    spec: str | dict[str, Any] | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    background: bool = False


class StatusArgs(BaseModel):
    run_id: str | None = None


class ResultArgs(BaseModel):
    run_id: str
    node: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=12000, ge=1, le=16000)


class IdArgs(BaseModel):
    run_id: str


def supervisor(ctx: ToolContext) -> Any:
    if ctx.delegation is None:
        raise ToolError("delegation is not configured")
    return ctx.delegation


def snapshot() -> frozenset[str]:
    value = dispatch_authorization.get()
    if value is None:
        raise ToolError("workflow requires dispatch authorization")
    return value


@tool(high_risk=True)
async def workflow_run(args: RunArgs, ctx: ToolContext) -> str:
    """Run a named or inline bounded workflow. Background results arrive automatically; do not poll."""
    from lingcore.workflows.runtime import run_workflow

    result = await run_workflow(supervisor(ctx), snapshot(), **args.model_dump())
    return offload_text(
        ctx,
        source="workflow",
        text=json.dumps(result, ensure_ascii=False),
        threshold=16000,
    )


@tool()
async def workflow_status(args: StatusArgs, ctx: ToolContext) -> str:
    """Read run status and spend without waiting. Omit run_id to list this session's runs."""
    s = supervisor(ctx)
    runs = (
        [s.owned_run(args.run_id, read_only=True)]
        if args.run_id
        else s.journal.list(s.parent._session_id)
    )
    return json.dumps(
        [
            {
                "run_id": r.id,
                "name": r.name,
                "status": r.status,
                "progress": r.progress,
                "usage": r.usage.model_dump(),
            }
            for r in runs[:100]
        ]
    )


@tool()
async def workflow_result(args: ResultArgs, ctx: ToolContext) -> str:
    """Read a completed run or node result in bounded pages."""
    s = supervisor(ctx)
    run = s.owned_run(args.run_id, read_only=True)
    value = run.model_dump(mode="json")
    if args.node:
        nodes = [
            n
            for n in s.journal.nodes(run.id)
            if n.identity == args.node or n.worker_id == args.node
        ]
        if not nodes:
            raise ToolError("unknown run node")
        value = nodes[0].model_dump(mode="json", by_alias=True)
    text = json.dumps(value, ensure_ascii=False)
    return json.dumps(
        {
            "text": text[args.offset : args.offset + args.limit],
            "next_offset": args.offset + args.limit
            if args.offset + args.limit < len(text)
            else None,
        }
    )


@tool(high_risk=True)
async def workflow_stop(args: IdArgs, ctx: ToolContext) -> str:
    """Stop a run, preserving its sessions and artifacts for inspection."""
    await supervisor(ctx).stop(args.run_id)
    return "Run stopped and detached. Its records are retained."


@tool(high_risk=True)
async def workflow_resume(args: IdArgs, ctx: ToolContext) -> str:
    """Confirm and resume a retained workflow. Recorded effects are never repeated."""
    from lingcore.workflows.runtime import resume_workflow

    result = await resume_workflow(supervisor(ctx), args.run_id, snapshot())
    return offload_text(
        ctx,
        source="workflow",
        text=json.dumps(result, ensure_ascii=False),
        threshold=16000,
    )
