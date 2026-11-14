"""Model-facing entry into the inherited-authority delegation kernel."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lingcore._agent.dispatch import dispatch_authorization
from lingcore.delegation.schema import compile_schema
from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin._offload import offload_text


class DelegateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=20000)
    role: str | None = None
    continue_worker: str | None = None
    label: str | None = Field(default=None, max_length=200)
    output: str | dict[str, Any] | None = None
    workspace: Literal["shared", "frozen", "snapshot", "scratch"] | None = None
    tools: list[str] | None = None
    background: bool = False


@tool(high_risk=True)
async def delegate(args: DelegateArgs, ctx: ToolContext) -> str:
    """Delegate a bounded task to a role. Background results arrive automatically; never poll or wait."""
    supervisor = ctx.delegation
    if supervisor is None:
        raise ToolError("delegation is not configured")
    snapshot = dispatch_authorization.get()
    if snapshot is None:
        raise ToolError("delegation requires a dispatch authorization snapshot")
    if args.continue_worker:
        result = await supervisor.continue_worker(
            args.continue_worker, args.prompt, snapshot, background=args.background
        )
    else:
        schema = args.output
        if isinstance(schema, str):
            schema = supervisor.profile.delegation.schemas.get(schema)
            if schema is None:
                raise ToolError("unknown output schema")
        if schema is not None:
            compile_schema(schema)
        role = args.role or supervisor.profile.delegation.default_role
        envelope = supervisor.envelope(
            role, snapshot, tools=args.tools, workspace=args.workspace
        )
        handle = supervisor.new_run(background=args.background, name=args.label or role)
        await supervisor.confirm_run(handle, {role: envelope}, 1)
        result = await supervisor.execute(
            handle,
            lambda: supervisor.worker(
                handle, args.prompt, role, envelope, schema=schema, label=args.label
            ),
        )
    value = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    return offload_text(
        ctx,
        source="delegate",
        text=json.dumps(value, ensure_ascii=False),
        threshold=16000,
    )
