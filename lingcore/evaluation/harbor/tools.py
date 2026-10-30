"""Trial-local tools; importing this module never changes the global registry."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lingcore.errors import ToolError
from lingcore.evaluation.harbor.binding import HarborTerminal, RefusingTerminal
from lingcore.tools import ToolContext, ToolRegistry, tool


class ExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    command: str = Field(min_length=1)
    cwd: str | None = None
    timeout_sec: float | None = Field(default=None, gt=0)

    @field_validator("cwd")
    @classmethod
    def _absolute_cwd(cls, value: str | None) -> str | None:
        if value is not None and not PurePosixPath(value).is_absolute():
            raise ValueError("cwd must be absolute in the task environment")
        return value


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    handle: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=16384, gt=0)


def _binding(ctx: ToolContext) -> HarborTerminal | RefusingTerminal:
    if not isinstance(ctx.environment_binding, (HarborTerminal, RefusingTerminal)):
        raise ToolError("terminal tools require a bound Harbor environment")
    return ctx.environment_binding


def terminal_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @tool(registry=registry, high_risk=True)
    async def terminal_exec(args: ExecArgs, ctx: ToolContext) -> str:
        """Run a command in the assigned task environment with combined stdout/stderr.

        Files persist between calls; working directory and shell variables do
        not. Use absolute cwd to select a directory. Long output is captured
        within a cap; use its handle with terminal_read_output for more.
        """
        value = await _binding(ctx).execute(args.command, args.cwd, args.timeout_sec)
        return json.dumps(value, ensure_ascii=False)

    @tool(registry=registry)
    async def terminal_read_output(args: ReadArgs, ctx: ToolContext) -> str:
        """Read a bounded byte slice of a captured terminal output handle."""
        return json.dumps(
            _binding(ctx).read_output(args.handle, args.offset, args.limit),
            ensure_ascii=False,
        )

    return registry
