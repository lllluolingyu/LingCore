"""Validated background-process options and tool argument schemas."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lingcore.errors import ToolError
from lingcore.sandbox import (
    SandboxConfig,
    parse_shell_options,
)

OPTION_PATH = "tool_options.process"
_DEFAULT_TAIL = 200


class ProcessOptions(BaseModel):
    """Validated ``tool_options.process``."""

    model_config = ConfigDict(extra="forbid")

    max_processes: int = Field(default=3, ge=1, le=16)
    max_exited: int = Field(default=5, ge=0, le=50)
    max_log_bytes: int = Field(default=1024 * 1024, ge=64 * 1024, le=64 * 1024 * 1024)
    max_line_chars: int = Field(default=2000, ge=200, le=20_000)
    ready_timeout: float = Field(default=60.0, gt=0)
    max_ready_timeout: float = Field(default=300.0, gt=0)
    max_lifetime: float = Field(default=3600.0, gt=0, le=7 * 24 * 3600)
    require_confirmation: bool = True
    allow_patterns: list[str] = Field(default_factory=list)
    offload_over_chars: int = Field(default=8_000, ge=0)
    max_output_chars: int = Field(default=16_000, ge=1)
    # Unset: the run_shell sandbox. Set: processes run under this one instead.
    sandbox: SandboxConfig | None = None

    @model_validator(mode="after")
    def _ready_ceiling(self) -> ProcessOptions:
        if self.max_ready_timeout < self.ready_timeout:
            raise ValueError("max_ready_timeout must be at least ready_timeout")
        return self


def parse_process_options(
    tool_options: Mapping[str, object],
) -> tuple[ProcessOptions, SandboxConfig]:
    """The plugin options plus the sandbox processes run in; ToolError if none."""
    raw = tool_options.get("process", {})
    if not isinstance(raw, Mapping):
        raise ToolError(f"{OPTION_PATH} must be a mapping")
    try:
        options = ProcessOptions.model_validate(raw)
    except ValidationError as exc:
        raise ToolError(f"invalid {OPTION_PATH}: {_errors(exc)}") from None
    sandbox = options.sandbox
    if sandbox is None:
        try:
            sandbox = parse_shell_options(tool_options.get("run_shell", {})).sandbox
        except (ValidationError, ValueError) as exc:
            raise ToolError(f"invalid tool_options.run_shell: {exc}") from None
    if sandbox is None:
        raise ToolError(
            "background processes require a sandbox: configure "
            f"tool_options.run_shell.sandbox or {OPTION_PATH}.sandbox (Bubblewrap "
            "or OCI). The host runner cannot guarantee a process dies with LingCore."
        )
    return options, sandbox


def _errors(exc: ValidationError) -> str:
    return "; ".join(
        ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
        for error in exc.errors()
    )


class StartArgs(BaseModel):
    command: str = Field(
        min_length=1,
        description="The long-running shell command, e.g. 'npm run dev -- --port 5173 --strictPort'.",
    )
    name: str | None = Field(
        default=None, max_length=40, description="A short label, e.g. 'dev'."
    )
    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description=(
            "The port the command's server listens on inside the sandbox. The "
            "browser tools then reach it at http://localhost:<port>/. Pin the "
            "server to this port (e.g. --strictPort) so it cannot move."
        ),
    )
    ready_pattern: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Regex on log lines that means the process is up, e.g. "
            "'Local:\\s+http' or 'ready in'; the call waits for it."
        ),
    )
    ready_timeout: float | None = Field(
        default=None,
        gt=0,
        description="Seconds to wait for ready_pattern; clamped to the profile's ceiling.",
    )


class IdArgs(BaseModel):
    id: str = Field(description="The process id from process_start, e.g. 'p-3f9a1c'.")


class MonitorArgs(BaseModel):
    command: str = Field(
        min_length=1,
        description=(
            "Command that prints a line only when something needs attention. "
            "Use unbuffered output (e.g. python -u). Redirect noisy stderr if needed."
        ),
    )
    name: str | None = Field(default=None, max_length=40)
    timeout: float = Field(
        default=300,
        gt=0,
        le=1800,
        description="Watch deadline in seconds; also capped by process.max_lifetime.",
    )


class LogsArgs(IdArgs):
    since: int | None = Field(
        default=None,
        ge=1,
        description="Only lines numbered from this one on: pass the previous result's next cursor.",
    )
    tail: int | None = Field(
        default=None,
        ge=1,
        le=5000,
        description=f"Only the last N matching lines (default {_DEFAULT_TAIL} without since).",
    )
    grep: str | None = Field(
        default=None,
        max_length=500,
        description="Only lines matching this regex; matches are numbered.",
    )


class NoArgs(BaseModel):
    pass
