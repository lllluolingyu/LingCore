"""Supervised execution and durable aliases for external coding agents.

Shared by the bundled ``codex`` and ``claude-code`` skills. Each skill module
owns only what differs between the CLIs — how a session is established and how
its output is parsed — and imports everything else from here:

- ``OuterAgentSpec`` names one CLI (tool name, executable, label, state
  namespace); ``OUTER_AGENTS`` lists the bundled ones so ``lingcore doctor`` can
  validate their options and executables without importing skill code.
- ``OuterAgentArgs`` is the shared tool argument schema and ``frame_prompt``
  the shared brief, so both agents get identical instructions.
- ``run_outer_agent`` runs one turn under a bounded timeout/capture and kills
  the process group on timeout or cancellation; ``render_outer_agent_reply``
  offloads oversized output exactly like the builtin tools.
- Conversation aliases (``load/save_conversation_session``) map a logical
  name to the CLI's own session/thread id. The state file is confined under
  the profile directory and scoped by ``(workspace, LingCore session id)``, so
  resuming a LingCore session resumes its external conversations while two
  chats never share one. Without a persisted LingCore session
  (``--no-session``), the scope degrades to the workspace alone: every such
  run in that workspace shares one alias namespace.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from lingcore.errors import ConfigError, ToolError
from lingcore.paths import PathEscapeError, confined_directory
from lingcore.sandbox import (
    _kill_and_reap,
    _process_group_kwargs,
    resolve_sandbox_executable,
)
from lingcore.tools import ToolContext
from lingcore.tools.builtin._offload import offload_text


@dataclass(frozen=True, slots=True)
class OuterAgentSpec:
    """Static identity of one external coding-agent CLI."""

    # Tool name; doubles as the ``tool_options`` key.
    tool: str
    # Executable looked up on PATH when ``executable`` is not configured.
    program: str
    # Human-facing name used in prompts, confirmations, and errors.
    label: str
    # Namespace for alias state and offloaded output files.
    provider: str

    @property
    def output_source(self) -> str:
        return f"{self.provider}-agent"


CODEX = OuterAgentSpec(
    tool="codex_agent", program="codex", label="Codex", provider="codex"
)
CLAUDE_CODE = OuterAgentSpec(
    tool="claude_code_agent",
    program="claude",
    label="Claude Code",
    provider="claude-code",
)
OUTER_AGENTS: tuple[OuterAgentSpec, ...] = (CODEX, CLAUDE_CODE)


class OuterAgentOptions(BaseModel):
    """Validated per-tool process and output limits (``tool_options.<tool>``)."""

    model_config = ConfigDict(extra="forbid")

    executable: str | None = None
    timeout: float = Field(default=900.0, gt=0, le=3600)
    max_capture_bytes: int = Field(default=1_048_576, ge=1, le=16_777_216)
    offload_over_chars: int = Field(default=8_000, ge=0)
    max_output_chars: int = Field(default=16_000, ge=1)

    @field_validator("executable", mode="before")
    @classmethod
    def _blank_executable_is_unset(cls, value: object) -> object:
        """Treat an empty/whitespace-only override like an omitted one."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


def parse_outer_agent_options(raw: object) -> OuterAgentOptions:
    """Validate one tool's options; ``ConfigError`` on any unknown/invalid key."""
    try:
        return OuterAgentOptions.model_validate(raw if raw is not None else {})
    except (ValidationError, ValueError) as exc:
        raise ConfigError(str(exc)) from None


class OuterAgentArgs(BaseModel):
    """Arguments shared by every outer-agent tool."""

    prompt: str = Field(
        min_length=1,
        max_length=100_000,
        description="Task or follow-up question to send to the external agent.",
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
            "Logical conversation name. Reusing it continues the prior external "
            "session in this LingCore chat."
        ),
    )
    restart: bool = Field(
        default=False,
        description="Start this named conversation afresh instead of resuming it.",
    )


_CONSULT_ACTION = (
    "Analyze the request without modifying files. Return concise findings with "
    "file references."
)
_IMPLEMENT_ACTION = (
    "Implement the requested change in the shared workspace. Stay within the "
    "stated scope, verify your work, and report changed files and checks run."
)


def frame_prompt(spec: OuterAgentSpec, args: OuterAgentArgs) -> str:
    """Wrap the model's task in the brief every external collaborator receives."""
    action = _IMPLEMENT_ACTION if args.mode == "implement" else _CONSULT_ACTION
    return (
        f"You are an external {spec.label} collaborator called by a LingCore agent. "
        f"{action} Do not delegate to another agent.\n\n"
        f"Task from the orchestrating agent:\n{args.prompt.strip()}"
    )


def turn_header(
    spec: OuterAgentSpec, args: OuterAgentArgs, *, existing: str | None
) -> str:
    """First line of the tool result: which alias this was and what happened."""
    if existing is None:
        state = "started"
    else:
        state = "restarted" if args.restart else "resumed"
    return f"{spec.label} conversation {args.conversation!r} ({state}):"


# --------------------------------------------------------------------------- #
# Conversation aliases                                                        #
# --------------------------------------------------------------------------- #

_CONVERSATION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_STATE_DIRECTORY = Path(".lingcore") / "outer-agent-conversations"
_STATE_MAX_BYTES = 1_048_576
_LOCKS: weakref.WeakValueDictionary[tuple[int, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)
_LOCKS_GUARD = threading.Lock()


def validate_conversation_name(value: str) -> str:
    """Return a safe logical conversation alias or raise a tool-facing error."""
    if not _CONVERSATION_RE.fullmatch(value):
        raise ToolError(
            "conversation must be 1-64 characters and contain only letters, "
            "numbers, '.', '_', or '-', starting with a letter or number"
        )
    return value


def normalize_external_session_id(value: str) -> str:
    """Validate and normalize a Codex/Claude session UUID."""
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise ToolError(
            f"external agent returned an invalid session id: {value!r}"
        ) from None


def _state_base(ctx: ToolContext) -> Path:
    return (ctx.profile_dir or ctx.workspace).resolve()


def _state_filename(ctx: ToolContext, spec: OuterAgentSpec) -> str:
    workspace = str(ctx.workspace.resolve())
    # No persisted LingCore session ⇒ workspace-wide scope (see module doc).
    scope = ctx.session_id or "workspace-fallback"
    digest = hashlib.sha256(f"{workspace}\0{scope}".encode()).hexdigest()
    return f"{spec.provider}-{digest}.json"


def _read_conversations(ctx: ToolContext, spec: OuterAgentSpec) -> dict[str, str]:
    filename = _state_filename(ctx, spec)
    try:
        with confined_directory(_state_base(ctx), _STATE_DIRECTORY) as directory:
            if not directory.entry_exists(filename):
                return {}
            raw = directory.read_regular(filename, max_bytes=_STATE_MAX_BYTES)
    except FileNotFoundError:
        return {}
    except (OSError, PathEscapeError) as exc:
        raise ToolError(
            f"cannot read external-agent conversation state: {exc}"
        ) from None

    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported state format")
        if payload.get("provider") != spec.provider:
            raise ValueError("provider does not match the state filename")
        values = payload.get("conversations")
        if not isinstance(values, dict):
            raise ValueError("conversations is not an object")
        conversations: dict[str, str] = {}
        for name, session_id in values.items():
            if not isinstance(name, str) or not isinstance(session_id, str):
                raise ValueError("conversation entries must map strings to strings")
            validate_conversation_name(name)
            conversations[name] = normalize_external_session_id(session_id)
        return conversations
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, ToolError) as exc:
        raise ToolError(
            f"external-agent conversation state is corrupt ({filename}): {exc}"
        ) from None


def load_conversation_session(
    ctx: ToolContext, spec: OuterAgentSpec, conversation: str
) -> str | None:
    """Resolve an external session UUID for this LingCore conversation."""
    name = validate_conversation_name(conversation)
    return _read_conversations(ctx, spec).get(name)


def save_conversation_session(
    ctx: ToolContext, spec: OuterAgentSpec, conversation: str, session_id: str
) -> None:
    """Atomically persist an external session UUID under a logical alias."""
    name = validate_conversation_name(conversation)
    normalized = normalize_external_session_id(session_id)
    conversations = _read_conversations(ctx, spec)
    conversations[name] = normalized
    payload = json.dumps(
        {
            "version": 1,
            "provider": spec.provider,
            "conversations": conversations,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _STATE_MAX_BYTES:
        raise ToolError("external-agent conversation state has reached its size limit")
    filename = _state_filename(ctx, spec)
    temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
    try:
        with confined_directory(
            _state_base(ctx), _STATE_DIRECTORY, create=True
        ) as directory:
            try:
                with directory.open_exclusive(temporary, mode=0o600) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                directory.replace(temporary, filename)
            finally:
                directory.unlink(temporary, missing_ok=True)
    except (OSError, PathEscapeError) as exc:
        raise ToolError(
            f"cannot save external-agent conversation state: {exc}"
        ) from None


@asynccontextmanager
async def conversation_lock(
    ctx: ToolContext, spec: OuterAgentSpec, conversation: str
) -> AsyncIterator[None]:
    """Serialize turns that target the same durable external conversation."""
    name = validate_conversation_name(conversation)
    loop = asyncio.get_running_loop()
    identity = str(_state_base(ctx) / _STATE_DIRECTORY / _state_filename(ctx, spec))
    key = (id(loop), f"{identity}\0{name}")
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        yield


# --------------------------------------------------------------------------- #
# Process supervision                                                         #
# --------------------------------------------------------------------------- #


def _options(ctx: ToolContext, spec: OuterAgentSpec) -> OuterAgentOptions:
    raw = ctx.options.get(spec.tool, {}) if ctx.options else {}
    try:
        return parse_outer_agent_options(raw)
    except ConfigError as exc:
        raise ToolError(f"invalid {spec.tool} options: {exc}") from None


def resolve_outer_agent_executable(
    spec: OuterAgentSpec, configured: str | None, workspace: Path
) -> Path:
    """Locate the CLI (configured path or PATH lookup) outside the workspace."""
    if configured is None:
        candidate = shutil.which(spec.program)
        if candidate is None:
            raise ToolError(
                f"{spec.program} CLI is not installed or is not on PATH; install "
                "and authenticate it before using this skill"
            )
    else:
        expanded = os.path.expanduser(configured)
        candidate = expanded if os.path.isabs(expanded) else shutil.which(expanded)
        if candidate is None:
            raise ToolError(f"configured executable is not on PATH: {configured!r}")
    # Same trust rule as the sandbox backends: a real executable file that the
    # model cannot have written through the workspace tools.
    return resolve_sandbox_executable(candidate, workspace, f"{spec.label} agent")


async def _read_capped(
    process: asyncio.subprocess.Process, cap: int
) -> tuple[bytes, bool]:
    assert process.stdout is not None
    retained = bytearray()
    truncated = False
    while True:
        chunk = await process.stdout.read(65_536)
        if not chunk:
            break
        remaining = cap - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    await process.wait()
    return bytes(retained), truncated


async def _feed_prompt(process: asyncio.subprocess.Process, prompt: str) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        # Preserve the CLI's own diagnostic when it exits before reading stdin.
        pass
    finally:
        process.stdin.close()


async def _exchange(
    process: asyncio.subprocess.Process, prompt: str, cap: int
) -> tuple[bytes, bool]:
    _, output = await asyncio.gather(
        _feed_prompt(process, prompt), _read_capped(process, cap)
    )
    return output


async def confirm_outer_agent_write(ctx: ToolContext, spec: OuterAgentSpec) -> None:
    """Require fresh consent before an external agent may edit the workspace."""
    if ctx.confirm is None:
        raise ToolError(
            f"{spec.label} implementation mode requires confirmation, but this "
            "frontend has no confirmation handler"
        )
    approved = await ctx.confirm(
        f"Allow the external {spec.label} agent to modify files in {ctx.workspace}?"
    )
    if not approved:
        raise ToolError(f"user declined {spec.label} implementation mode")


@dataclass(frozen=True, slots=True)
class OuterAgentOutput:
    """Decoded output of one external-agent turn that exited successfully.

    ``text`` is the raw combined stdout/stderr (a skill may parse it and hand a
    ``dataclasses.replace``d copy to ``render_outer_agent_reply``); ``limits``
    are the options the turn ran under, reused for rendering.
    """

    text: str
    truncated: bool
    limits: OuterAgentOptions


def _offload(
    ctx: ToolContext, spec: OuterAgentSpec, text: str, limits: OuterAgentOptions
) -> str:
    return offload_text(
        ctx,
        source=spec.output_source,
        text=text,
        threshold=limits.offload_over_chars,
        fallback_max_chars=limits.max_output_chars,
    )


async def run_outer_agent(
    spec: OuterAgentSpec,
    *,
    arguments: list[str],
    prompt: str,
    ctx: ToolContext,
) -> OuterAgentOutput:
    """Run one external-agent turn with bounded output and hard cleanup.

    A non-zero exit becomes a ``ToolError`` carrying the (offloaded) output;
    the caller never sees a partially failed turn as a reply.
    """
    options = _options(ctx, spec)
    executable = resolve_outer_agent_executable(spec, options.executable, ctx.workspace)
    try:
        process = await asyncio.create_subprocess_exec(
            str(executable),
            *arguments,
            cwd=str(ctx.workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **_process_group_kwargs(),
        )
    except OSError as exc:
        raise ToolError(f"failed to launch {spec.label}: {exc}") from None

    try:
        output, truncated = await asyncio.wait_for(
            _exchange(process, prompt, options.max_capture_bytes),
            timeout=options.timeout,
        )
    except asyncio.TimeoutError:
        await asyncio.shield(_kill_and_reap(process))
        raise ToolError(
            f"{spec.label} timed out after {options.timeout:g}s and was killed"
        ) from None
    except BaseException:
        # Cancellation or any other interruption: never leave the child behind.
        await asyncio.shield(_kill_and_reap(process))
        raise

    text = output.decode("utf-8", errors="replace").strip() or "(no output)"
    code = process.returncode
    if code not in (0, None):
        body = _offload(ctx, spec, text, options)
        raise ToolError(f"{spec.label} exited with code {code}:\n{body}")
    return OuterAgentOutput(text=text, truncated=truncated, limits=options)


def render_outer_agent_reply(
    spec: OuterAgentSpec, ctx: ToolContext, output: OuterAgentOutput, *, header: str
) -> str:
    """Prefix ``header`` and offload/truncate the reply under the turn's limits."""
    text = output.text
    if output.truncated:
        text += (
            f"\n... (output exceeded {output.limits.max_capture_bytes} bytes and "
            "was truncated)"
        )
    return f"{header}\n{_offload(ctx, spec, text, output.limits)}"
