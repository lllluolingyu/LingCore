"""Supervised execution and durable aliases for external coding agents."""

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
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lingcore.errors import ToolError
from lingcore.paths import PathEscapeError, confined_directory
from lingcore.sandbox import _kill_and_reap
from lingcore.tools import ToolContext
from lingcore.tools.builtin._offload import offload_text


class OuterAgentOptions(BaseModel):
    """Validated per-tool process and output limits."""

    model_config = ConfigDict(extra="forbid")

    executable: str | None = None
    timeout: float = Field(default=900.0, gt=0, le=3600)
    max_capture_bytes: int = Field(default=1_048_576, ge=1, le=16_777_216)
    offload_over_chars: int = Field(default=8_000, ge=0)
    max_output_chars: int = Field(default=16_000, ge=1)


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


def _state_filename(ctx: ToolContext, provider: str) -> str:
    workspace = str(ctx.workspace.resolve())
    scope = ctx.session_id or "workspace-fallback"
    digest = hashlib.sha256(f"{workspace}\0{scope}".encode()).hexdigest()
    return f"{provider}-{digest}.json"


def _read_conversations(ctx: ToolContext, provider: str) -> dict[str, str]:
    filename = _state_filename(ctx, provider)
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
        if payload.get("provider") != provider:
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
    ctx: ToolContext, provider: str, conversation: str
) -> str | None:
    """Resolve an external session UUID for this LingCore conversation."""
    name = validate_conversation_name(conversation)
    return _read_conversations(ctx, provider).get(name)


def save_conversation_session(
    ctx: ToolContext, provider: str, conversation: str, session_id: str
) -> None:
    """Atomically persist an external session UUID under a logical alias."""
    name = validate_conversation_name(conversation)
    normalized = normalize_external_session_id(session_id)
    conversations = _read_conversations(ctx, provider)
    conversations[name] = normalized
    payload = json.dumps(
        {
            "version": 1,
            "provider": provider,
            "conversations": conversations,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _STATE_MAX_BYTES:
        raise ToolError("external-agent conversation state has reached its size limit")
    filename = _state_filename(ctx, provider)
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
    ctx: ToolContext, provider: str, conversation: str
) -> AsyncIterator[None]:
    """Serialize turns that target the same durable external conversation."""
    name = validate_conversation_name(conversation)
    loop = asyncio.get_running_loop()
    identity = str(_state_base(ctx) / _STATE_DIRECTORY / _state_filename(ctx, provider))
    key = (id(loop), f"{identity}\0{name}")
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        yield


def _options(ctx: ToolContext, key: str) -> OuterAgentOptions:
    raw = ctx.options.get(key, {}) if ctx.options else {}
    try:
        return OuterAgentOptions.model_validate(raw)
    except (ValidationError, ValueError) as exc:
        raise ToolError(f"invalid {key} options: {exc}") from None


def _executable(configured: str | None, program: str, workspace: Path) -> Path:
    candidate = configured or shutil.which(program)
    if candidate is None:
        raise ToolError(
            f"{program} CLI is not installed or is not on PATH; install and "
            "authenticate it before using this skill"
        )
    if not Path(candidate).is_absolute():
        candidate = shutil.which(candidate)
        if candidate is None:
            raise ToolError(f"configured executable is not on PATH: {configured!r}")
    path = Path(candidate)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ToolError(
            f"external agent executable is unavailable at {path}: {exc}"
        ) from None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ToolError(f"external agent executable is not executable: {resolved}")
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        pass
    else:
        raise ToolError(
            "external agent executable must be installed outside the writable workspace"
        )
    return resolved


def _process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": getattr(
                asyncio.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200
            )
        }
    return {"start_new_session": True}


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


async def confirm_outer_agent_write(ctx: ToolContext, label: str) -> None:
    """Require fresh consent before an external agent may edit the workspace."""
    if ctx.confirm is None:
        raise ToolError(
            f"{label} implementation mode requires confirmation, but this frontend "
            "has no confirmation handler"
        )
    approved = await ctx.confirm(
        f"Allow the external {label} agent to modify files in {ctx.workspace}?"
    )
    if not approved:
        raise ToolError(f"user declined {label} implementation mode")


async def run_outer_agent(
    *,
    program: str,
    arguments: list[str],
    prompt: str,
    ctx: ToolContext,
    option_key: str,
    label: str,
    output_source: str,
    transform_output: Callable[[str], str] | None = None,
) -> str:
    """Run one external-agent turn with bounded output and hard cleanup."""
    options = _options(ctx, option_key)
    executable = _executable(options.executable, program, ctx.workspace)
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
        raise ToolError(f"failed to launch {label}: {exc}") from None

    try:
        output, truncated = await asyncio.wait_for(
            _exchange(process, prompt, options.max_capture_bytes),
            timeout=options.timeout,
        )
    except asyncio.TimeoutError:
        await asyncio.shield(_kill_and_reap(process))
        raise ToolError(
            f"{label} timed out after {options.timeout:g}s and was killed"
        ) from None
    except asyncio.CancelledError:
        await asyncio.shield(_kill_and_reap(process))
        raise
    except BaseException:
        await asyncio.shield(_kill_and_reap(process))
        raise

    text = output.decode("utf-8", errors="replace").strip()
    if not text:
        text = "(no output)"
    code = process.returncode
    if code not in (0, None):
        body = offload_text(
            ctx,
            source=output_source,
            text=text,
            threshold=options.offload_over_chars,
            fallback_max_chars=options.max_output_chars,
        )
        raise ToolError(f"{label} exited with code {code}:\n{body}")
    if transform_output is not None:
        text = transform_output(text)
    if truncated:
        text += f"\n... (output exceeded {options.max_capture_bytes} bytes and was truncated)"
    body = offload_text(
        ctx,
        source=output_source,
        text=text,
        threshold=options.offload_over_chars,
        fallback_max_chars=options.max_output_chars,
    )
    return f"{label} response:\n{body}"
