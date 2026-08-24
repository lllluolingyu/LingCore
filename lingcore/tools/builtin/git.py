"""Read-only Git inspection for the coding agent.

The shell tool is intentionally confirmation-gated because it can execute
arbitrary code.  This builtin covers the common read-only Git operations with
structured arguments instead: the model cannot inject flags or shell syntax,
and every invocation stays rooted at the workspace checkout.

Repository-changing and networked operations deliberately do not exist here;
they continue to go through ``run_shell`` and its consent/sandbox policy.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import signal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.paths import PathEscapeError, resolve_confined
from lingcore.tools import ToolContext, tool

_TIMEOUT_SECONDS = 20.0
_MAX_CAPTURE_BYTES = 1024 * 1024
_MAX_OUTPUT_CHARS = 30_000
_MAX_ERROR_CHARS = 4_000
_MAX_PATHS = 100
_MAX_PATH_CHARS = 4_096
_MAX_REVISION_CHARS = 256

GitAction = Literal["status", "diff", "log", "show", "branches"]


class GitArgs(BaseModel):
    action: GitAction = Field(
        description=(
            "Read-only Git operation: `status`, `diff`, `log`, `show`, or `branches`."
        )
    )
    revision: str | None = Field(
        default=None,
        description=(
            "Revision or revision range for diff/log/show (for example `HEAD~1` "
            "or `main..feature`). show defaults to HEAD."
        ),
    )
    staged: bool = Field(
        default=False,
        description="For diff only, compare the index instead of the working tree.",
    )
    paths: list[str] = Field(
        default_factory=list,
        max_length=_MAX_PATHS,
        description=(
            "Optional workspace-relative literal paths for status/diff/log/show. "
            "Git pathspec magic is disabled."
        ),
    )
    max_count: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum commits returned by log (1-100).",
    )


def _git_executable() -> str:
    """Find Git on the platform's trusted default executable path."""
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise ToolError("git executable is not installed or is not on the system path")
    return str(Path(executable).resolve())


def _git_environment(workspace: Path) -> dict[str, str]:
    """Return a small environment with ambient Git behavior disabled."""
    environment = {
        "PATH": os.defpath,
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_LITERAL_PATHSPECS": "1",
        # A partial clone must not turn an inspection into a network request.
        "GIT_ALLOW_PROTOCOL": "",
        # Git must not discover a repository above the workspace.
        "GIT_CEILING_DIRECTORIES": str(workspace.parent),
    }
    # Git for Windows needs these to locate basic OS facilities. They do not
    # widen repository or command behavior.
    for name in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _validate_checkout(ctx: ToolContext) -> Path:
    """Require a normal checkout whose repository metadata is in the workspace.

    Refusing parent repositories, linked worktrees, and separate Git directories
    keeps even object/config reads within the workspace boundary. A normal clone
    or ``git init`` checkout has a real ``.git`` directory and is accepted.
    """
    workspace = ctx.workspace.resolve()
    marker = workspace / ".git"
    if marker.is_symlink() or not marker.is_dir():
        raise ToolError(
            "git requires the workspace root to be a normal checkout with its own "
            ".git directory; parent repositories and linked/separate worktrees are "
            "refused"
        )
    try:
        resolved_marker = marker.resolve(strict=True)
        resolved_marker.relative_to(workspace)
    except (OSError, ValueError):
        raise ToolError("the checkout's .git directory escapes the workspace") from None

    # Check the principal metadata/object paths Git reads for these operations.
    # This catches the useful escape cases without recursively walking a large
    # object database on every call.
    for relative in (
        "HEAD",
        "config",
        "config.worktree",
        "index",
        "logs",
        "objects",
        "packed-refs",
        "refs",
    ):
        metadata = resolved_marker / relative
        if not metadata.exists() and not metadata.is_symlink():
            continue
        try:
            metadata.resolve(strict=True).relative_to(workspace)
        except (OSError, ValueError):
            raise ToolError(
                f"the checkout's Git metadata path escapes the workspace: {relative}"
            ) from None

    # A commondir file redirects refs and objects, most commonly for linked
    # worktrees. Keep that redirection confined too, even though the normal
    # linked-worktree marker above is already refused.
    commondir = resolved_marker / "commondir"
    if commondir.exists():
        if commondir.is_symlink():
            raise ToolError("the checkout's Git common-directory file is a symlink")
        try:
            target_text = commondir.read_text("utf-8").strip()
            target = (resolved_marker / target_text).resolve(strict=True)
            target.relative_to(workspace)
        except (OSError, UnicodeError, ValueError):
            raise ToolError(
                "the checkout's Git common directory escapes the workspace"
            ) from None

    # Alternates can make `show` read object contents from an arbitrary path.
    # They are uncommon in ordinary coding checkouts, so fail closed instead of
    # trying to reproduce Git's complete alternate-object resolution rules.
    if (resolved_marker / "objects" / "info" / "alternates").exists():
        raise ToolError(
            "Git object alternates are not supported by the confined git tool"
        )
    return workspace


def _validate_revision(revision: str) -> str:
    value = revision.strip()
    if (
        not value
        or len(value) > _MAX_REVISION_CHARS
        or value.startswith("-")
        or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)
    ):
        raise ToolError(
            "invalid Git revision (must be a non-option token without whitespace)"
        )
    return value


def _validate_paths(ctx: ToolContext, paths: list[str]) -> list[str]:
    if len(paths) > _MAX_PATHS:
        raise ToolError(f"too many Git paths ({len(paths)}; limit {_MAX_PATHS})")
    validated: list[str] = []
    for path in paths:
        candidate = Path(path)
        if (
            not path
            or len(path) > _MAX_PATH_CHARS
            or "\x00" in path
            or candidate.is_absolute()
            or any(part == ".." for part in candidate.parts)
        ):
            raise ToolError(f"Git path escapes workspace: {path!r}")
        try:
            resolve_confined(ctx.workspace, path)
        except PathEscapeError as exc:
            raise ToolError(str(exc)) from None
        validated.append(path)
    return validated


def _operation_args(args: GitArgs, ctx: ToolContext) -> list[str]:
    paths = _validate_paths(ctx, args.paths)
    revision = _validate_revision(args.revision) if args.revision is not None else None

    if args.staged and args.action != "diff":
        raise ToolError("staged is supported only for the diff action")
    if args.action in {"status", "branches"} and revision is not None:
        raise ToolError(f"revision is not supported for the {args.action} action")
    if args.action == "branches" and paths:
        raise ToolError("paths are not supported for the branches action")

    if args.action == "status":
        command = [
            "status",
            "--short",
            "--branch",
            "--untracked-files=normal",
            "--ignore-submodules=all",
        ]
    elif args.action == "diff":
        command = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--ignore-submodules=all",
        ]
        if args.staged:
            command.append("--cached")
        if revision is not None:
            command.append(revision)
    elif args.action == "log":
        command = [
            "log",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--decorate=short",
            "--date=short",
            "--format=%h %ad %d %s",
            f"--max-count={args.max_count}",
        ]
        if revision is not None:
            command.append(revision)
    elif args.action == "show":
        command = [
            "show",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--format=fuller",
            "--stat",
            "--patch",
            revision or "HEAD",
        ]
    else:
        command = [
            "branch",
            "--list",
            "--no-color",
            "--format=%(HEAD) %(refname:short) %(upstream:short) %(upstream:trackshort)",
        ]

    if paths:
        command.extend(["--", *paths])
    return command


async def _read_capped(
    proc: asyncio.subprocess.Process,
) -> tuple[bytes, bool, bool]:
    """Drain output while supervising the main Git process.

    A repository-controlled helper should be disabled by the fixed options, but
    if a Git descendant nevertheless inherits stdout and outlives Git, it must
    not hold this tool open until the full command timeout. ``pipe_held_open``
    lets the caller report that capture ended after the main process exited.
    """
    stream = proc.stdout
    assert stream is not None
    output = bytearray()
    truncated = False

    async def drain() -> None:
        nonlocal truncated
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            remaining = _MAX_CAPTURE_BYTES - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                truncated = True

    reader = asyncio.create_task(drain())
    try:
        await proc.wait()
        try:
            await asyncio.wait_for(reader, timeout=0.5)
            pipe_held_open = False
        except TimeoutError:
            pipe_held_open = True
        return bytes(output), truncated, pipe_held_open
    finally:
        if not reader.done():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)


async def _abort(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass
    await proc.wait()


@tool(
    name="git",
    description=(
        "Inspect the workspace Git checkout without confirmation using structured, "
        "read-only status/diff/log/show/branches operations. This tool cannot add, "
        "commit, restore, fetch, push, or otherwise change the repository; use "
        "confirmation-gated run_shell for those operations."
    ),
)
async def git(args: GitArgs, ctx: ToolContext) -> str:
    workspace = _validate_checkout(ctx)
    operation = _operation_args(args, ctx)
    executable = _git_executable()
    # These fixed overrides suppress local executable hooks/helpers for every
    # supported operation. Operation-specific flags also disable diff/textconv.
    command = [
        executable,
        "--no-pager",
        "--literal-pathspecs",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "credential.helper=",
        *operation,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace,
            env=_git_environment(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise ToolError(f"could not start git: {exc}") from None

    try:
        raw, truncated, pipe_held_open = await asyncio.wait_for(
            _read_capped(proc), timeout=_TIMEOUT_SECONDS
        )
    except TimeoutError:
        await asyncio.shield(_abort(proc))
        raise ToolError(
            f"git {args.action} timed out after {_TIMEOUT_SECONDS:g}s and was killed"
        ) from None
    except asyncio.CancelledError:
        await asyncio.shield(_abort(proc))
        raise
    except BaseException:
        await asyncio.shield(_abort(proc))
        raise

    output = raw.decode("utf-8", errors="replace").rstrip()
    if proc.returncode != 0:
        detail = output or "(no output)"
        if len(detail) > _MAX_ERROR_CHARS:
            detail = detail[:_MAX_ERROR_CHARS] + "\n... (error output truncated)"
        raise ToolError(
            f"git {args.action} failed (exit code {proc.returncode}):\n{detail}"
        )
    if truncated:
        output += (
            f"\n... (output exceeded {_MAX_CAPTURE_BYTES} bytes and was truncated)"
        )
    if pipe_held_open:
        output += "\n... (a Git child kept output open; capture stopped)"
    if not output:
        output = "(no output)"
    elif len(output) > _MAX_OUTPUT_CHARS:
        omitted = len(output) - _MAX_OUTPUT_CHARS
        output = output[:_MAX_OUTPUT_CHARS] + (
            f"\n... (truncated, {omitted} more chars; narrow paths or revision)"
        )

    display = shlex.join(["git", *operation])
    return f"$ {display}\n(read-only)\n{output}"


__all__ = ["GitArgs", "git"]
