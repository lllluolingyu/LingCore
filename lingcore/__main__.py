"""``python -m lingcore --profile <path>`` — launch an agent over the CLI.

This is the composition root: it parses args, loads the profile, opens the
profile's session store (history + resume), builds the CLI frontend, wires the
frontend's confirmation handler into the agent's tool context, and runs the
session loop. Everything below ``Agent`` and ``Frontend`` stays untouched when
a different frontend is added.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, LingCoreError, SessionError
from lingcore.io.base import run_session
from lingcore.io.cli import CLIFrontend, rel_time
from lingcore.sessions import SessionMeta, SessionStore, open_store

# Bundled profiles live at the repo root — outside the package tree — so their
# sessions.db / memory.md are writable. The default only resolves in a repo
# checkout; wheel installs must pass --profile.
_DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "coding"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lingcore",
        description="Run a LingCore agent from a profile over an interactive CLI.",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default=str(_DEFAULT_PROFILE),
        help="Path to an agent profile YAML (default: built-in coding profile).",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default=None,
        help="Override the profile's workspace directory.",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--continue",
        "-c",
        dest="continue_",
        action="store_true",
        help="Resume the most recent session for this profile.",
    )
    g.add_argument(
        "--resume",
        metavar="ID",
        help="Resume a stored session by unique id prefix.",
    )
    g.add_argument(
        "--no-session",
        action="store_true",
        help="Run without persisting this session.",
    )
    g.add_argument(
        "--list-sessions",
        action="store_true",
        help="List stored sessions for this profile and exit.",
    )
    return parser.parse_args(argv)


def _print_sessions(store: SessionStore | None, notice: str | None) -> int:
    """``--list-sessions``: print a table and exit (never builds an Agent)."""
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    if store is None:
        console.print(f"[dim]{notice or 'sessions are disabled for this profile'}[/]")
        return 0
    sessions = store.list()
    if not sessions:
        console.print("[dim]no stored sessions[/]")
        return 0
    table = Table(box=None, pad_edge=False)
    table.add_column("id", style="cyan")
    table.add_column("title")
    table.add_column("msgs", justify="right")
    table.add_column("updated", style="dim")
    table.add_column("created", style="dim")
    for s in sessions:
        table.add_row(
            s.id[:8],
            escape(s.title or "(untitled)"),
            str(s.message_count),
            rel_time(s.updated_at),
            rel_time(s.created_at),
        )
    console.print(table)
    return 0


async def _main_async(args: argparse.Namespace) -> int:
    profile_path = Path(args.profile)
    try:
        profile = AgentProfile.load(profile_path)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.workspace:
        profile.workspace = args.workspace

    try:
        store, notice = (None, None) if args.no_session else open_store(profile)
    except (ConfigError, SessionError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    try:
        if args.list_sessions:
            return _print_sessions(store, notice)

        resume_meta: SessionMeta | None = None
        if args.continue_ or args.resume:
            if store is None:
                reason = notice or "sessions are disabled for this profile"
                print(f"cannot resume: {reason}", file=sys.stderr)
                return 2
            if args.resume:
                try:
                    resume_meta = store.resolve_prefix(args.resume)
                except SessionError as e:
                    print(str(e), file=sys.stderr)
                    return 2
            else:
                resume_meta = store.latest()
                if resume_meta is None:
                    print(
                        f"no sessions to continue for profile {profile.name!r}",
                        file=sys.stderr,
                    )
                    return 2

        # tool_options is a shared mutable dict: the frontend's "allow always"
        # action writes into it and the agent's ToolContext reads from it on
        # every tool call.
        tool_options = dict(profile.tool_options)
        frontend = CLIFrontend(agent_name=profile.name, tool_options=tool_options)
        try:
            agent = Agent.from_profile(
                profile,
                confirm=frontend.confirm,
                base_dir=Path.cwd(),
                tool_options=tool_options,
                session_store=store,
                session_id=resume_meta.id if resume_meta else None,
            )
        except LingCoreError as e:
            print(f"failed to build agent: {e}", file=sys.stderr)
            return 2

        frontend.console.print(
            f"[bold]LingCore[/] · agent [cyan]{profile.name}[/] · "
            f"model [cyan]{profile.llm.model}[/] · workspace [cyan]{agent.tool_ctx.workspace}[/]"
        )
        if notice:
            frontend.console.print(f"[dim]{notice}[/]")
        if resume_meta is not None:
            frontend.show_resume(resume_meta, agent.memory.messages)
        frontend.console.print("[dim]Type your message. /exit to quit.[/]")

        try:
            await run_session(agent, frontend)
        except KeyboardInterrupt:
            frontend.console.print("\n[dim]interrupted[/]")

        if store is not None:
            sid: str = agent.memory.session_id  # SessionMemory when a store is wired
            if store.get(sid) is not None:  # row exists only if something was said
                frontend.console.print(
                    f"[dim]session [/][cyan]{sid[:8]}[/][dim] saved — resume with: lingcore -c[/]"
                )
        return 0
    finally:
        if store is not None:
            store.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
