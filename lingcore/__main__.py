"""Launch a profile over the CLI or inspect it with ``lingcore doctor``.

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
from lingcore.profiles import (
    PROFILE_TEMPLATE_FILES,
    default_profile_path,
    initialize_profile,
    user_profiles_dir,
)
from lingcore.sessions import SessionMeta, SessionStore, open_store

# Source checkouts use their repository example; installed wheels use the
# writable copy created by ``lingcore profile init`` in the user state dir.
_DEFAULT_PROFILE = default_profile_path()


def _telegram_dependency_missing(exc: ModuleNotFoundError) -> bool:
    name = exc.name or ""
    return name == "telegram" or name.startswith("telegram.") or name == "tornado"


def _print_telegram_install_hint() -> None:
    print(
        "Telegram support is not installed. Install it with:\n"
        'pip install "lingcore[telegram]"',
        file=sys.stderr,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lingcore",
        description="Run, diagnose, or initialize a LingCore agent profile.",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default=str(_DEFAULT_PROFILE),
        help="Path to an agent profile YAML (default: repository coding "
        "profile, or the user copy created by `lingcore profile init`).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("doctor", "telegram", "profile"),
        help="Run diagnostics, Telegram, or manage installed profile templates.",
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
    parser.add_argument(
        "--telegram-config",
        default=None,
        help="Telegram YAML path (default: <profile>/telegram.yaml).",
    )
    parser.add_argument(
        "--telegram-mode",
        choices=("polling", "webhook"),
        default=None,
        help="Override the Telegram config's polling/webhook mode.",
    )
    args = parser.parse_args(argv)
    if args.command == "profile":
        parser.error("use profile management as `lingcore profile init|list`")
    if args.command == "telegram":
        conflicts = []
        if args.continue_:
            conflicts.append("--continue")
        if args.resume:
            conflicts.append("--resume")
        if args.no_session:
            conflicts.append("--no-session")
        if args.list_sessions:
            conflicts.append("--list-sessions")
        if args.workspace is not None:
            conflicts.append("--workspace")
        if conflicts:
            parser.error("Telegram mode does not accept " + ", ".join(conflicts))
    elif args.telegram_config is not None and args.command != "doctor":
        parser.error("--telegram-config is only valid with doctor or telegram")
    if args.telegram_mode is not None and args.command != "telegram":
        parser.error("--telegram-mode is only valid with telegram")
    return args


def _profile_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lingcore profile",
        description="List templates or initialize a writable agent profile.",
    )
    commands = parser.add_subparsers(dest="profile_command", required=True)
    init = commands.add_parser(
        "init", help="Copy an immutable template into writable user state."
    )
    init.add_argument(
        "template",
        nargs="?",
        choices=tuple(PROFILE_TEMPLATE_FILES),
        default="coding",
        help="Template to copy (default: coding).",
    )
    target = init.add_mutually_exclusive_group()
    target.add_argument(
        "--name",
        help="Name under the user profiles directory (default: template name).",
    )
    target.add_argument(
        "--destination",
        "-d",
        help="Explicit destination directory instead of the user state dir.",
    )
    commands.add_parser("list", help="List the immutable templates in this wheel.")
    args = parser.parse_args(argv)
    if args.profile_command == "list":
        print("Available profile templates:")
        for name in PROFILE_TEMPLATE_FILES:
            print(f"  {name}")
        print(f"Writable profiles directory: {user_profiles_dir()}")
        return 0
    try:
        destination = initialize_profile(
            args.template,
            name=args.name,
            destination=args.destination,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(f"Initialized {args.template!r} profile at {destination}")
    if (destination / ".env.example").is_file():
        print("Next: copy .env.example to .env, fill the required values, and run:")
    else:
        print("Run it with:")
    if destination == _DEFAULT_PROFILE:
        print("  lingcore doctor")
        print("  lingcore")
    else:
        print(f"  lingcore doctor --profile {destination}")
        print(f"  lingcore --profile {destination}")
    return 0


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
        if args.profile == str(_DEFAULT_PROFILE) and not _DEFAULT_PROFILE.exists():
            print(
                "note: initialize the wheel's default writable profile with "
                "`lingcore profile init`, or pass --profile pointing at an "
                "existing profile directory.",
                file=sys.stderr,
            )
        return 2

    if args.workspace:
        profile.workspace = args.workspace

    if args.command == "doctor":
        from lingcore.doctor import diagnose_profile, print_doctor_report

        additional_requirements = None
        if args.telegram_config is not None:
            from lingcore.integrations.telegram import load_telegram_config

            try:
                telegram = load_telegram_config(
                    profile,
                    args.telegram_config,
                    require_secrets=False,
                )
            except ConfigError as e:
                print(f"config error: {e}", file=sys.stderr)
                return 2
            additional_requirements = {
                telegram.token_env: "telegram.token_env",
            }
            if (
                telegram.mode == "webhook"
                and telegram.webhook.secret_token_env is not None
            ):
                additional_requirements[telegram.webhook.secret_token_env] = (
                    "telegram.webhook.secret_token_env"
                )
        report = diagnose_profile(
            profile,
            additional_environment_requirements=additional_requirements,
        )
        print_doctor_report(report)
        return report.exit_code

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
        except asyncio.CancelledError:
            # asyncio.Runner implements Ctrl-C by cancelling the main task.
            # Agent.run deliberately retains its checkpoint on cancellation;
            # repair it before the store closes, then let Runner translate the
            # cancellation to KeyboardInterrupt (main returns exit status 130).
            if agent.turn_pending_finalization:
                try:
                    frontend.render(agent.finalize_cancelled_turn(reason="interrupted"))
                except Exception as exc:
                    frontend.console.print(
                        f"failed to clean up interrupted turn: {exc}",
                        style="red",
                        markup=False,
                    )
            raise
        except KeyboardInterrupt:
            frontend.console.print("\n[dim]interrupted[/]")

        if store is not None:
            sid = getattr(agent.memory, "session_id", None)
            if not isinstance(sid, str):
                raise RuntimeError("session-backed memory did not expose its id")
            if store.get(sid) is not None:  # row exists only if something was said
                frontend.console.print(
                    f"[dim]session [/][cyan]{sid[:8]}[/][dim] saved — resume with: lingcore -c[/]"
                )
        return 0
    finally:
        if store is not None:
            store.close()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "profile":
        return _profile_command(raw_argv[1:])
    args = _parse_args(raw_argv)
    if args.command == "telegram":
        try:
            # This import is deliberately after command selection. Ordinary
            # LingCore and PTB-free doctor runs never import python-telegram-bot.
            from lingcore.integrations.telegram.application import run_telegram
        except ModuleNotFoundError as exc:
            if _telegram_dependency_missing(exc):
                _print_telegram_install_hint()
                return 2
            raise
        try:
            return run_telegram(
                args.profile,
                telegram_config_path=args.telegram_config,
                mode=args.telegram_mode,
            )
        except ConfigError as e:
            print(f"config error: {e}", file=sys.stderr)
            return 2
        except ModuleNotFoundError as exc:
            if _telegram_dependency_missing(exc):
                _print_telegram_install_hint()
                return 2
            raise
        except KeyboardInterrupt:
            return 130
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
