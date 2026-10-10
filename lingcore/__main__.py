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
import copy
import sys
from pathlib import Path

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, LingCoreError, SessionError
from lingcore.io.base import run_session
from lingcore.io.cli import CLIFrontend, session_table
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
        choices=("doctor", "telegram", "profile", "plugin"),
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
    if args.command == "plugin":
        parser.error("use plugin management as `lingcore plugin list|info|new`")
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

    console = Console()
    if store is None:
        console.print(f"[dim]{notice or 'sessions are disabled for this profile'}[/]")
        return 0
    sessions = store.list()
    if not sessions:
        console.print("[dim]no stored sessions[/]")
        return 0
    console.print(session_table(sessions))
    return 0


def _plugin_command(argv: list[str]) -> int:
    """Inspect plugin data or create a local skeleton without executing code."""
    import re

    from lingcore.plugins.discovery import (
        discover_plugins,
        engaged_plugins,
        plugin_skills,
    )
    from lingcore.plugins.manifest import EnvironmentName, component_path
    from lingcore.plugins.scaffold import scaffold_plugin

    parser = argparse.ArgumentParser(
        prog="lingcore plugin",
        description="Discover, inspect, or scaffold profile plugins.",
    )
    subcommands = parser.add_subparsers(dest="plugin_command", required=True)
    for action in ("list", "info", "new"):
        command = subcommands.add_parser(action)
        command.add_argument("--profile", "-p", default=str(_DEFAULT_PROFILE))
        if action != "list":
            command.add_argument("name")
        if action == "new":
            command.add_argument("--no-hooks", action="store_true")
            command.add_argument("--no-commands", action="store_true")
    args = parser.parse_args(argv)
    try:
        profile = AgentProfile.load(Path(args.profile))
        profile_dir = getattr(profile, "_source_dir", None)
        if args.plugin_command == "new":
            if profile_dir is None:
                raise ConfigError("plugin scaffolding requires a profile directory")
            destination = scaffold_plugin(
                profile_dir,
                args.name,
                hooks=not args.no_hooks,
                commands=not args.no_commands,
            )
            prefix = args.name.replace("-", "_")
            print(
                f"Created plugin at {destination}\nAdd to your profile (merge with existing lists):"
            )
            print(f"  plugins: [{args.name}]\n  tools: [{prefix}_echo]")
            return 0
        problems: dict[str, str] = {}
        discovered = discover_plugins(profile_dir, problems=problems)
        engagement = engaged_plugins(profile, discovered)

        def components(name: str) -> str:
            plugin = discovered[name]
            manifest = plugin.manifest
            items = ["tools"] if manifest.provides else []
            for label, path in (
                ("skills", manifest.skills),
                ("commands", manifest.commands),
                ("prompt", manifest.prompt),
            ):
                if path is not None and component_path(plugin.root, path).exists():
                    items.append(label)
            if manifest.hooks:
                items.append("hooks")
            return ", ".join(items) or "none"

        if args.plugin_command == "list":
            print("NAME  VERSION  SOURCE  ENGAGEMENT  COMPONENTS")
            for name, plugin in sorted(discovered.items()):
                reason = engagement.get(name, "not engaged")
                print(
                    f"{name}  {plugin.manifest.version}  {plugin.source}  {reason}  {components(name)}"
                )
                for shadowed in plugin.shadowed:
                    print(f"  shadows {shadowed.source}: {shadowed.root}")
            for name, problem in sorted(problems.items()):
                print(f"{name}  -  skipped  {problem}")
            return 0
        if args.name in problems:
            raise ConfigError(problems[args.name])
        if args.name not in discovered:
            raise ConfigError(f"unknown plugin {args.name!r}")
        plugin = discovered[args.name]
        manifest = plugin.manifest
        print(f"{manifest.name} {manifest.version}\n{manifest.description}")
        print(f"Source: {plugin.source}\nRoot: {plugin.root}\nAPI: {manifest.api}")
        if manifest.min_lingcore:
            print(f"Minimum LingCore: {manifest.min_lingcore}")
        print(
            f"Engagement: {engagement.get(args.name, 'not engaged')}\nComponents: {components(args.name)}"
        )
        print(
            f"Options key: {manifest.options_key}\nHook errors: {manifest.on_hook_error}\nHook timeout: {manifest.hook_timeout:g}s"
        )
        for label, value in (
            ("Module", manifest.module),
            ("Hooks class", manifest.hooks),
            ("Skills", manifest.skills),
            ("Commands", manifest.commands),
            ("Prompt", manifest.prompt),
        ):
            if value is not None:
                print(f"{label}: {value}")
        options = profile.tool_options.get(manifest.options_key or manifest.prefix, {})
        print("Environment names:")
        for declaration in manifest.environment:
            env_name = (
                declaration.name
                if isinstance(declaration, EnvironmentName)
                else options.get(declaration.option, declaration.default)
            )
            display = (
                env_name
                if isinstance(env_name, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name)
                else "<unset or invalid variable name>"
            )
            print(f"  {display} ({'required' if declaration.required else 'optional'})")
        names = set(manifest.provides)
        for skill in plugin_skills(plugin).values():
            names.update(skill.provides)
        print("Tools:")
        for name in sorted(names):
            print(
                f"  {name}: {'authorized' if name in profile.tools else 'unauthorized'}"
            )
        for requirement in manifest.requires.executables:
            print(
                f"Executable: {requirement.name}"
                + (f" (option {requirement.option})" if requirement.option else "")
            )
        for option_requirement in manifest.requires.options:
            print(
                f"Required option: {option_requirement.key}"
                + (f" — {option_requirement.hint}" if option_requirement.hint else "")
            )
        for shadowed in plugin.shadowed:
            print(f"Shadows {shadowed.source}: {shadowed.root}")
        return 0
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


def _print_saved_session(
    frontend: CLIFrontend, store: SessionStore | None, agent: Agent
) -> None:
    if store is None:
        return
    sid = getattr(agent.memory, "session_id", None)
    if not isinstance(sid, str):
        raise RuntimeError("session-backed memory did not expose its id")
    if store.get(sid) is not None:  # row exists only if something was said
        frontend.console.print(
            f"\n[dim]session [/][cyan]{sid[:8]}[/][dim] saved · resume with[/] "
            f"[bold]lingcore --resume {sid[:8]}[/]"
        )


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

        frontend = CLIFrontend(
            agent_name=profile.name, store=store, model=profile.llm.model
        )
        session_id = resume_meta.id if resume_meta else None
        first = True
        while True:
            # tool_options is a shared mutable dict: the frontend's "allow
            # always" action writes into it and the agent's ToolContext reads
            # from it on every tool call. Deep-copied per session so a session
            # allowlist never leaks into the profile or the next session.
            tool_options = copy.deepcopy(profile.tool_options)
            frontend.attach(tool_options)
            try:
                agent = Agent.from_profile(
                    profile,
                    confirm=frontend.confirm,
                    base_dir=Path.cwd(),
                    tool_options=tool_options,
                    session_store=store,
                    session_id=session_id,
                )
            except LingCoreError as e:
                print(f"failed to build agent: {e}", file=sys.stderr)
                return 2
            frontend.set_commands(agent.commands)
            if store is not None:
                frontend.set_session(getattr(agent.memory, "session_id", None))

            if first:
                frontend.show_banner(
                    model=profile.llm.model,
                    workspace=agent.tool_ctx.workspace,
                    notice=notice,
                )
            if session_id is not None and store is not None:
                meta = store.get(session_id)
                if meta is not None:
                    frontend.show_resume(meta, agent.memory.messages)
            elif not first:
                frontend.console.rule("[dim]new session[/]", style="dim")
            first = False

            try:
                await run_session(agent, frontend)
            except asyncio.CancelledError:
                # asyncio.Runner implements Ctrl-C by cancelling the main task.
                # Agent.run deliberately retains its checkpoint on cancellation;
                # repair it before the store closes, then let Runner translate
                # the cancellation to KeyboardInterrupt (exit status 130).
                if agent.turn_pending_finalization:
                    try:
                        for plugin_notice in agent.drain_plugin_notices():
                            frontend.render(plugin_notice)
                        frontend.render(
                            agent.finalize_cancelled_turn(reason="interrupted")
                        )
                    except Exception as exc:
                        frontend.console.print(
                            f"failed to clean up interrupted turn: {exc}",
                            style="red",
                            markup=False,
                        )
                raise
            except KeyboardInterrupt:
                frontend.console.print("\n[dim]interrupted[/]")
            finally:
                await agent.aclose()

            _print_saved_session(frontend, store, agent)
            switch = frontend.take_session_switch()
            if switch is None:
                break
            session_id = switch.session_id
        return 0
    finally:
        if store is not None:
            store.close()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "profile":
        return _profile_command(raw_argv[1:])
    if raw_argv and raw_argv[0] == "plugin":
        return _plugin_command(raw_argv[1:])
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
