"""``python -m lingcore --profile <path>`` — launch an agent over the CLI.

This is the composition root: it parses args, loads the profile, builds the
CLI frontend, wires the frontend's confirmation handler into the agent's tool
context, and runs the session loop. Everything below ``Agent`` and ``Frontend``
stays untouched when a different frontend is added.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, LingCoreError
from lingcore.io.base import run_session
from lingcore.io.cli import CLIFrontend

_DEFAULT_PROFILE = Path(__file__).parent / "profiles" / "coding"


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
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    profile_path = Path(args.profile)
    try:
        profile = AgentProfile.load(profile_path)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.workspace:
        profile.workspace = args.workspace

    # tool_options is a shared mutable dict: the frontend's "allow always" action
    # writes into it and the agent's ToolContext reads from it on every tool call.
    tool_options = dict(profile.tool_options)
    frontend = CLIFrontend(agent_name=profile.name, tool_options=tool_options)
    try:
        agent = Agent.from_profile(
            profile,
            confirm=frontend.confirm,
            base_dir=Path.cwd(),
            tool_options=tool_options,
        )
    except LingCoreError as e:
        print(f"failed to build agent: {e}", file=sys.stderr)
        return 2

    frontend.console.print(
        f"[bold]LingCore[/] · agent [cyan]{profile.name}[/] · "
        f"model [cyan]{profile.llm.model}[/] · workspace [cyan]{agent.tool_ctx.workspace}[/]"
    )
    frontend.console.print("[dim]Type your message. /exit to quit.[/]")

    try:
        await run_session(agent, frontend)
    except KeyboardInterrupt:
        frontend.console.print("\n[dim]interrupted[/]")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
