"""Source composition and operator CLI for the unattended service."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.superagent.runtime import Runtime, validate_profile
from lingcore.superagent.store import SuperagentStore


def sources(runtime: Runtime) -> list[Any]:
    result = []
    root = getattr(runtime.profile, "_source_dir", runtime.state_dir)
    for name, config in runtime.config.sources.items():
        if not config.enabled:
            continue
        path = root / config.config if config.config else None
        if name == "lark":
            from lingcore.integrations.lark.application import create_lark_channel
            from lingcore.integrations.lark.config import load_lark_config
            from lingcore.superagent.sources.lark import LarkSource

            transport = load_lark_config(runtime.profile, path, transport_only=True)
            source: Any = LarkSource(
                runtime, create_lark_channel(transport, require_mention=False)
            )
        elif name == "telegram":
            from telegram.ext import Application

            from lingcore.integrations.telegram.config import load_telegram_config
            from lingcore.superagent.sources.telegram import TelegramSource

            transport_tg = load_telegram_config(
                runtime.profile, path, transport_only=True
            )
            if transport_tg.mode != "polling":
                raise ConfigError(
                    "superagent Telegram source currently requires polling mode"
                )
            source = TelegramSource(
                runtime,
                Application.builder()
                .token(transport_tg.resolve_token())
                .concurrent_updates(False)
                .build(),
            )
        elif name == "schedule":
            from lingcore.superagent.sources.schedule import ScheduleSource

            source = ScheduleSource(runtime, config)
        elif name == "github":
            from lingcore.superagent.sources.github import GitHubSource

            source = GitHubSource(runtime, config)
        else:
            raise ConfigError(f"unknown superagent source: {name}")
        runtime.channels[name] = source
        result.append(source)
    return result


async def serve(profile: AgentProfile, *, state_dir: Path | None = None) -> None:
    runtime = Runtime(profile, state_dir=state_dir)
    started = []
    try:
        for source in sources(runtime):
            started.append(source)
            await source.start()
        await runtime.run()
    finally:
        await asyncio.gather(
            *(source.close() for source in reversed(started)), return_exceptions=True
        )
        await runtime.close()


def service_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lingcore superagent")
    parser.add_argument(
        "action", choices=["start", "status"], nargs="?", default="start"
    )
    parser.add_argument("-p", "--profile", default="profiles/team")
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        profile = AgentProfile.load(args.profile)
        validate_profile(profile)
        assert profile.superagent is not None
        if args.action == "status":
            path = (
                args.state_dir or getattr(profile, "_source_dir")
            ) / profile.superagent.store
            if not path.exists():
                print(json.dumps({"tickets": [], "health": {}, "source_positions": {}}))
            else:
                from lingcore.superagent.status import status_payload

                store = SuperagentStore(path)
                try:
                    print(
                        json.dumps(
                            status_payload(store.read()), ensure_ascii=False, indent=2
                        )
                    )
                finally:
                    store.close()
        else:
            asyncio.run(serve(profile, state_dir=args.state_dir))
        return 0
    except KeyboardInterrupt:
        return 130
    except (ConfigError, OSError, ValueError) as exc:
        print(f"superagent: {exc}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as exc:
        print(
            f"superagent: optional dependency {exc.name} is missing; install lingcore[lark] or lingcore[telegram] for the configured source",
            file=sys.stderr,
        )
        return 2
