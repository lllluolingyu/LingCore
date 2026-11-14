"""Inspect, resume, continue and prune persisted delegated work."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from lingcore._sessions.runs import RunJournal
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.delegation.retention import prune
from lingcore.delegation.supervisor import Supervisor
from lingcore.delegation.workspaces import WorkspaceManager
from lingcore.errors import ConfigError, SessionError, ToolError
from lingcore.io.cli import CLIFrontend
from lingcore.sessions import open_store


def output(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


async def execute(
    args: Any, profile: AgentProfile, store: Any, journal: RunJournal
) -> int:
    if args.action == "continue":
        node = journal.worker(args.id)
        record = journal.get(node.run_id)
    elif args.action == "resume":
        record = journal.get(args.id)
    else:
        record = None
    parent = record.parent_session_id if record else args.session
    frontend = CLIFrontend(profile.name, store=store, model=profile.llm.model)
    async with Agent.from_profile(
        profile, session_store=store, session_id=parent, confirm=frontend.confirm
    ) as agent:
        supervisor = agent.tool_ctx.delegation or Supervisor(agent, profile)
        agent.tool_ctx.delegation = supervisor
        snapshot = agent._authorized_tool_names()
        if args.action == "start":
            from lingcore.workflows.runtime import run_workflow

            output(
                await run_workflow(
                    supervisor, snapshot, name=args.id, args=json.loads(args.args)
                )
            )
        elif args.action == "resume":
            from lingcore.workflows.runtime import resume_workflow

            output(await resume_workflow(supervisor, args.id, snapshot))
        else:
            prompt = args.prompt
            while True:
                if prompt is None:
                    raw = await frontend.read_input()
                    if raw is None:
                        break
                    prompt = raw if isinstance(raw, str) else raw.text
                output(await supervisor.continue_worker(args.id, prompt, snapshot))
                if args.prompt is not None:
                    break
                prompt = None
    return 0


def run_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lingcore run")
    parser.add_argument(
        "action", choices=["list", "show", "start", "resume", "continue", "prune"]
    )
    parser.add_argument("id", nargs="?")
    parser.add_argument("-p", "--profile", default="profiles/coding")
    parser.add_argument("--session")
    parser.add_argument("--prompt")
    parser.add_argument("--args", default="{}", help="Workflow input JSON")
    args = parser.parse_args(argv)
    if args.action not in {"list", "prune"} and args.id is None:
        parser.error("this action requires a run, worker, or workflow id")
    store = None
    try:
        profile = AgentProfile.load(args.profile)
        store, notice = open_store(profile)
        if store is None:
            raise ConfigError(notice or "runs require persistent sessions")
        journal = RunJournal(store)
        if args.action == "list":
            output(
                [
                    {
                        "id": r.id,
                        "name": r.name,
                        "status": r.status,
                        "parent_session_id": r.parent_session_id,
                        "agents": r.agents,
                        "usage": r.usage.model_dump(),
                    }
                    for r in journal.list(args.session)
                ]
            )
        elif args.action == "show":
            output(
                {
                    "run": journal.get(args.id).model_dump(),
                    "nodes": [n.model_dump() for n in journal.nodes(args.id)],
                }
            )
        elif args.action == "prune":
            output(
                prune(
                    journal,
                    WorkspaceManager(
                        profile.workspace_path(), profile.delegation.workspace
                    ),
                    profile.delegation.retention,
                )
            )
        else:
            return asyncio.run(execute(args, profile, store, journal))
        return 0
    except KeyboardInterrupt:
        return 130
    except (ConfigError, SessionError, ToolError, ValueError, OSError) as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 2
    finally:
        if store:
            store.close()
