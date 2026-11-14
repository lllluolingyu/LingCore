"""Publish host authority for panel transactions in a separate process."""

from __future__ import annotations

from typing import Any

from lingcore.delegation.models import digest
from lingcore.errors import ToolError
from lingcore.superagent.config import SuperagentConfig
from lingcore.superagent.core.engine import Engine
from lingcore.superagent.core.models import State


def constraints(profile: Any, context: Any) -> str:
    return digest(
        {
            "tool_options": profile.tool_options,
            "guardrail": profile.guardrail.model_dump(),
            "plugins": profile.plugins,
            "prepare": profile.superagent.executor.prepare,
            "environment": dict(context.environment),
            "binding": getattr(context.environment_binding, "binding_id", None),
        }
    )


def publish(engine: Engine) -> None:
    engine.state.authority = {
        "config": engine.config.model_dump(mode="json"),
        "ceiling": sorted(engine.ceiling),
        "roles": {k: sorted(v) for k, v in engine.roles.items()},
        "router_tools": sorted(engine.router_tools),
        "constraints": engine.constraints,
    }


def panel_engine(state: State) -> Engine:
    value = state.authority
    if not value:
        raise ToolError("service authority is unavailable; restart the superagent")
    return Engine(
        state,
        SuperagentConfig.model_validate(value["config"]),
        ceiling=frozenset(value["ceiling"]),
        roles={k: frozenset(v) for k, v in value["roles"].items()},
        router_tools=frozenset(value["router_tools"]),
        constraints_digest=value["constraints"],
    )
