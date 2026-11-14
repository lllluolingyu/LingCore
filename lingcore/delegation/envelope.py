"""Authority only narrows; text and skill activation cannot grant tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lingcore.delegation.models import Envelope, NodeClass, WorkspaceMode, digest
from lingcore.delegation.roles import DELEGATION_TOOLS, STATE_WRITERS, Role
from lingcore.errors import ConfigError

if TYPE_CHECKING:
    from lingcore.agent import Agent
    from lingcore.config import AgentProfile
    from lingcore.tools import ToolRegistry

READERS = frozenset({"read_file", "list_dir", "search", "git", "knowledge", "pdf2md"})
PURE = frozenset({"todo_write", "activate_skill", "deactivate_skill"})
NETWORK = frozenset({"fetch_url", "search_web"})


def high_risk_tools(
    parent: Agent, tools: frozenset[str], *, registry: ToolRegistry | None = None
) -> frozenset[str]:
    from lingcore.skills import DEFAULT_HIGH_RISK_TOOLS

    floor = DEFAULT_HIGH_RISK_TOOLS | frozenset(
        parent.tool_ctx.options.get("activate_skill", {}).get("high_risk_tools") or []
    )
    if parent.skill_state is not None:
        floor |= parent.skill_state.high_risk_tools
    registry = registry or parent.tools
    scoped = frozenset(parent.tools.names())
    return tools & (
        floor
        | {
            t
            for t in tools
            if registry.get(t).high_risk
            or t in scoped
            and parent.tools.get(t).high_risk
        }
    )


def effecting(tools: frozenset[str], high_risk: frozenset[str]) -> bool:
    # Unknown plugin tools fail conservatively: only audited readers are reusable.
    return bool(
        tools & (high_risk | STATE_WRITERS)
        or tools - READERS - NETWORK - PURE - DELEGATION_TOOLS
    )


def node_class(envelope: Envelope) -> NodeClass:
    if (
        effecting(envelope.tools, envelope.high_risk)
        or envelope.tools & DELEGATION_TOOLS
    ):
        return "effecting"
    if envelope.workspace == "frozen":
        return "frozen"
    if envelope.tools & (READERS | NETWORK) or any(
        t.startswith("browser_") for t in envelope.tools
    ):
        return "live"
    return "pure"


def compute(
    parent: Agent,
    profile: AgentProfile,
    role: Role,
    snapshot: frozenset[str],
    *,
    depth: int,
    tools: list[str] | None = None,
    workspace: WorkspaceMode | None = None,
    recorded: Envelope | None = None,
) -> Envelope:
    wanted = frozenset(role.tools)
    if tools is not None and not set(tools) <= wanted:
        raise ConfigError("requested tools may only narrow the role")
    effective = wanted & snapshot & frozenset(profile.tools)
    if tools is not None:
        effective &= frozenset(tools)
    if depth >= profile.delegation.max_depth:
        effective -= DELEGATION_TOOLS
    mode = workspace or role.workspace
    if mode != role.workspace and not (
        role.workspace == "shared" and mode in {"frozen", "scratch"}
    ):
        raise ConfigError("requested workspace may only narrow the role")
    high = high_risk_tools(parent, effective)
    from lingcore.delegation.worker import effective_options

    options = effective_options(parent)
    if (
        "knowledge" in effective
        and options.get("knowledge", {}).get("backend", "grep") != "grep"
    ):
        high |= {"knowledge"}  # index writes and remote embedding are not frozen reads
    constraints = digest(
        {
            "guardrail": profile.guardrail.model_dump(),
            "options": options,
            "plugins": profile.plugins,
            "environment": dict(parent.tool_ctx.environment),
            "binding": getattr(parent.tool_ctx.environment_binding, "binding_id", None),
        }
    )
    if recorded:
        effective &= recorded.tools
        high &= effective
        high |= effective & recorded.high_risk
        if recorded.workspace != mode or recorded.constraints_digest != constraints:
            raise ConfigError(
                "worker sandbox, environment or workspace changed; start a new run"
            )
    result = Envelope(
        tools=effective,
        high_risk=high,
        workspace=mode,
        depth=depth,
        confirm="deny"
        if role.confirm == "deny" or (recorded and recorded.confirm == "deny")
        else "broker",
        constraints_digest=constraints,
    )
    if mode == "frozen" and (
        effecting(effective, high)
        or effective & (NETWORK | DELEGATION_TOOLS)
        or any(t.startswith("browser_") for t in effective)
    ):
        raise ConfigError("frozen roles cannot write, delegate or access the network")
    return result
