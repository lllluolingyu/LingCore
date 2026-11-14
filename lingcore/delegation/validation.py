"""Offline validation shared by profile loading and doctor."""

from __future__ import annotations

from typing import Any

from lingcore.delegation.envelope import NETWORK, effecting
from lingcore.delegation.models import Envelope
from lingcore.delegation.roles import DELEGATION_TOOLS, load_roles
from lingcore.delegation.schema import compile_schema
from lingcore.errors import ConfigError
from lingcore.workflows.catalog import load_catalog
from lingcore.workflows.spec import ValidatedSpec


def validate_definitions(profile: Any) -> tuple[int, dict[str, int]]:
    roles = load_roles(profile)
    for schema in profile.delegation.schemas.values():
        compile_schema(schema)
    for role in roles.values():
        if role.workspace == "ticket":
            raise ConfigError("ticket workspace is reserved for the superagent host")
        if role.workspace == "frozen" and (
            effecting(frozenset(role.tools), frozenset())
            or set(role.tools) & (NETWORK | DELEGATION_TOOLS)
        ):
            raise ConfigError("frozen role contains effecting or network tools")
    catalog = load_catalog(profile)

    def envelope(name: str, **options: Any) -> Envelope:
        if name not in roles:
            raise ConfigError("workflow names an unknown role")
        role = roles[name]
        tools = frozenset(
            role.tools if options.get("tools") is None else options["tools"]
        )
        if not tools <= set(role.tools):
            raise ConfigError("workflow tools exceed role")
        mode = options.get("workspace") or role.workspace
        if mode != role.workspace and not (
            role.workspace == "shared" and mode in {"frozen", "scratch"}
        ):
            raise ConfigError("workflow workspace widens role")
        return Envelope(tools=tools, workspace=mode)

    bounds = {
        name: ValidatedSpec(
            spec,
            profile.delegation.limits,
            envelope,
            catalog,
            models=frozenset(profile.delegation.models),
        ).bound
        for name, spec in catalog.items()
    }
    if profile.superagent:
        from lingcore.superagent.runtime import validate_profile

        validate_profile(profile)
    return len(roles), bounds
