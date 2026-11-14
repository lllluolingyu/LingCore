"""Profile-local role overlays, loaded without executing code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import Field

from lingcore.delegation.config import StrictModel
from lingcore.delegation.models import WorkspaceMode
from lingcore.errors import ConfigError
from lingcore.paths import confined_directory

if TYPE_CHECKING:
    from lingcore.config import AgentProfile

DELEGATION_TOOLS = frozenset(
    {
        "delegate",
        "workflow_run",
        "workflow_resume",
        "workflow_status",
        "workflow_result",
        "workflow_stop",
    }
)
STATE_WRITERS = frozenset(
    {"memory", "world", "calendar_add", "calendar_update", "calendar_remove"}
)


class Role(StrictModel):
    description: str = Field(default="", max_length=1000)
    persona: str = Field(default="", max_length=30000)
    model: str | None = None
    tools: list[str] = Field(default_factory=list)
    workspace: WorkspaceMode = "shared"
    context: list[Literal["project_instructions", "memory", "world"]] = Field(
        default=["project_instructions"]
    )
    skills: list[str] = Field(default_factory=list)
    max_iters: int = Field(default=40, ge=1, le=1000)
    memory: dict[str, int] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)
    confirm: Literal["broker", "deny"] = "broker"


def read_asset(root: Path, path: str, limit: int = 131072) -> str:
    rel = Path(path)
    with confined_directory(root, rel.parent) as directory:
        return directory.read_regular(rel.name, max_bytes=limit).decode("utf-8")


def load_roles(profile: AgentProfile) -> dict[str, Role]:
    roles = {
        "general": Role(
            description="General-purpose worker",
            tools=[
                t for t in profile.tools if t not in DELEGATION_TOOLS | STATE_WRITERS
            ],
            max_iters=profile.loop.max_iters,
        )
    }
    root = getattr(profile, "_source_dir", None)
    if root is not None:
        directory = Path(root) / profile.delegation.roles_dir
        if directory.exists():
            if directory.is_symlink():
                raise ConfigError("roles directory cannot be a symlink")
            for path in sorted(directory.glob("*.yaml")):
                if not path.stem.replace("-", "_").isidentifier():
                    raise ConfigError("role names must be identifiers")
                try:
                    role = Role.model_validate(
                        yaml.safe_load(
                            read_asset(Path(root), str(path.relative_to(root)))
                        )
                    )
                    md = path.with_suffix(".md")
                    if md.exists():
                        role.persona = read_asset(
                            Path(root), str(md.relative_to(root)), 30000
                        )
                    roles[path.stem] = role
                except Exception as exc:
                    raise ConfigError(f"invalid role {path.stem}: {exc}") from None
    for name, role in roles.items():
        if not set(role.tools) <= set(profile.tools):
            raise ConfigError(f"role {name} tools exceed the profile ceiling")
        if role.model is not None and role.model not in profile.delegation.models:
            raise ConfigError(f"role {name} names an unknown model alias")
        if set(role.memory) - {"max_tokens", "max_messages"}:
            raise ConfigError(f"role {name} has unsupported memory limits")
        if any(v <= 0 for v in role.memory.values()):
            raise ConfigError(f"role {name} memory limits must be positive")
        profile.delegation.limits.narrow(role.limits)
    if profile.delegation.default_role not in roles:
        raise ConfigError("delegation.default_role does not exist")
    return roles
