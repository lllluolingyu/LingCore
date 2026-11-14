"""Cache identity uses the same context assembly as the actual worker."""

from __future__ import annotations

from typing import Any

from lingcore.composer import read_project_instructions
from lingcore.delegation.worker import inherited_context, overlay
from lingcore.plugins.loader import load_plugins


def worker_context(
    supervisor: Any, handle: Any, definition: Any, envelope: Any
) -> tuple[list[str], bool]:
    profile = supervisor.profile
    workspace = (
        supervisor.workspace.path(handle.record.id, "input")
        if envelope.workspace == "frozen"
        else supervisor.parent.tool_ctx.workspace
    )
    child = overlay(profile, definition, envelope, workspace)
    layers = list(getattr(child, "_worker_layers"))
    layers.extend(inherited_context(supervisor.parent, definition))
    loaded = load_plugins(child)
    layers.extend(loaded.prompt_layers)
    layers.extend(loaded.skills[name].instructions for name in child.skills)
    if "project_instructions" in definition.context:
        layers.append(
            read_project_instructions(workspace, profile.persona.project_instructions)
            or ""
        )
    # Hooks can consult undeclared external state. Merely enabling a workflows,
    # commands, or static-prompt plugin does not disable builtin worker reuse.
    return layers, not loaded.hook_factories
