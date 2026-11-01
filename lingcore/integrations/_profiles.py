"""Per-conversation profile scoping shared by chat channels."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError

_PER_USER_FILES = (
    ("memory", "memory.md", frozenset({"memory"})),
    (
        "calendar",
        "calendar.ics",
        frozenset(
            {"calendar_add", "calendar_update", "calendar_list", "calendar_remove"}
        ),
    ),
)


def scoped_profile(
    profile: AgentProfile, *, state_dir: str, user_dir: Path
) -> AgentProfile:
    scoped = profile.model_copy(deep=True)
    # Be explicit even though pydantic's deep copy currently handles private
    # mappings: future model changes must not alias mutable tenant options.
    scoped.tool_options = copy.deepcopy(scoped.tool_options)
    workspace = user_dir / "workspace"
    absolute_state = Path(state_dir).expanduser().is_absolute()
    source_dir = getattr(scoped, "_source_dir", None)
    if not absolute_state and source_dir is None:
        # guarded by channel config loading
        raise ConfigError("Channel scoped profile has no source directory")

    def configured(name: str) -> str:
        path = user_dir / name
        if absolute_state:
            return str(path)
        assert source_dir is not None
        return str(path.relative_to(source_dir.resolve()))

    configured_sessions_path = configured("sessions.db")
    scoped.workspace = str(workspace)
    session_updates: dict[str, Any] = {
        "enabled": True,
        "path": configured_sessions_path,
    }
    if absolute_state:
        # The state root already passed its single operator-consent gate;
        # all derived absolute files inherit that consent.
        session_updates["allow_absolute_path"] = True
    scoped.sessions = scoped.sessions.model_copy(update=session_updates)
    if scoped.tracing.enabled:
        scoped.tracing = scoped.tracing.model_copy(
            update={
                "path": configured("traces"),
                "allow_absolute_path": absolute_state,
            }
        )
    # Profile-scoped files each user writes get a per-user path, like the
    # sessions database: the memory file and the calendar.
    for key, filename, tools in _PER_USER_FILES:
        if not tools & set(scoped.tools):
            continue
        raw_options = scoped.tool_options.get(key, {})
        if not isinstance(raw_options, Mapping):
            raise ConfigError(f"tool_options.{key} must be a mapping")
        options = copy.deepcopy(dict(raw_options))
        options["path"] = configured(filename)
        if absolute_state:
            options["allow_absolute_path"] = True
        scoped.tool_options[key] = options

    return scoped
