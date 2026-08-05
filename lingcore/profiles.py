"""Immutable profile templates and writable user-profile initialization."""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from lingcore.errors import ConfigError

_REPO_PROFILE_ROOT = Path(__file__).resolve().parents[1] / "profiles"
_PACKAGED_TEMPLATE_ROOT = Path(__file__).resolve().parent / "profile_templates"
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Keep this manifest explicit: profile workspaces, sessions, memories, secrets,
# and other ignored local artifacts must never leak into a wheel template.
PROFILE_TEMPLATE_FILES: dict[str, tuple[str, ...]] = {
    "coding": (
        ".env.example",
        "config.yaml",
        "role.md",
        "workflow.md",
        "world.md",
    ),
    "coding_ollama": ("config.yaml",),
    "daily": (
        ".env.example",
        "config.yaml",
        "role.md",
        "workflow.md",
        "world.md",
    ),
    "teaching": (
        ".env.example",
        "config.yaml",
        "role.md",
        "workflow.md",
        "world.md",
    ),
}


def state_home(environment: Mapping[str, str] | None = None) -> Path:
    """Return LingCore's per-user writable application-state directory."""
    env = os.environ if environment is None else environment
    explicit = env.get("LINGCORE_STATE_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if sys.platform == "win32":
        base = env.get("LOCALAPPDATA")
        if base:
            return (Path(base).expanduser() / "LingCore").resolve()
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "LingCore").resolve()
    xdg = env.get("XDG_STATE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return (base / "lingcore").resolve()


def user_profiles_dir(environment: Mapping[str, str] | None = None) -> Path:
    return state_home(environment) / "profiles"


def default_profile_path(
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Use the checkout example when present, else the initialized user copy."""
    checkout = _REPO_PROFILE_ROOT / "coding"
    if (checkout / "config.yaml").is_file():
        return checkout
    return user_profiles_dir(environment) / "coding"


def _template_root() -> Path:
    if _PACKAGED_TEMPLATE_ROOT.is_dir():
        return _PACKAGED_TEMPLATE_ROOT
    if _REPO_PROFILE_ROOT.is_dir():
        return _REPO_PROFILE_ROOT
    raise ConfigError("LingCore profile templates are missing from this installation")


def initialize_profile(
    template: str = "coding",
    *,
    name: str | None = None,
    destination: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Copy one immutable template into a new writable profile directory.

    Existing destinations are never merged or overwritten. Copying through a
    sibling temporary directory makes a successfully returned profile complete,
    while failures leave no half-initialized destination behind.
    """
    files = PROFILE_TEMPLATE_FILES.get(template)
    if files is None:
        raise ConfigError(
            f"unknown profile template {template!r}; "
            f"available: {', '.join(PROFILE_TEMPLATE_FILES)}"
        )
    profile_name = name or template
    if not _PROFILE_NAME.fullmatch(profile_name):
        raise ConfigError(
            "profile name must start with an ASCII letter or digit and contain "
            "only letters, digits, '.', '_', or '-'"
        )
    if destination is not None and name is not None:
        raise ConfigError(
            "pass either a profile name or an explicit destination, not both"
        )
    target = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else user_profiles_dir(environment) / profile_name
    )
    if target.exists():
        raise ConfigError(f"profile destination already exists: {target}")

    source = _template_root() / template
    missing = [relative for relative in files if not (source / relative).is_file()]
    if missing:
        raise ConfigError(
            f"profile template {template!r} is incomplete; missing: {missing}"
        )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            for relative in files:
                out = temporary / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, out)
            temporary.replace(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    except OSError as exc:
        raise ConfigError(f"could not initialize profile at {target}: {exc}") from None
    return target
