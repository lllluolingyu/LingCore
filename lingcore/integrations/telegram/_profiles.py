"""Per-user profile scoping for Telegram runtime assembly."""

from __future__ import annotations

from pathlib import Path

from lingcore.config import AgentProfile
from lingcore.integrations._profiles import scoped_profile as _scoped_profile
from lingcore.integrations.telegram.config import TelegramConfig


def scoped_profile(
    profile: AgentProfile, config: TelegramConfig, user_dir: Path
) -> AgentProfile:
    return _scoped_profile(profile, state_dir=config.state_dir, user_dir=user_dir)
