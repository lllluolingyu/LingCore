"""Official Telegram Bot API channel for LingCore.

This module itself is safe to import without the ``telegram`` extra. PTB is
loaded only when :func:`create_telegram_application` is called.
"""

from __future__ import annotations

from typing import Any

from lingcore.integrations.telegram.config import (
    TelegramConfig,
    load_telegram_config,
)


def create_telegram_application(*args: Any, **kwargs: Any) -> Any:
    """Build the PTB application, importing the optional dependency lazily."""
    from lingcore.integrations.telegram.application import (
        create_telegram_application as _create,
    )

    return _create(*args, **kwargs)


__all__ = [
    "TelegramConfig",
    "create_telegram_application",
    "load_telegram_config",
]
