"""LingCore — a lightweight, config-driven async agent framework."""

from lingcore.errors import (
    ConfigError,
    LingCoreError,
    MaxIterationsError,
    ToolError,
)
from lingcore.message import Attachment, Conversation, Message, ToolCall, ToolResult, UserInput

# Single source of truth for the package version: pyproject.toml declares
# ``dynamic = ["version"]`` and hatchling reads it from here at build time, so
# wheel metadata can never drift from what the code (and the HTTP user-agent
# strings derived from this) reports.
__version__ = "0.1.0"

__all__ = [
    "Attachment",
    "ConfigError",
    "Conversation",
    "LingCoreError",
    "MaxIterationsError",
    "Message",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "UserInput",
    "__version__",
]
