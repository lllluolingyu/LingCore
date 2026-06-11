"""LingCore — a lightweight, config-driven async agent framework."""

from lingcore.errors import (
    ConfigError,
    LingCoreError,
    MaxIterationsError,
    ToolError,
)
from lingcore.message import Attachment, Conversation, Message, ToolCall, ToolResult, UserInput

__version__ = "0.0.1"

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
