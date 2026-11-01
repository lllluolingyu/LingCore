"""Lark/Feishu channel; importing configuration never requires the SDK."""

from __future__ import annotations

from lingcore.integrations.lark.config import LarkConfig, load_lark_config

__all__ = ["LarkConfig", "load_lark_config"]
