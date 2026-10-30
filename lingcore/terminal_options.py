"""Profile and binding share one terminal option contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TerminalOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    timeout: float = Field(default=120, gt=0)
    max_timeout: float = Field(default=900, gt=0)
    capture_bytes: int = Field(default=1048576, gt=0, le=2 * 1024 * 1024)
    display_bytes: int = Field(default=16384, gt=0, le=2 * 1024 * 1024)
