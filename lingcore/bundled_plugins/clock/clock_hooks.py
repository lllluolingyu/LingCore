"""The current date and time for the model, attached to each user message.

Models have no clock: without this they guess "today" from their training data.
``ClockHooks.user_message`` adds one line such as

    [clock] Now: Monday 2026-10-19 14:32 Asia/Shanghai (UTC+08:00)

as hook context, so it travels with the stored user message rather than the
system prompt, whose cached prefix stays byte-stable. Each message keeps the
time it was sent, which also lets the model see how much time passed between
messages. Options (``tool_options.clock``) are validated when the Agent is
built, so a bad timezone fails at launch, not on the first message.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, ValidationError

from lingcore.errors import ConfigError
from lingcore.plugins import (
    PluginContext,
    PluginHooks,
    UserMessageDecision,
    UserMessageEvent,
)
from lingcore.timezones import WEEKDAYS, Zone, format_offset, resolve_zone


class ClockOptions(BaseModel):
    """Validated ``tool_options.clock``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # IANA name; unset or blank uses the host's timezone.
    timezone: str | None = None


def parse_clock_options(raw: Mapping[str, object]) -> tuple[ClockOptions, Zone]:
    try:
        options = ClockOptions.model_validate(dict(raw))
    except ValidationError as exc:
        fields = ", ".join(
            ".".join(map(str, error["loc"])) + ": " + error["msg"]
            for error in exc.errors()
        )
        raise ConfigError(f"invalid tool_options.clock: {fields}") from None
    try:
        return options, resolve_zone(options.timezone)
    except ValueError as exc:
        raise ConfigError(f"invalid tool_options.clock: {exc}") from None


def clock_line(now: datetime, zone: Zone) -> str:
    local = now.astimezone(zone.tzinfo)
    offset = format_offset(local.utcoffset())
    label = f"{zone.name} ({offset})" if zone.iana else offset
    return f"[clock] Now: {WEEKDAYS[local.weekday()]} {local:%Y-%m-%d %H:%M} {label}"


class ClockHooks(PluginHooks):
    def __init__(
        self,
        ctx: PluginContext,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        super().__init__(ctx)
        _, self.zone = parse_clock_options(ctx.options)
        self._now = now

    async def user_message(self, event: UserMessageEvent) -> UserMessageDecision:
        return UserMessageDecision.add_context(clock_line(self._now(), self.zone))
