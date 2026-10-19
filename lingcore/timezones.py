"""Timezone resolution shared by the bundled clock and calendar plugins.

A configured IANA name (``Asia/Shanghai``) is used as given. Without one the
host's zone is detected from ``TZ``, the ``/etc/localtime`` link or
``/etc/timezone``; a host whose zone has no IANA name (Windows without tzdata,
for example) falls back to its current fixed UTC offset, which cannot follow
daylight-saving changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


@dataclass(frozen=True, slots=True)
class Zone:
    """A resolved timezone. ``iana`` is false for a fixed-offset fallback."""

    tzinfo: tzinfo
    name: str
    iana: bool


def format_offset(offset: timedelta | None) -> str:
    """``UTC+08:00`` style, with seconds only when the offset has them."""
    seconds = int((offset or timedelta()).total_seconds())
    sign = "-" if seconds < 0 else "+"
    hours, rest = divmod(abs(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    text = f"UTC{sign}{hours:02d}:{minutes:02d}"
    return text + (f":{secs:02d}" if secs else "")


def _zoneinfo(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def _host_zone_name() -> str | None:
    tz = os.environ.get("TZ", "").strip().lstrip(":")
    if tz and _zoneinfo(tz) is not None:
        return tz
    try:
        target = os.readlink("/etc/localtime")
    except OSError:
        target = ""
    if "zoneinfo/" in target:
        name = target.rsplit("zoneinfo/", 1)[1]
        if _zoneinfo(name) is not None:
            return name
    try:
        name = Path("/etc/timezone").read_text("utf-8").strip()
    except (OSError, UnicodeError):
        name = ""
    if name and _zoneinfo(name) is not None:
        return name
    return None


def resolve_zone(name: str | None) -> Zone:
    """The configured zone, else the host's. ``ValueError`` names a bad one.

    A blank name (an unset ``${VAR:-}`` expansion) means "not configured".
    """
    configured = (name or "").strip()
    if configured:
        zone = _zoneinfo(configured)
        if zone is None:
            raise ValueError(
                f"unknown timezone {configured!r}: use an IANA name such as "
                "'Asia/Shanghai' or 'Europe/Berlin' (on Windows, install the "
                "tzdata package)"
            )
        return Zone(zone, configured, True)
    host = _host_zone_name()
    if host is not None:
        zone = _zoneinfo(host)
        assert zone is not None
        return Zone(zone, host, True)
    local = datetime.now().astimezone().tzinfo
    assert local is not None
    return Zone(local, format_offset(local.utcoffset(None)), False)
