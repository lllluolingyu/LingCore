"""Validated browser launch, attachment, and settling options."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from lingcore.bundled_plugins.browser.proxy import _normal_host
from lingcore.errors import ConfigError
from lingcore.tool_options import (
    FETCH_OPTION_PATH,
    NETWORK_POLICY_KEYS,
    bool_option,
    int_option,
)

if TYPE_CHECKING:
    pass


OPTION_PATH = "tool_options.browser"
_OPTIONS = frozenset(
    {
        "headless",
        "timeout",
        "executable_path",
        "chromium_sandbox",
        "viewport_width",
        "viewport_height",
        "offload_over_chars",
        "max_chars",
        "cdp_url",
        "settle_quiet_ms",
        "settle_timeout_ms",
    }
)
_LAUNCH_ONLY = frozenset(
    {
        "headless",
        "executable_path",
        "chromium_sandbox",
        "viewport_width",
        "viewport_height",
    }
)
_INSTALL_HINT = (
    "install the browser extra and Chromium: "
    "pip install 'lingcore[browser]' && playwright install chromium"
)
_MIN_WIDTH, _MAX_WIDTH = 320, 3840
_MIN_HEIGHT, _MAX_HEIGHT = 240, 2160


@dataclass(frozen=True, slots=True)
class BrowserOptions:
    headless: bool = True
    timeout_ms: int = 15_000
    executable_path: str | None = None
    chromium_sandbox: bool = False
    viewport: tuple[int, int] = (1280, 800)
    offload_over_chars: int = 20_000
    max_chars: int = 40_000
    cdp_url: str | None = None
    settle_quiet_ms: int = 500
    settle_timeout_ms: int = 3_000


def _parse_cdp_url(raw: object) -> str:
    """Validate ``cdp_url`` as a credential-free loopback DevTools endpoint.

    Errors never echo the value: a ``ws://`` endpoint's path is the browser's
    session id, which grants full control of it.
    """
    error = ConfigError(
        f"{OPTION_PATH}.cdp_url must be an http(s) or ws(s) URL of a "
        "DevTools endpoint on loopback (localhost, 127.0.0.0/8 or ::1)"
    )
    if not isinstance(raw, str) or not raw:
        raise error
    try:
        parsed = urlparse(raw)
        parsed.port  # noqa: B018  (raises on an out-of-range port)
    except ValueError:
        raise error from None
    host = parsed.hostname
    if (
        parsed.scheme not in ("http", "https", "ws", "wss")
        or parsed.username is not None
        or parsed.password is not None
        or not host
    ):
        raise error
    if _normal_host(host) != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise error
        except ValueError:
            raise error from None
    return raw


def parse_browser_options(raw: object) -> BrowserOptions:
    """Validate the complete ``tool_options.browser`` mapping."""
    if raw is None:
        return BrowserOptions()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{OPTION_PATH} must be a mapping")
    moved = sorted(str(name) for name in raw if name in NETWORK_POLICY_KEYS)
    if moved:
        raise ConfigError(
            f"{OPTION_PATH}.{moved[0]} is not supported: the network policy is "
            f"shared with fetch_url, so set it under {FETCH_OPTION_PATH}"
        )
    unknown = sorted(str(name) for name in raw if name not in _OPTIONS)
    if unknown:
        raise ConfigError(f"unknown {OPTION_PATH} option(s): {', '.join(unknown)}")
    path = OPTION_PATH
    cdp_url = None
    if "cdp_url" in raw:
        cdp_url = _parse_cdp_url(raw["cdp_url"])
        launch_only = sorted(str(name) for name in raw if name in _LAUNCH_ONLY)
        if launch_only:
            raise ConfigError(
                f"{path}.{launch_only[0]} cannot be combined with {path}.cdp_url: "
                "it configures a browser LingCore launches, not one it attaches to"
            )
    executable = raw.get("executable_path")
    if executable is not None and (not isinstance(executable, str) or not executable):
        raise ConfigError(f"{path}.executable_path must be a non-empty string")
    defaults = BrowserOptions()
    quiet_ms = int_option(
        raw, "settle_quiet_ms", 500, minimum=0, maximum=5_000, option_path=path
    )
    settle_ms = int_option(
        raw, "settle_timeout_ms", 3_000, minimum=1, maximum=10_000, option_path=path
    )
    if quiet_ms > settle_ms:
        raise ConfigError(f"{path}.settle_quiet_ms must be <= settle_timeout_ms")
    return BrowserOptions(
        headless=bool_option(raw, "headless", True, option_path=path),
        timeout_ms=int_option(
            raw, "timeout", 15, minimum=1, maximum=120, option_path=path
        )
        * 1000,
        executable_path=executable,
        chromium_sandbox=bool_option(raw, "chromium_sandbox", False, option_path=path),
        viewport=(
            int_option(
                raw,
                "viewport_width",
                1280,
                minimum=_MIN_WIDTH,
                maximum=_MAX_WIDTH,
                option_path=path,
            ),
            int_option(
                raw,
                "viewport_height",
                800,
                minimum=_MIN_HEIGHT,
                maximum=_MAX_HEIGHT,
                option_path=path,
            ),
        ),
        offload_over_chars=int_option(
            raw, "offload_over_chars", defaults.offload_over_chars, option_path=path
        ),
        max_chars=int_option(
            raw, "max_chars", defaults.max_chars, minimum=1_000, option_path=path
        ),
        cdp_url=cdp_url,
        settle_quiet_ms=quiet_ms,
        settle_timeout_ms=settle_ms,
    )
