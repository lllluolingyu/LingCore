"""Strict, secret-safe configuration for the first-party Telegram channel."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from lingcore.config import AgentProfile, _expand
from lingcore.errors import ConfigError

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WEBHOOK_SECRET = re.compile(r"[A-Za-z0-9_-]{1,256}")
_PACKAGE_DIR = Path(__file__).resolve().parents[2]


def _environment_name(value: str, *, field: str) -> str:
    name = value.strip()
    if not _ENV_NAME.fullmatch(name):
        raise ValueError(f"{field} must name an environment variable")
    return name


class TelegramWebhookConfig(BaseModel):
    """Webhook listener settings.

    TLS is deliberately terminated by a reverse proxy. ``public_url`` is the
    externally reachable HTTPS URL while ``listen``/``port`` describe the
    local HTTP listener.
    """

    model_config = ConfigDict(extra="forbid")

    public_url: str | None = None
    listen: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65_535)
    secret_token_env: str | None = None

    @field_validator("secret_token_env")
    @classmethod
    def _valid_secret_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _environment_name(value, field="webhook.secret_token_env")

    @field_validator("listen")
    @classmethod
    def _nonempty_listener(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("webhook.listen must not be empty")
        return value.strip()


class TelegramConfig(BaseModel):
    """Validated Telegram channel configuration.

    Secret values are never model fields. The private profile environment is
    retained only so named values can be resolved with the same profile-first
    precedence used by :class:`~lingcore.config.AgentProfile`.
    """

    model_config = ConfigDict(extra="forbid")

    token_env: str
    allowed_user_ids: list[Annotated[int, Field(strict=True, gt=0)]]
    mode: Literal["polling", "webhook"] = "polling"
    state_dir: str = ".lingcore/telegram"
    allow_absolute_state_dir: bool = False
    stream_edit_interval: float = Field(default=0.75, ge=0)
    confirmation_timeout: float = Field(default=60.0, gt=0)
    webhook: TelegramWebhookConfig = Field(default_factory=TelegramWebhookConfig)

    _profile_env: dict[str, str] = PrivateAttr(default_factory=dict)
    _profile_dir: Path | None = PrivateAttr(default=None)
    _state_path: Path | None = PrivateAttr(default=None)

    @field_validator("token_env")
    @classmethod
    def _valid_token_name(cls, value: str) -> str:
        return _environment_name(value, field="token_env")

    @field_validator("allowed_user_ids")
    @classmethod
    def _nonempty_deduplicated_users(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("allowed_user_ids must contain at least one user id")
        return list(dict.fromkeys(value))

    @field_validator("state_dir")
    @classmethod
    def _nonempty_state_dir(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("state_dir must not be empty")
        return value

    @model_validator(mode="after")
    def _webhook_shape(self) -> "TelegramConfig":
        if self.mode != "webhook":
            return self
        raw_url = self.webhook.public_url
        if raw_url is None or not raw_url.strip():
            raise ValueError("webhook mode requires webhook.public_url")
        parsed = urlsplit(raw_url)
        try:
            _parsed_port = parsed.port
        except ValueError:
            invalid_port = True
        else:
            invalid_port = False
        if (
            raw_url != raw_url.strip()
            or parsed.scheme.lower() != "https"
            or not parsed.netloc
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or invalid_port
        ):
            raise ValueError(
                "webhook.public_url must be an HTTPS URL without credentials, "
                "a query, or a fragment"
            )
        if self.webhook.secret_token_env is None:
            raise ValueError(
                "webhook mode requires webhook.secret_token_env to name a secret"
            )
        return self

    @property
    def state_path(self) -> Path:
        """Resolved and validated state root."""
        if self._state_path is None:
            raise ConfigError("Telegram config is not bound to a profile directory")
        return self._state_path

    @property
    def webhook_path(self) -> str:
        """Listener path derived from ``webhook.public_url`` (without a slash)."""
        if not self.webhook.public_url:
            return ""
        return urlsplit(self.webhook.public_url).path.lstrip("/")

    def resolve_token(
        self, environment: Mapping[str, str] | None = None
    ) -> str:
        return self._resolve_named_secret(
            self.token_env, "token_env", environment=environment
        )

    def resolve_webhook_secret(
        self, environment: Mapping[str, str] | None = None
    ) -> str:
        name = self.webhook.secret_token_env
        if name is None:
            raise ConfigError(
                "webhook mode requires webhook.secret_token_env to name a secret"
            )
        value = self._resolve_named_secret(
            name, "webhook.secret_token_env", environment=environment
        )
        if not _WEBHOOK_SECRET.fullmatch(value):
            raise ConfigError(
                "webhook.secret_token_env resolves to an invalid secret token; "
                "expected 1-256 characters from [A-Za-z0-9_-]"
            )
        return value

    def _resolve_named_secret(
        self,
        name: str,
        consumer: str,
        *,
        environment: Mapping[str, str] | None,
    ) -> str:
        if environment is not None:
            value = environment.get(name)
        elif name in self._profile_env:
            value = self._profile_env.get(name)
        else:
            value = os.environ.get(name)
        if not value:
            raise ConfigError(
                f"{consumer} names {name!r}, but that variable is not set or is "
                "empty in the profile .env or process environment"
            )
        return value


def _safe_validation_message(exc: ValidationError) -> str:
    """Format pydantic failures without ever echoing YAML input values."""
    parts: list[str] = []
    for error in exc.errors(
        include_url=False, include_context=False, include_input=False
    ):
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "validation failed"


def _reject_literal_secrets(raw: Mapping[object, object]) -> None:
    """Fail before pydantic can include a literal secret in an error repr."""
    forbidden = {"token", "bot_token", "secret", "secret_token", "webhook_secret"}
    if any(str(key).lower() in forbidden for key in raw):
        raise ConfigError(
            "Telegram secrets must be named through token_env and "
            "webhook.secret_token_env; literal secrets are not allowed"
        )
    webhook = raw.get("webhook")
    if isinstance(webhook, Mapping) and any(
        str(key).lower() in forbidden for key in webhook
    ):
        raise ConfigError(
            "Telegram secrets must be named through token_env and "
            "webhook.secret_token_env; literal secrets are not allowed"
        )


def _resolve_state_path(profile_dir: Path, config: TelegramConfig) -> Path:
    raw = Path(config.state_dir).expanduser()
    if raw.is_absolute():
        if not config.allow_absolute_state_dir:
            raise ConfigError(
                "Telegram state_dir is absolute; set "
                "allow_absolute_state_dir: true to permit this"
            )
        resolved = raw.resolve()
    else:
        resolved = (profile_dir / raw).resolve()
        if not resolved.is_relative_to(profile_dir.resolve()):
            raise ConfigError("Telegram state_dir escapes the profile directory")
    try:
        resolved.relative_to(_PACKAGE_DIR)
    except ValueError:
        return resolved
    raise ConfigError(
        "Telegram state_dir must not be inside the installed lingcore package tree"
    )


def _validate_profile_policy(profile: "AgentProfile") -> None:
    if not profile.sessions.enabled:
        raise ConfigError("Telegram requires sessions.enabled: true")
    if "run_shell" not in profile.tools:
        return
    raw_options = profile.tool_options.get("run_shell", {})
    if not isinstance(raw_options, Mapping):
        raise ConfigError("Telegram requires tool_options.run_shell to be a mapping")
    if not bool(raw_options.get("require_confirmation", True)):
        raise ConfigError(
            "Telegram refuses run_shell unless require_confirmation is true"
        )
    allow_patterns = raw_options.get("allow_patterns", [])
    if not isinstance(allow_patterns, list) or allow_patterns:
        raise ConfigError(
            "Telegram refuses run_shell unless allow_patterns is an empty list"
        )


def load_telegram_config(
    profile: AgentProfile | str | Path,
    path: str | Path | None = None,
    *,
    mode: Literal["polling", "webhook"] | None = None,
    require_secrets: bool = True,
) -> TelegramConfig:
    """Load and bind ``telegram.yaml`` for a selected agent profile.

    YAML expansion and named-secret lookup use the selected profile's exact
    ``.env`` mapping, not the Telegram file's directory or the caller's CWD.
    Set ``require_secrets=False`` only for offline diagnostics.
    """
    if not isinstance(profile, AgentProfile):
        profile = AgentProfile.load(profile)
    profile_dir = getattr(profile, "_source_dir", None)
    if profile_dir is None:
        raise ConfigError("Telegram requires a profile loaded from a file")
    config_path = Path(path) if path is not None else profile_dir / "telegram.yaml"
    if not config_path.is_file():
        raise ConfigError(f"Telegram config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text("utf-8")) or {}
    except yaml.YAMLError:
        raise ConfigError(f"invalid YAML in Telegram config {config_path}") from None
    except (OSError, UnicodeError) as exc:
        raise ConfigError(
            f"could not read Telegram config {config_path}: {exc}"
        ) from None
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Telegram config {config_path} must be a mapping at the top level"
        )
    _reject_literal_secrets(raw)

    profile_env: dict[str, str] = dict(getattr(profile, "_profile_env", {}))
    effective_env = {**os.environ, **profile_env}
    for name, value in profile_env.items():
        if not value:
            effective_env.pop(name, None)
    expanded = _expand(raw, effective_env)
    if mode is not None:
        expanded["mode"] = mode
    try:
        config = TelegramConfig.model_validate(expanded)
    except ValidationError as exc:
        raise ConfigError(
            f"invalid Telegram config {config_path}: "
            f"{_safe_validation_message(exc)}"
        ) from None

    _validate_profile_policy(profile)
    state_path = _resolve_state_path(profile_dir, config)
    config._profile_env = profile_env
    config._profile_dir = profile_dir.resolve()
    config._state_path = state_path
    if require_secrets:
        config.resolve_token()
        if config.mode == "webhook":
            config.resolve_webhook_secret()
    return config
