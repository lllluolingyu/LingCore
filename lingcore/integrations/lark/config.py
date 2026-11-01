"""Profile-bound configuration for the Lark and Feishu WebSocket channel."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
)

from lingcore.config import AgentProfile, _expand
from lingcore.errors import ConfigError


class LarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_id_env: str = "LARK_APP_ID"
    app_secret_env: str = "LARK_APP_SECRET"
    domain: Literal["feishu", "lark"] = "feishu"
    allowed_user_ids: list[str] = Field(min_length=1)
    allowed_chat_ids: list[str] = Field(default_factory=list)
    state_dir: str = ".lingcore/lark"
    allow_absolute_state_dir: bool = False
    confirmation_timeout: float = Field(default=60, gt=0, allow_inf_nan=False)

    _profile_env: dict[str, str] = PrivateAttr(default_factory=dict)
    _state_path: Path | None = PrivateAttr(default=None)

    @field_validator("app_id_env", "app_secret_env")
    @classmethod
    def _env_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("must name an environment variable")
        return value

    @field_validator("allowed_user_ids", "allowed_chat_ids")
    @classmethod
    def _ids(cls, value: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", item) for item in value):
            raise ValueError(
                "ids must contain only letters, digits, underscores or dashes"
            )
        return list(dict.fromkeys(value))

    @field_validator("state_dir")
    @classmethod
    def _state_dir(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("state_dir must not be empty")
        return value

    @property
    def api_domain(self) -> str:
        return (
            "https://open.feishu.cn"
            if self.domain == "feishu"
            else "https://open.larksuite.com"
        )

    @property
    def state_path(self) -> Path:
        if self._state_path is None:
            raise ConfigError("Lark config is not bound to a profile directory")
        return self._state_path

    def resolve_credentials(
        self, environment: Mapping[str, str] | None = None
    ) -> tuple[str, str]:
        values = []
        for name in (self.app_id_env, self.app_secret_env):
            if environment is not None:
                value = environment.get(name)
            elif name in self._profile_env:
                value = self._profile_env[name]
            else:
                value = os.environ.get(name)
            if not value or not value.strip():
                raise ConfigError(
                    f"Lark variable {name!r} is not set or is empty in the profile "
                    ".env or process environment"
                )
            values.append(value)
        return values[0], values[1]


def _validate_profile(profile: AgentProfile) -> None:
    if not profile.sessions.enabled:
        raise ConfigError("Lark requires sessions.enabled: true")
    if {"process_start", "process_monitor"} & set(profile.tools):
        raise ConfigError(
            "Lark refuses process_start/process_monitor; use an attended frontend"
        )
    if "run_shell" in profile.tools:
        options = profile.tool_options.get("run_shell", {})
        if not isinstance(options, Mapping) or (
            options.get("require_confirmation", True) is not True
            or options.get("allow_patterns", []) != []
        ):
            raise ConfigError(
                "Lark requires run_shell require_confirmation: true and allow_patterns: []"
            )


def load_lark_config(
    profile: AgentProfile | str | Path,
    path: str | Path | None = None,
    *,
    require_secrets: bool = True,
) -> LarkConfig:
    if not isinstance(profile, AgentProfile):
        profile = AgentProfile.load(profile)
    root = getattr(profile, "_source_dir", None)
    if root is None:
        raise ConfigError("Lark requires a profile loaded from a file")
    root = root.resolve()
    config_path = Path(path) if path is not None else root / "lark.yaml"
    try:
        raw = yaml.safe_load(config_path.read_text("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ConfigError(f"cannot read Lark YAML config: {config_path}") from None
    if not isinstance(raw, dict):
        raise ConfigError("Lark config must be a mapping")
    if {"app_id", "app_secret", "token", "secret"} & raw.keys():
        raise ConfigError(
            "Lark credentials must use app_id_env and app_secret_env, not literal values"
        )
    profile_env = dict(getattr(profile, "_profile_env", {}))
    environment = {**os.environ, **profile_env}
    environment = {key: value for key, value in environment.items() if value}
    try:
        config = LarkConfig.model_validate(_expand(raw, environment))
    except ValidationError as exc:
        errors = exc.errors(
            include_url=False, include_context=False, include_input=False
        )
        detail = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors
        )
        raise ConfigError(f"invalid Lark config: {detail}") from None
    _validate_profile(profile)
    state = Path(config.state_dir).expanduser()
    if state.is_absolute() and not config.allow_absolute_state_dir:
        raise ConfigError(
            "absolute Lark state_dir requires allow_absolute_state_dir: true"
        )
    resolved = (root / state).resolve()
    if not state.is_absolute() and not resolved.is_relative_to(root):
        raise ConfigError("Lark state_dir escapes the profile directory")
    if resolved.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ConfigError(
            "Lark state_dir must not be inside the installed lingcore package tree"
        )
    config._state_path = resolved
    config._profile_env = profile_env
    if require_secrets:
        config.resolve_credentials()
    return config
