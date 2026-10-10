"""Strict, data-only plugin manifests and confined component paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from lingcore.errors import ConfigError
from lingcore.paths import PathEscapeError, resolve_confined

PLUGIN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}\Z")
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def version_tuple(version: str) -> tuple[int, int, int]:
    if not _VERSION.fullmatch(version):
        raise ValueError(
            "version must be a numeric semantic version (major.minor.patch)"
        )
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def relative_component(value: str) -> str:
    if (
        not value
        or "\\" in value
        or ":" in value
        or Path(value).is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("plugin components must use safe relative paths")
    return value


def component_path(root: Path, value: str) -> Path:
    try:
        relative_component(value)
        return resolve_confined(root, value)
    except (ValueError, PathEscapeError) as exc:
        raise ConfigError(f"unsafe plugin component path: {value!r}") from exc


class EnvironmentName(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    required: bool = True


class EnvironmentOption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    option: str = Field(min_length=1)
    default: str | None = None
    required: bool = True

    @field_validator("default")
    @classmethod
    def _env_default(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_NAME.fullmatch(value):
            raise ValueError("environment default must name a variable")
        return value


class ExecutableRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    option: str | None = None


class OptionRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(min_length=1)
    hint: str | None = None


class ModuleRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    hint: str | None = None


class PluginRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executables: list[ExecutableRequirement] = Field(default_factory=list)
    options: list[OptionRequirement] = Field(default_factory=list)
    modules: list[ModuleRequirement] = Field(default_factory=list)


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    api: Literal[1] = 1
    min_lingcore: str | None = None
    description: str = ""
    module: str | None = None
    provides: list[str] = Field(default_factory=list)
    hooks: str | None = None
    skills: str | None = "skills"
    commands: str | None = "commands"
    prompt: str | None = None
    options_key: str | None = None
    on_hook_error: Literal["block", "ignore"] = "block"
    hook_timeout: float = Field(default=10, gt=0, le=60)
    environment: list[EnvironmentName | EnvironmentOption] = Field(default_factory=list)
    requires: PluginRequirements = Field(default_factory=PluginRequirements)

    @property
    def prefix(self) -> str:
        return self.name.replace("-", "_")

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not PLUGIN_NAME.fullmatch(value):
            raise ValueError("invalid plugin name")
        return value

    @field_validator("version", "min_lingcore")
    @classmethod
    def _version(cls, value: str | None) -> str | None:
        if value is not None:
            version_tuple(value)
        return value

    @field_validator("module", "skills", "commands", "prompt")
    @classmethod
    def _path(cls, value: str | None) -> str | None:
        return relative_component(value) if value is not None else None

    @field_validator("hooks")
    @classmethod
    def _hooks(cls, value: str | None) -> str | None:
        if value is not None and (not value.isidentifier() or value.startswith("_")):
            raise ValueError("hooks must name a public attribute in the plugin module")
        return value

    @model_validator(mode="after")
    def _contract(self) -> PluginManifest:
        if len(self.provides) != len(set(self.provides)):
            raise ValueError("provides cannot contain duplicate tools")
        for name in self.provides:
            if not re.fullmatch(r"[a-z0-9_]+", name) or not (
                name == self.prefix or name.startswith(self.prefix + "_")
            ):
                raise ValueError(
                    f"tool {name!r} must use plugin prefix {self.prefix!r}"
                )
        if self.hooks is not None and self.module is None:
            raise ValueError("hooks requires a module")
        if self.options_key is None:
            self.options_key = self.prefix
        elif not self.options_key.strip():
            raise ValueError("options_key cannot be empty")
        return self

    def check_compatibility(self) -> None:
        from lingcore import __version__

        if self.min_lingcore is not None and version_tuple(__version__) < version_tuple(
            self.min_lingcore
        ):
            raise ConfigError(
                f"plugin {self.name!r} requires LingCore >= {self.min_lingcore}"
            )


def read_manifest(root: Path) -> PluginManifest:
    path = component_path(root, "plugin.yaml")
    try:
        raw = yaml.safe_load(path.read_text("utf-8"))
        manifest = PluginManifest.model_validate(raw)
        for value in (
            manifest.module,
            manifest.skills,
            manifest.commands,
            manifest.prompt,
        ):
            if value is not None:
                component_path(root, value)
        return manifest
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        # Validation errors can carry input values, including arbitrary options.
        # Report locations and types only, never echo manifest values.
        if isinstance(exc, ValidationError):
            details = ", ".join(
                ".".join(str(part) for part in error["loc"]) + ": " + error["type"]
                for error in exc.errors()
            )
        else:
            details = type(exc).__name__
        raise ConfigError(f"invalid plugin manifest {path}: {details}") from None
