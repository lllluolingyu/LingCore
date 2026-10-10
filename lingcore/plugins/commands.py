"""Data-only slash command loading and ambiguity-safe resolution."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from lingcore.errors import ConfigError
from lingcore.message import UserInput
from lingcore.paths import PathEscapeError, confined_directory

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9-]{0,39}\Z")
_TELEGRAM = re.compile(r"[a-z0-9_]{1,32}\Z")
_MAX_BYTES = 128_000


@dataclass(frozen=True)
class Command:
    name: str
    body: str
    namespace: str | None = None
    description: str = ""
    argument_hint: str = ""

    @property
    def qualified_name(self) -> str:
        return f"/{self.namespace}:{self.name}" if self.namespace else f"/{self.name}"

    @property
    def telegram_name(self) -> str:
        return (
            f"{self.namespace}_{self.name}" if self.namespace else self.name
        ).replace("-", "_")

    def metadata(self) -> dict[str, str]:
        return {
            "name": self.qualified_name,
            "description": self.description,
            "argument_hint": self.argument_hint,
            "telegram_name": self.telegram_name,
        }


class CommandCatalog:
    def __init__(self, commands: Iterable[Command] = ()) -> None:
        self._commands: dict[str, Command] = {}
        for command in commands:
            self.add(command)

    @property
    def commands(self) -> tuple[Command, ...]:
        return tuple(self._commands.values())

    def add(self, command: Command) -> None:
        if not _NAME.fullmatch(command.name) or (
            command.namespace is not None
            and not _NAMESPACE.fullmatch(command.namespace)
        ):
            raise ConfigError("invalid plugin command name")
        if command.qualified_name in self._commands:
            raise ConfigError(f"duplicate command {command.qualified_name}")
        self._commands[command.qualified_name] = command

    def extend(self, other: CommandCatalog) -> None:
        duplicates = self._commands.keys() & other._commands.keys()
        if duplicates:
            raise ConfigError(f"duplicate command {sorted(duplicates)[0]}")
        self._commands.update(other._commands)

    def metadata(self) -> list[dict[str, str]]:
        return [command.metadata() for command in self.commands]

    def telegram_commands(self, *, reserved: Iterable[str] = ()) -> tuple[Command, ...]:
        blocked = {name.lstrip("/") for name in reserved}
        counts: dict[str, int] = {}
        for command in self.commands:
            counts[command.telegram_name] = counts.get(command.telegram_name, 0) + 1
        return tuple(
            command
            for command in self.commands
            if _TELEGRAM.fullmatch(command.telegram_name)
            and command.telegram_name not in blocked
            and counts[command.telegram_name] == 1
            and self.resolve("/" + command.telegram_name, reserved=blocked) is not None
        )

    def resolve(self, raw: str, *, reserved: Iterable[str]) -> UserInput | None:
        match = re.match(r"^/([^\s]+)(?:\s+([\s\S]*))?$", raw.lstrip())
        if match is None:
            return None
        # Command names are lowercase by construction; Telegram clients and
        # users may still type /Review, so matching is case-insensitive.
        name = match[1].split("@", 1)[0].lower()
        blocked = {item.lstrip("/").lower() for item in reserved}
        if name in blocked:
            return None
        # Qualified names cannot collide with frontend commands. Everything else
        # has one combined candidate set, including Telegram's lossy aliases.
        if ":" in name:
            command = self._commands.get("/" + name)
        else:
            candidates = [
                command
                for command in self.commands
                if command.name == name
                or (command.telegram_name == name and _TELEGRAM.fullmatch(name))
            ]
            command = candidates[0] if len(candidates) == 1 else None
        if command is None:
            return None
        return UserInput(
            text=command.body.replace("$ARGUMENTS", match[2] or ""), display_text=raw
        )


def load_commands(root: Path, namespace: str | None = None) -> CommandCatalog:
    """Read ``*.md`` in a commands directory without importing plugin code."""
    root = Path(root)
    catalog = CommandCatalog()
    if not root.exists():
        return catalog
    try:
        with confined_directory(root.parent, root.name) as directory:
            for name, _, is_file, is_symlink in sorted(directory.iter_entries()):
                if not name.endswith(".md"):
                    continue
                if is_symlink or not is_file:
                    raise ConfigError(f"command {name!r} must be a regular file")
                with directory.open_regular(name) as handle:
                    payload = handle.read(_MAX_BYTES + 1)
                if len(payload) > _MAX_BYTES:
                    raise ConfigError(f"command {name!r} is too large")
                content = payload.decode("utf-8").replace("\r\n", "\n")
                metadata: dict[str, str] = {}
                body = content
                if content.startswith("---\n"):
                    parts = content.split("\n---", 1)
                    if len(parts) != 2 or (parts[1] and not parts[1].startswith("\n")):
                        raise ConfigError(f"invalid command frontmatter in {name!r}")
                    parsed = yaml.safe_load(parts[0][4:]) or {}
                    if (
                        not isinstance(parsed, dict)
                        or set(parsed) - {"description", "argument_hint"}
                        or any(not isinstance(value, str) for value in parsed.values())
                    ):
                        raise ConfigError(f"invalid command metadata in {name!r}")
                    metadata = parsed
                    body = parts[1].removeprefix("\n")
                catalog.add(Command(Path(name).stem, body, namespace, **metadata))
    except (OSError, UnicodeError, yaml.YAMLError, PathEscapeError) as exc:
        raise ConfigError(
            f"cannot load commands from {root.name!r}: {type(exc).__name__}"
        ) from None
    return catalog


def discover_commands(profile: object) -> CommandCatalog:
    """Discover explicitly enabled commands without loading tool/hook modules."""
    from lingcore.plugins.discovery import discover_plugins, require_healthy
    from lingcore.plugins.manifest import component_path

    source_dir = getattr(profile, "_source_dir", None)
    catalog = CommandCatalog()
    problems: dict[str, str] = {}
    found = discover_plugins(source_dir, problems=problems)
    require_healthy(getattr(profile, "plugins", ()), problems)
    for name in getattr(profile, "plugins", ()):
        if name not in found:
            raise ConfigError(f"unknown enabled plugin {name!r}")
        plugin = found[name]
        if plugin.manifest.commands is not None:
            catalog.extend(
                load_commands(
                    component_path(plugin.root, plugin.manifest.commands), name
                )
            )
    if source_dir is not None:
        catalog.extend(load_commands(source_dir / "commands"))
    return catalog
