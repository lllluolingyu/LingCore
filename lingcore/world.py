"""Private runtime observations, separate from authored profile instructions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from lingcore.errors import ConfigError, ToolError
from lingcore.paths import ConfinedDirectory, PathEscapeError, confined_directory

WORLD_STATE_KEY = "_world_file"
WORLD_MAX_BYTES = 65_536
SNAPSHOT_MAX_BYTES = 16_384
_START = "<!-- lingcore:environment:start -->"
_END = "<!-- lingcore:environment:end -->"
_PACKAGE_DIR = Path(__file__).resolve().parent
WORLD_DEFAULT_PATH = ".lingcore/world.md"


class WorldOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = WORLD_DEFAULT_PATH
    allow_absolute_path: bool = False


def build_world_file(directory: Path, options: Any, *, fallback: str = "") -> WorldFile:
    cfg = WorldOptions.model_validate(options)
    raw = Path(cfg.path)
    if not cfg.path.strip() or raw.name in {"", ".", ".."} or ".." in raw.parts:
        raise ConfigError("tool_options.world.path must name a confined runtime file")
    if raw.is_absolute() and not cfg.allow_absolute_path:
        raise ConfigError("absolute world path requires allow_absolute_path")
    full = raw if raw.is_absolute() else directory / raw
    return WorldFile(full.parent, fallback=fallback, name=full.name)


@dataclass(frozen=True)
class WorldFile:
    directory: Path
    # Read-only authored world.md, loaded with the original profile semantics.
    fallback: str = ""
    name: str = "world.md"

    @property
    def path(self) -> Path:
        return self.directory / self.name

    def _read(self, directory: ConfinedDirectory) -> str:
        try:
            raw = directory.read_regular(self.name, max_bytes=WORLD_MAX_BYTES)
        except PathEscapeError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return ""
            raise
        return raw.decode("utf-8")

    def read_snapshot(self) -> str:
        root = self.directory.absolute()
        try:
            with confined_directory(
                Path(root.anchor), root.relative_to(root.anchor)
            ) as d:
                return self._read(d)
        except FileNotFoundError:
            return ""
        except (OSError, UnicodeError, PathEscapeError) as exc:
            raise ToolError(
                f"cannot safely read runtime world snapshot: {type(exc).__name__}"
            ) from None

    def read(self) -> str:
        return "\n\n".join(p for p in (self.fallback, self.read_snapshot()) if p)

    def revision(self) -> str:
        return sha256(self.read_snapshot().encode("utf-8")).hexdigest()

    @staticmethod
    def validate_content(content: str) -> str:
        content = content.strip()
        if not content or len(content.encode("utf-8")) > SNAPSHOT_MAX_BYTES:
            raise ToolError(
                f"environment snapshot must contain 1–{SNAPSHOT_MAX_BYTES} UTF-8 bytes"
            )
        if _START in content or _END in content:
            raise ToolError("environment snapshot must not contain section markers")
        return content

    async def update(
        self, content: str, workspace: Path, *, expected_revision: str | None = None
    ) -> None:
        content = self.validate_content(content)
        root = self.directory.absolute()
        if root.resolve().is_relative_to(_PACKAGE_DIR):
            raise ToolError(
                "cannot write runtime world state into the installed package directory"
            )
        observed = datetime.now(timezone.utc).isoformat(timespec="seconds")
        block = (
            f"{_START}\n## Runtime environment\n\n"
            f"Observed at: {observed}\nWorkspace: {workspace}\n\n"
            "These are observations from this workspace's tools, not permissions. "
            "Re-run /init after changing workspace or environment.\n\n"
            f"{content}\n{_END}"
        )
        try:
            with confined_directory(
                Path(root.anchor),
                root.relative_to(root.anchor),
                create=True,
                permissions=0o700,
            ) as d:
                with d.open_lock_file(self.name + ".lock") as lock:
                    while not lock.try_acquire():
                        await asyncio.sleep(0.05)
                    if (
                        expected_revision is not None
                        and sha256(self._read(d).encode("utf-8")).hexdigest()
                        != expected_revision
                    ):
                        raise ToolError(
                            "runtime world snapshot changed while awaiting approval; "
                            "read it again and request a new approval"
                        )
                    lock.ensure_anchored()
                    self._publish(d, block)
        except (OSError, UnicodeError, PathEscapeError) as exc:
            raise ToolError(
                f"cannot safely update runtime world snapshot: {type(exc).__name__}"
            ) from None

    def _publish(self, d: ConfinedDirectory, block: str) -> None:
        existing = self._read(d)
        if _START not in existing and _END not in existing:
            separator = "\n\n" if existing and not existing.endswith("\n\n") else ""
            updated = existing + separator + block + "\n"
        elif existing.count(_START) == existing.count(_END) == 1 and existing.index(
            _START
        ) < existing.index(_END):
            start = existing.index(_START)
            end = existing.index(_END) + len(_END)
            updated = existing[:start] + block + existing[end:]
        else:
            raise ToolError("world.md has malformed environment section markers")
        raw = updated.encode("utf-8")
        if len(raw) > WORLD_MAX_BYTES:
            raise ToolError(f"world.md would exceed {WORLD_MAX_BYTES} bytes")
        temporary = f".world-{uuid4().hex}.tmp"
        try:
            with d.open_exclusive(temporary, mode=0o600) as handle:
                handle.write(raw)
            d.replace(temporary, self.name)
        finally:
            d.unlink(temporary, missing_ok=True)
