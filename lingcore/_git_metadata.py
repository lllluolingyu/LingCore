"""Copy untrusted Git metadata without importing executable repository settings."""

from __future__ import annotations

import asyncio
import configparser
import tempfile
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator

from lingcore.errors import ToolError
from lingcore.paths import ConfinedDirectory, confined_directory


def copy_metadata(
    workspace: Path, target: Path, *, files: int = 20000, size: int = 268435456
) -> None:
    count = total = 0

    def copy(
        source: ConfinedDirectory, dest: ConfinedDirectory, prefix: str = ""
    ) -> None:
        nonlocal count, total
        for name, directory, regular, symlink in source.iter_entries():
            relative = prefix + name
            if relative in {"config", "config.worktree", "hooks", "logs"}:
                continue
            if symlink or name in {"commondir", "alternates", "http-alternates"}:
                raise ToolError(
                    "Git metadata symlinks/alternates/redirections are refused"
                )
            count += 1
            if count > files:
                raise ToolError("Git metadata exceeds file cap")
            if directory:
                with (
                    source.subdirectory(name) as src,
                    dest.subdirectory(name, create=True) as dst,
                ):
                    copy(src, dst, relative + "/")
            elif regular:
                data = source.read_regular(name, max_bytes=size - total)
                total += len(data)
                with dest.open_exclusive(name, mode=0o600) as output:
                    output.write(data)
            else:
                raise ToolError("Git metadata must contain ordinary files")

    with (
        confined_directory(workspace, ".git") as source,
        confined_directory(target, ".git", create=True) as dest,
    ):
        # Parse only the object format as inert data. Never follow include paths
        # or copy filters, helpers, aliases, hooks, fsmonitor, or worktree settings.
        parser = configparser.RawConfigParser(strict=False)
        if source.entry_exists("config"):
            parser.read_string(
                source.read_regular("config", max_bytes=1048576).decode()
            )
        fmt = parser.get("extensions", "objectformat", fallback="sha1").strip()
        if fmt not in {"sha1", "sha256"}:
            raise ToolError("unsupported Git object format")
        copy(source, dest)
        config = "[core]\nrepositoryformatversion = 0\nbare = false\n"
        if fmt == "sha256":
            config = "[core]\nrepositoryformatversion = 1\nbare = false\n[extensions]\nobjectformat = sha256\n"
        with dest.open_exclusive("config", mode=0o600) as output:
            output.write(config.encode())


@contextmanager
def metadata_snapshot(workspace: Path, **limits: int) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="lingcore-git-") as name:
        target = Path(name)
        copy_metadata(workspace, target, **limits)
        yield target


@asynccontextmanager
async def isolated_metadata(workspace: Path) -> AsyncIterator[Path]:
    with tempfile.TemporaryDirectory(prefix="lingcore-git-") as name:
        target = Path(name)
        task = asyncio.create_task(asyncio.to_thread(copy_metadata, workspace, target))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise
        yield target
