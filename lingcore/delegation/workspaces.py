"""Bounded, no-follow workspace copies and hardened independent Git clones."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from lingcore._git_metadata import metadata_snapshot
from lingcore.delegation.config import WorkspaceConfig
from lingcore.delegation.models import digest
from lingcore.errors import ToolError
from lingcore.paths import ConfinedDirectory, confined_directory

ID = re.compile(r"[a-z][a-z0-9_-]{0,100}\Z")


def safe_id(value: str) -> str:
    if not ID.fullmatch(value):
        raise ToolError("invalid run or worker identifier")
    return value


def git(
    directory: Path,
    *args: str,
    data: bytes | None = None,
    cap: int = 268435456,
    worktree: Path | None = None,
) -> bytes:
    if not Path("/proc/self/fd").is_dir():
        raise ToolError("confined snapshot Git operations require Linux procfs")
    env = {
        "PATH": os.defpath,
        "HOME": str(directory),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_AUTHOR_NAME": "LingCore",
        "GIT_AUTHOR_EMAIL": "lingcore@localhost",
        "GIT_COMMITTER_NAME": "LingCore",
        "GIT_COMMITTER_EMAIL": "lingcore@localhost",
    }
    with (
        confined_directory(directory, ".") as opened,
        tempfile.TemporaryFile() as output,
        ExitStack() as stack,
    ):
        # Keep cwd pinned even if a sibling replaces a pathname. No shell,
        # inherited credentials, global Git config, hooks, or network protocols.
        opened.ensure_anchored()
        if args[0] not in {"clone", "init"}:
            env["GIT_DIR"] = f"/proc/self/fd/{opened._fd}/.git"
            env["GIT_WORK_TREE"] = f"/proc/self/fd/{opened._fd}"
        descriptors = [opened._fd]
        if worktree is not None:
            source = stack.enter_context(confined_directory(worktree, "."))
            env["GIT_WORK_TREE"] = f"/proc/self/fd/{source._fd}"
            descriptors.append(source._fd)
        proc = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "protocol.file.allow=always",
                *args,
            ],
            cwd=f"/proc/self/fd/{opened._fd}",
            pass_fds=tuple(descriptors),
            env=env,
            input=data,
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        opened.ensure_anchored()
        if proc.returncode:
            raise ToolError(f"workspace git operation failed ({args[0]})")
        if output.tell() > cap:
            raise ToolError("workspace Git output exceeds snapshot byte cap")
        output.seek(0)
        return output.read()


class WorkspaceManager:
    def __init__(
        self,
        workspace: Path,
        config: WorkspaceConfig,
        *,
        excludes: set[str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.excludes = {
            ".git",
            ".lingcore",
            "node_modules",
            "__pycache__",
            ".venv",
        } | (excludes or set())

    def path(self, run_id: str, worker_id: str | None = None) -> Path:
        rel = Path(".lingcore/runs") / safe_id(run_id)
        if worker_id:
            rel /= safe_id(worker_id)
        with confined_directory(self.workspace, rel, create=True) as directory:
            return directory.path

    def _copy(
        self,
        source: ConfinedDirectory,
        target: ConfinedDirectory,
        prefix: str,
        count: list[int],
        skipped: list[str],
        selected: set[str] | None = None,
        force_dirs: set[str] | None = None,
    ) -> None:
        for name, directory, regular, symlink in sorted(source.iter_entries()):
            rel = f"{prefix}{name}"
            if name == ".git":
                continue
            if symlink:
                skipped.append(rel + " (symlink)")
            elif directory and (
                name not in self.excludes or force_dirs and rel in force_dirs
            ):
                with (
                    source.subdirectory(name) as src,
                    target.subdirectory(name, create=True) as dst,
                ):
                    self._copy(
                        src, dst, rel + "/", count, skipped, selected, force_dirs
                    )
            elif regular and (selected is None or rel in selected):
                size = source.regular_size(name) or 0
                if (
                    count[0] >= self.config.snapshot_max_files
                    or count[1] + size > self.config.snapshot_max_bytes
                ):
                    skipped.append(rel + " (cap)")
                    continue
                data, info = source.read_regular_with_stat(
                    name, max_bytes=self.config.snapshot_max_bytes - count[1]
                )
                if not target.entry_exists(name):
                    with target.open_exclusive(
                        name, mode=stat.S_IMODE(info.st_mode) & 0o755
                    ) as handle:
                        handle.write(data)
                count[0] += 1
                count[1] += len(data)

    def capture(
        self, run_id: str, worker_id: str, *, frozen: bool = False
    ) -> tuple[Path, str, list[str]]:
        target = self.path(run_id, worker_id)
        if any(target.iterdir()):
            raise ToolError("snapshot destination must be empty")
        skipped: list[str] = []
        marker = self.workspace / ".git"
        if marker.exists() or marker.is_symlink():
            from lingcore.tools import ToolContext
            from lingcore.tools.builtin.git import _validate_checkout

            _validate_checkout(ToolContext(self.workspace))
            self._metadata()
            with metadata_snapshot(
                self.workspace,
                files=self.config.snapshot_max_files,
                size=self.config.snapshot_max_bytes,
            ) as metadata:
                base = (
                    git(metadata, "rev-parse", "--revs-only", "HEAD").decode().strip()
                )
                if base:
                    git(
                        target,
                        "clone",
                        "--local",
                        "--no-hardlinks",
                        "--no-checkout",
                        "--",
                        str(metadata),
                        ".",
                    )
                selected = set(
                    git(
                        metadata,
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                        "-z",
                        worktree=self.workspace,
                    )
                    .decode()
                    .split("\0")
                ) | set(
                    git(metadata, "ls-files", "--cached", "-z").decode().split("\0")
                )
            if base:
                git(target, "checkout", "--detach", base)
                patch = git(
                    target,
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--ignore-submodules=all",
                    "--binary",
                    "HEAD",
                    cap=self.config.snapshot_max_bytes,
                    worktree=self.workspace,
                )
                if patch:
                    git(target, "apply", "--binary", "-", data=patch)
        else:
            base, selected = "", None
        with (
            confined_directory(self.workspace, ".") as src,
            confined_directory(target, ".") as dst,
        ):
            self._skip_links(dst, "", skipped)
            self._copy(src, dst, "", [0, 0], skipped, selected)
        if not frozen:
            if not base:
                git(target, "init")
            git(target, "add", "-A")
            git(target, "commit", "--allow-empty", "-m", "LingCore snapshot base")
            # This metadata lives outside the worker's mounted/file-tool root.
            # Never consult the worker's .git again to construct its patch.
            baseline = self.path(run_id, "base-" + worker_id)
            git(
                baseline,
                "clone",
                "--local",
                "--no-hardlinks",
                "--no-checkout",
                "--",
                str(target),
                ".",
            )
            git(baseline, "read-tree", "HEAD")
        revision = self.revision(target, skipped)
        if frozen:
            self._readonly(target)
        return target, revision, skipped

    def revision(
        self, target: Path, skipped: list[str], *, source: bool = False
    ) -> str:
        manifest: list[Any] = []
        size = 0

        def walk(directory: ConfinedDirectory, prefix: str) -> None:
            nonlocal size
            for name, is_dir, regular, symlink in sorted(directory.iter_entries()):
                if name == ".git" or source and name in self.excludes:
                    continue
                if symlink:
                    if source:
                        manifest.append(
                            (
                                prefix + name,
                                "symlink",
                                os.readlink(name, dir_fd=directory._fd),
                            )
                        )
                        continue
                    raise ToolError("snapshot contains a symlink")
                rel = prefix + name
                if is_dir:
                    with directory.subdirectory(name) as child:
                        walk(child, rel + "/")
                elif regular:
                    data, info = directory.read_regular_with_stat(
                        name, max_bytes=self.config.snapshot_max_bytes
                    )
                    size += len(data)
                    if (
                        size > self.config.snapshot_max_bytes
                        or len(manifest) >= self.config.snapshot_max_files
                    ):
                        raise ToolError("snapshot exceeds workspace limits")
                    manifest.append(
                        (
                            rel,
                            bool(info.st_mode & 0o111),
                            hashlib.sha256(data).hexdigest(),
                        )
                    )

        with confined_directory(target, ".") as directory:
            walk(directory, "")
        base = (
            git(target, "rev-parse", "--revs-only", "HEAD").decode().strip()
            if (target / ".git").is_dir()
            else ""
        )
        return digest({"base": base, "files": manifest, "skipped": sorted(skipped)})

    def source_revision(self) -> str | None:
        try:
            return self.revision(self.workspace, [], source=True)
        except (OSError, ToolError):
            return None

    def patch(self, run_id: str, worker_id: str) -> str | None:
        target = self.path(run_id, worker_id)
        baseline = self.path(run_id, "base-" + worker_id)
        if not (baseline / ".git").is_dir():
            raise ToolError("host-owned snapshot baseline is missing")
        # Copy data only, excluding all nested Git metadata and symlinks. Even a
        # replacement .git or a nested repository must never reach host Git.
        with tempfile.TemporaryDirectory(prefix="lingcore-patch-") as name:
            tree = Path(name)
            tracked = git(baseline, "ls-files", "-z").decode().split("\0")
            force_dirs = {
                str(parent) for file in tracked for parent in Path(file).parents
            }
            with (
                confined_directory(target, ".") as src,
                confined_directory(tree, ".") as dst,
            ):
                skipped: list[str] = []
                self._copy(src, dst, "", [0, 0], skipped, force_dirs=force_dirs)
                if any(v.endswith("(cap)") for v in skipped):
                    raise ToolError("worker patch exceeds workspace limits")
                if skipped:
                    raise ToolError(
                        "worker patch contains unsupported symlinks: "
                        + ", ".join(skipped[:20])
                    )
            git(baseline, "add", "-A", worktree=tree)
            data = git(
                baseline,
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--cached",
                "--binary",
                "HEAD",
                cap=self.config.snapshot_max_bytes,
            )
        name = safe_id(worker_id) + ".patch"
        self.write(run_id, name, data)
        return str(Path(".lingcore/runs") / run_id / name)

    def _skip_links(
        self, directory: ConfinedDirectory, prefix: str, skipped: list[str]
    ) -> None:
        for name, is_dir, _, symlink in directory.iter_entries():
            if name == ".git":
                continue
            if symlink:
                directory.unlink(name)
                skipped.append(prefix + name + " (symlink)")
            elif is_dir:
                with directory.subdirectory(name) as child:
                    self._skip_links(child, prefix + name + "/", skipped)

    def write(self, run_id: str, name: str, data: bytes) -> None:
        import secrets

        with confined_directory(
            self.workspace, Path(".lingcore/runs") / safe_id(run_id), create=True
        ) as directory:
            temp = secrets.token_hex(10) + ".part"
            with directory.open_exclusive(temp, mode=0o600) as handle:
                handle.write(data)
            directory.replace(temp, name)

    def read(self, relative: str) -> bytes:
        path = Path(relative)
        with confined_directory(self.workspace, path.parent) as directory:
            return directory.read_regular(
                path.name, max_bytes=self.config.snapshot_max_bytes
            )

    def artifact_digest(self, relative: str) -> str:
        path = Path(relative)
        # Preserve the existing digest(hex(payload)) journal format while
        # bounding memory to one chunk rather than a payload plus its hex copy.
        hashed, total = hashlib.sha256(b'"'), 0
        with confined_directory(self.workspace, path.parent) as directory:
            with directory.open_regular(path.name) as source:
                while chunk := source.read(65536):
                    total += len(chunk)
                    if total > self.config.snapshot_max_bytes:
                        raise ToolError("artifact exceeds workspace byte cap")
                    hashed.update(chunk.hex().encode())
        hashed.update(b'"')
        return hashed.hexdigest()

    def verify_artifacts(self, expected: dict[str, str]) -> bool:
        return all(
            self.artifact_digest(path) == value for path, value in expected.items()
        )

    def retain_artifact(
        self, workspace: Path, relative: str, run_id: str, name: str, remaining: int
    ) -> int:
        from uuid import uuid4

        path = Path(relative)
        total = 0
        with (
            confined_directory(workspace, path.parent) as origin,
            confined_directory(
                self.workspace, Path(".lingcore/runs") / safe_id(run_id), create=True
            ) as destination,
        ):
            temp = uuid4().hex + ".part"
            try:
                with (
                    origin.open_regular(path.name) as source,
                    destination.open_exclusive(temp) as target,
                ):
                    while chunk := source.read(65536):
                        total += len(chunk)
                        if total > remaining:
                            raise ToolError("artifacts exceed workspace byte cap")
                        target.write(chunk)
                destination.replace(temp, name)
            finally:
                if destination.entry_exists(temp):
                    destination.unlink(temp)
        return total

    @staticmethod
    def _readonly(path: Path) -> None:
        for root, dirs, files in os.walk(path, topdown=False, followlinks=False):
            for name in files:
                file = Path(root) / name
                if file.is_symlink():
                    raise ToolError("snapshot metadata contains a symlink")
                os.chmod(file, file.stat().st_mode & ~0o222, follow_symlinks=False)
            for name in dirs:
                directory = Path(root) / name
                if directory.is_symlink():
                    raise ToolError("snapshot metadata contains a symlink")
                os.chmod(directory, 0o500, follow_symlinks=False)
        os.chmod(path, 0o500, follow_symlinks=False)

    def remove(self, run_id: str, worker_id: str | None = None) -> None:
        relative = Path(".lingcore/runs") / safe_id(run_id)
        if worker_id:
            relative /= safe_id(worker_id)
        with confined_directory(self.workspace, relative.parent) as directory:
            directory.ensure_anchored()

            def writable(opened: ConfinedDirectory) -> None:
                os.fchmod(opened._fd, 0o700)
                for name, is_dir, _, symlink in opened.iter_entries():
                    if is_dir and not symlink:
                        with opened.subdirectory(name) as child:
                            writable(child)

            with directory.subdirectory(relative.name) as target:
                writable(target)
            # Python's fd-based rmtree never traverses symlinks in the tree.
            shutil.rmtree(relative.name, dir_fd=directory._fd)

    def _metadata(self) -> None:
        """Refuse redirected/oversized object stores before invoking local clone."""
        count, size = 0, 0

        def walk(directory: ConfinedDirectory) -> None:
            nonlocal count, size
            for name, is_dir, regular, symlink in directory.iter_entries():
                if symlink or name in {"commondir", "alternates", "shallow"}:
                    raise ToolError("snapshot Git metadata redirects or is incomplete")
                count += 1
                if count > self.config.snapshot_max_files:
                    raise ToolError("Git metadata exceeds snapshot file cap")
                if is_dir:
                    with directory.subdirectory(name) as child:
                        walk(child)
                elif regular:
                    size += directory.regular_size(name) or 0
                    if size > self.config.snapshot_max_bytes:
                        raise ToolError("Git metadata exceeds snapshot byte cap")
                else:
                    raise ToolError("Git metadata must contain ordinary files")

        with confined_directory(self.workspace, ".git") as directory:
            walk(directory)
