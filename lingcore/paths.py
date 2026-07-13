"""Shared workspace-confinement helpers.

``resolve_confined`` is the common read/path validation guard.  Security-
sensitive file creation needs a stronger primitive: resolving a pathname and
opening it later leaves a window in which an intermediate directory can be
replaced by a symlink.  ``confined_directory`` walks from the workspace with
no-follow directory descriptors and keeps the final parent descriptor open;
``ConfinedDirectory`` then creates, unlinks, stats, and renames entries relative
to that stable descriptor.

The directory-descriptor API deliberately fails closed on platforms that do
not provide POSIX ``dir_fd`` + ``O_NOFOLLOW`` support.  Attachment ingest
degrades to a note and Canvas reports a tool error rather than falling back to
an unsafe pathname write.
"""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator


class PathEscapeError(Exception):
    """Raised when a candidate path escapes or cannot be opened safely."""


def resolve_confined(base: Path, rel: str | Path) -> Path:
    """Resolve ``rel`` under ``base``, rejecting anything that escapes it."""
    base_resolved = base.resolve()
    full = (base_resolved / rel).resolve()
    if full != base_resolved and not full.is_relative_to(base_resolved):
        raise PathEscapeError(f"path escapes workspace: {str(rel)!r}")
    return full


def _relative_parts(rel: str | Path) -> tuple[str, ...]:
    path = Path(rel)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise PathEscapeError(f"path escapes workspace: {str(rel)!r}")
    return tuple(part for part in path.parts if part not in ("", "."))


def _leaf_name(name: str) -> str:
    if not name or name in (".", "..") or Path(name).name != name:
        raise PathEscapeError(f"invalid confined filename: {name!r}")
    return name


def _dir_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise PathEscapeError(
            "secure confined writes are unsupported on this platform"
        )
    required = (os.open, os.mkdir, os.stat, os.unlink, os.rename)
    if any(fn not in os.supports_dir_fd for fn in required):
        raise PathEscapeError(
            "secure confined writes are unsupported on this platform"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


class ConfinedDirectory:
    """An opened directory beneath a confinement base.

    All entry operations are relative to ``_fd``.  Replacing the pathname with
    a symlink therefore cannot redirect an operation to the link target.  The
    anchor check also refuses an operation once the opened directory is no
    longer reachable at its validated workspace path.
    """

    def __init__(
        self,
        fd: int,
        path: Path,
        base_fd: int,
        parts: tuple[str, ...],
        flags: int,
    ) -> None:
        self._fd = fd
        self.path = path
        self._base_fd = base_fd
        self._parts = parts
        self._flags = flags

    def ensure_anchored(self) -> None:
        """Refuse operations after any validated parent component was replaced."""
        check_fd = os.dup(self._base_fd)
        try:
            for part in self._parts:
                child = os.open(part, self._flags, dir_fd=check_fd)
                os.close(check_fd)
                check_fd = child
            expected = os.fstat(check_fd)
            opened = os.fstat(self._fd)
        except OSError as e:
            raise PathEscapeError(
                f"confined directory changed while writing: {self.path}"
            ) from e
        finally:
            os.close(check_fd)
        if (
            not stat.S_ISDIR(expected.st_mode)
            or expected.st_dev != opened.st_dev
            or expected.st_ino != opened.st_ino
        ):
            raise PathEscapeError(
                f"confined directory changed while writing: {self.path}"
            )

    def open_exclusive(self, name: str, mode: int = 0o644) -> IO[bytes]:
        """Create a new regular-file entry without following any symlink."""
        leaf = _leaf_name(name)
        self.ensure_anchored()
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        fd = os.open(leaf, flags, mode, dir_fd=self._fd)
        return os.fdopen(fd, "wb")

    def entry_exists(self, name: str) -> bool:
        """Whether an entry exists, including a dangling symlink."""
        self.ensure_anchored()
        try:
            os.stat(_leaf_name(name), dir_fd=self._fd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False

    def same_bytes(self, name: str, data: bytes) -> bool:
        """Compare an existing regular file without following a final link."""
        self.ensure_anchored()
        fd = -1
        try:
            fd = os.open(
                _leaf_name(name),
                os.O_RDONLY
                | os.O_NOFOLLOW
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._fd,
            )
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != len(data):
                return False
            with os.fdopen(fd, "rb") as fh:
                fd = -1
                return fh.read(len(data) + 1) == data
        except OSError:
            return False
        finally:
            if fd >= 0:
                os.close(fd)

    def regular_size(self, name: str) -> int | None:
        """Return a no-follow regular file's size, else ``None``."""
        self.ensure_anchored()
        try:
            info = os.stat(
                _leaf_name(name), dir_fd=self._fd, follow_symlinks=False
            )
        except OSError:
            return None
        return info.st_size if stat.S_ISREG(info.st_mode) else None

    def unlink(self, name: str, *, missing_ok: bool = False) -> None:
        try:
            os.unlink(_leaf_name(name), dir_fd=self._fd)
        except FileNotFoundError:
            if not missing_ok:
                raise

    def replace(self, source: str, destination: str) -> None:
        """Atomically replace one entry with another in this same directory."""
        self.ensure_anchored()
        os.replace(
            _leaf_name(source),
            _leaf_name(destination),
            src_dir_fd=self._fd,
            dst_dir_fd=self._fd,
        )


@contextmanager
def confined_directory(
    base: Path, rel: str | Path = ".", *, create: bool = False
) -> Iterator[ConfinedDirectory]:
    """Open a directory below ``base`` without following intermediate links.

    When ``create`` is true, missing components are created one at a time
    relative to the already-open parent.  A concurrent symlink insertion loses
    either the ``mkdir`` race or the subsequent ``O_NOFOLLOW`` open and is
    refused; it is never traversed.
    """
    parts = _relative_parts(rel)
    flags = _dir_flags()
    base_resolved = base.resolve()
    base_fd = os.open(str(base_resolved), flags)
    fd = -1
    current = base_resolved
    try:
        fd = os.dup(base_fd)
        for part in parts:
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                child = os.open(part, flags, dir_fd=fd)
            except OSError as e:
                if e.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise PathEscapeError(
                        f"path escapes workspace or contains a symlink: {str(rel)!r}"
                    ) from None
                raise
            os.close(fd)
            fd = child
            current = current / part
        directory = ConfinedDirectory(fd, current, base_fd, parts, flags)
        directory.ensure_anchored()
        yield directory
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(base_fd)
