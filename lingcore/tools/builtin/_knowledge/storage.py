"""Confined SQLite index serialization and schema management."""

from __future__ import annotations

import secrets
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

from lingcore.errors import ConfigError, ToolError
from lingcore.paths import (
    ConfinedDirectory,
    PathEscapeError,
    confined_directory,
)
from lingcore.tools.builtin._knowledge.common import (
    _DEFAULT_INDEX_PATH,
    _MAX_INDEX_BYTES,
    _SCHEMA_VERSION,
    _confined,
)


def _index_path(base: Path, options: Mapping[str, Any]) -> Path:
    raw_value = options.get("index_path", _DEFAULT_INDEX_PATH)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ConfigError("tool_options.knowledge.index_path must be non-empty")
    raw = Path(raw_value)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise ToolError(f"knowledge index path escapes workspace: {raw_value!r}")
    candidate = base / raw
    if candidate.is_symlink():
        raise ToolError("knowledge index path must not be a symlink")
    resolved = candidate.resolve()
    if not _confined(base, resolved):
        raise ToolError(f"knowledge index path escapes workspace: {raw_value!r}")
    return resolved


def _open_index(
    base: Path,
    path: Path,
    *,
    create: bool,
    directory: ConfinedDirectory | None = None,
) -> sqlite3.Connection | None:
    """Load a confined SQLite file into an in-memory connection.

    SQLite's pathname API cannot accept an already-open no-follow descriptor.
    Deserializing a bounded, descriptor-read database avoids reopening an
    attacker-swappable workspace path, and indexing later publishes the whole
    database with the same atomic confined-write primitive as other runtime
    artifacts.
    """
    rel = path.relative_to(base)
    payload: bytes | None = None
    try:
        parent = (
            nullcontext(directory)
            if directory is not None
            else confined_directory(base, rel.parent, create=create)
        )
        with parent as directory:
            if not directory.entry_exists(rel.name):
                if not create:
                    return None
            else:
                payload = directory.read_regular(rel.name, max_bytes=_MAX_INDEX_BYTES)
    except FileNotFoundError:
        if not create:
            return None
        raise ToolError(
            f"cannot create knowledge index directory: {path.parent}"
        ) from None
    except (OSError, PathEscapeError) as exc:
        raise ToolError(f"cannot safely open knowledge index {path}: {exc}") from None

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(":memory:")
        if payload:
            connection.deserialize(payload)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, _SCHEMA_VERSION}:
            raise ConfigError(
                f"knowledge index schema {version} is unsupported; remove {path} "
                "and run knowledge action=index"
            )
        if version == 0:
            if not create and payload is not None:
                raise ConfigError(
                    f"knowledge index has no recognized schema; remove {path} "
                    "and rebuild it"
                )
            _create_schema(connection)
        return connection
    except ConfigError:
        if connection is not None:
            connection.close()
        raise
    except sqlite3.DatabaseError:
        if connection is not None:
            connection.close()
        raise ToolError(
            f"knowledge index is unreadable or corrupt: {path}; remove it and rebuild"
        ) from None


def _save_index(
    base: Path,
    path: Path,
    connection: sqlite3.Connection,
    *,
    directory: ConfinedDirectory,
) -> None:
    """Atomically publish an in-memory index through a confined directory."""
    try:
        payload = connection.serialize()
        if len(payload) > _MAX_INDEX_BYTES:
            raise ToolError(
                f"knowledge index exceeds the {_MAX_INDEX_BYTES}-byte size limit"
            )
        rel = path.relative_to(base)
        part = f".{rel.name}.{secrets.token_hex(8)}.part"
        try:
            with directory.open_exclusive(part) as handle:
                handle.write(payload)
            directory.replace(part, rel.name)
        finally:
            directory.unlink(part, missing_ok=True)
    except ToolError:
        raise
    except (OSError, PathEscapeError, sqlite3.DatabaseError) as exc:
        raise ToolError(f"cannot safely write knowledge index {path}: {exc}") from None


def _create_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.executescript(f"""
            CREATE TABLE IF NOT EXISTS documents (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                chunk_config TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                chunk_key TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL REFERENCES documents(path) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                page INTEGER,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding_identity TEXT,
                embedding_dim INTEGER,
                embedding BLOB,
                UNIQUE(path, ordinal)
            );
            CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
            CREATE INDEX IF NOT EXISTS chunks_vector_idx
                ON chunks(embedding_identity) WHERE embedding IS NOT NULL;
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            PRAGMA user_version = {_SCHEMA_VERSION};
        """)
        connection.commit()
    except sqlite3.DatabaseError:
        connection.rollback()
        raise ConfigError(
            "SQLite FTS5 support is required for the knowledge index backend"
        ) from None
