"""Side-effect-free parsing for built-in tool option mappings.

This module lives above ``lingcore.tools.builtin`` so profile loading and
offline diagnostics can validate options without importing and registering the
built-in tool package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lingcore.errors import ConfigError

SEARCH_OPTION_PATH = "tool_options.search"
SEARCH_DEFAULT_EXCLUDE_DIRS = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".next",
)
_SEARCH_OPTION_NAMES = frozenset(
    {
        "max_hits",
        "max_hits_per_file",
        "max_line_chars",
        "max_files_scanned",
        "max_dir_entries",
        "max_file_bytes",
        "max_depth",
        "max_context_lines",
        "time_budget_ms",
        "offload_over_chars",
        "exclude_dirs",
    }
)


@dataclass(frozen=True, slots=True)
class SearchOptions:
    max_hits: int
    max_hits_per_file: int
    max_line_chars: int
    max_files_scanned: int
    max_dir_entries: int
    max_file_bytes: int
    max_depth: int
    max_context_lines: int
    time_budget_ms: int
    offload_over_chars: int
    exclude_dirs: frozenset[str]


def int_option(
    options: Mapping[str, Any],
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    option_path: str,
) -> int:
    """Read a bounded integer while preserving the built-ins' error style."""
    raw = options.get(name, default)
    if isinstance(raw, bool):
        raise ConfigError(f"{option_path}.{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"{option_path}.{name} must be an integer") from None
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ConfigError(f"{option_path}.{name} must be >= {minimum}{upper}")
    return value


def bool_option(
    options: Mapping[str, Any], name: str, default: bool, *, option_path: str
) -> bool:
    """Read a strict boolean option (strings and integers are rejected)."""
    raw = options.get(name, default)
    if not isinstance(raw, bool):
        raise ConfigError(f"{option_path}.{name} must be a boolean")
    return raw


def str_tuple_option(
    options: Mapping[str, Any],
    name: str,
    default: Sequence[str],
    *,
    option_path: str,
) -> tuple[str, ...]:
    """Read a list/tuple of non-empty strings as an immutable tuple."""
    raw = options.get(name, default)
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in raw
    ):
        raise ConfigError(f"{option_path}.{name} must be a list of non-empty strings")
    return tuple(item.strip() for item in raw)


def parse_search_options(raw: object) -> SearchOptions:
    """Validate and normalize the complete ``tool_options.search`` mapping."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{SEARCH_OPTION_PATH} must be a mapping")
    unknown = [name for name in raw if name not in _SEARCH_OPTION_NAMES]
    if unknown:
        name = sorted(str(item) for item in unknown)[0]
        raise ConfigError(f"{SEARCH_OPTION_PATH}.{name} is not supported")
    excludes = str_tuple_option(
        raw,
        "exclude_dirs",
        SEARCH_DEFAULT_EXCLUDE_DIRS,
        option_path=SEARCH_OPTION_PATH,
    )
    for name in excludes:
        if name in {".", ".."} or Path(name).name != name:
            raise ConfigError(
                f"{SEARCH_OPTION_PATH}.exclude_dirs entries must be directory names"
            )
    return SearchOptions(
        max_hits=int_option(
            raw, "max_hits", 100, minimum=1, option_path=SEARCH_OPTION_PATH
        ),
        max_hits_per_file=int_option(
            raw, "max_hits_per_file", 10, minimum=1, option_path=SEARCH_OPTION_PATH
        ),
        max_line_chars=int_option(
            raw, "max_line_chars", 200, minimum=1, option_path=SEARCH_OPTION_PATH
        ),
        max_files_scanned=int_option(
            raw,
            "max_files_scanned",
            5_000,
            minimum=1,
            option_path=SEARCH_OPTION_PATH,
        ),
        max_dir_entries=int_option(
            raw,
            "max_dir_entries",
            10_000,
            minimum=1,
            option_path=SEARCH_OPTION_PATH,
        ),
        max_file_bytes=int_option(
            raw,
            "max_file_bytes",
            256 * 1024,
            minimum=1,
            option_path=SEARCH_OPTION_PATH,
        ),
        max_depth=int_option(
            raw, "max_depth", 25, minimum=0, option_path=SEARCH_OPTION_PATH
        ),
        max_context_lines=int_option(
            raw,
            "max_context_lines",
            5,
            minimum=0,
            option_path=SEARCH_OPTION_PATH,
        ),
        time_budget_ms=int_option(
            raw,
            "time_budget_ms",
            5_000,
            minimum=1,
            option_path=SEARCH_OPTION_PATH,
        ),
        offload_over_chars=int_option(
            raw,
            "offload_over_chars",
            8_000,
            minimum=0,
            option_path=SEARCH_OPTION_PATH,
        ),
        exclude_dirs=frozenset(excludes),
    )


TODO_OPTION_PATH = "tool_options.todo_write"
_TODO_OPTION_NAMES = frozenset({"max_items"})


def parse_todo_max_items(raw: object) -> int:
    """Validate ``tool_options.todo_write`` and return its item cap."""
    from lingcore.todos import DEFAULT_MAX_TODOS

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{TODO_OPTION_PATH} must be a mapping")
    unknown = sorted(str(name) for name in raw if name not in _TODO_OPTION_NAMES)
    if unknown:
        raise ConfigError(f"unknown {TODO_OPTION_PATH} option(s): {', '.join(unknown)}")
    return int_option(
        raw,
        "max_items",
        DEFAULT_MAX_TODOS,
        minimum=1,
        maximum=100,
        option_path=TODO_OPTION_PATH,
    )
