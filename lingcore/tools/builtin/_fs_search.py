"""Bounded descriptor-based workspace search and coverage reporting."""

from __future__ import annotations

import fnmatch
import heapq
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import regex
from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.media import is_probably_binary
from lingcore.paths import (
    ConfinedDirectory,
    PathEscapeError,
    confined_directory,
    resolve_confined,
)
from lingcore.tool_options import SearchOptions, parse_search_options
from lingcore.tools import ToolContext
from lingcore.tools.builtin._offload import RUNTIME_DIRNAME, offload_text

_SEARCH_CHECK_BATCH = 128


def _validate_search_glob(pattern: str) -> None:
    """Reject glob patterns that can enumerate outside the workspace."""
    if not pattern.strip():
        raise ToolError("search glob must not be empty")
    p = Path(pattern)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ToolError(f"glob escapes workspace: {pattern!r}")


class SearchArgs(BaseModel):
    query: str | None = Field(
        default=None,
        description="Content pattern; required when mode is `content`.",
    )
    mode: Literal["content", "files"] = Field(
        default="content",
        description=(
            "`content` returns path:line matches; `files` returns matching paths. "
            "In files mode, omit query to find files by name only."
        ),
    )
    path: str = Field(
        default=".", description="Workspace-relative directory that scopes the walk."
    )
    glob: str = Field(
        default="**/*",
        description=(
            "Filename filter. Without `/`, matches the basename at any depth "
            "(`*.py`); with `/`, matches paths relative to `path` using fnmatch "
            "semantics. A leading `**/` also matches files at the scope root."
        ),
    )
    regex: bool = Field(
        default=False,
        description="Use Python regex syntax with timed regex VERSION0 matching.",
    )
    ignore_case: bool = Field(
        default=False, description="Match content without case sensitivity."
    )
    context: int = Field(
        default=0,
        ge=0,
        description="Context lines around content matches (configuration-capped).",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Per-call hit cap, bounded by the configured max_hits.",
    )


@dataclass(slots=True)
class _SearchState:
    deadline: float
    matches: int = 0
    matched_files: int = 0
    scanned_files: int = 0
    scanned_dirs: int = 0
    skipped_too_large: int = 0
    skipped_binary: int = 0
    skipped_non_utf8: int = 0
    skipped_non_regular: int = 0
    skipped_symlink: int = 0
    skipped_unreadable: int = 0
    pruned: set[str] = field(default_factory=set)
    hit_match_cap: bool = False
    hit_file_cap: bool = False
    hit_dir_entry_cap: bool = False
    hit_depth_cap: bool = False
    stopped_time: bool = False
    content_results: list[tuple[str, str]] = field(default_factory=list)
    file_results: list[str] = field(default_factory=list)


def _search_options(ctx: ToolContext) -> SearchOptions:
    raw = ctx.options.get("search", {}) if ctx.options else {}
    return parse_search_options(raw)


def _search_matcher(
    args: SearchArgs, state: _SearchState
) -> Callable[[str], bool] | None:
    query = args.query
    if query is None:
        if args.mode == "content":
            raise ToolError("content search requires a query")
        return None
    if not query:
        raise ToolError("search query must not be empty")
    if args.regex:
        try:
            # Keep Python regex syntax validation. The timed engine uses its
            # VERSION0 semantics (Unicode case folding can differ from re).
            re.compile(query, re.IGNORECASE if args.ignore_case else 0)
            pattern = regex.compile(
                query, regex.VERSION0 | (regex.IGNORECASE if args.ignore_case else 0)
            )
        except (re.error, regex.error) as exc:
            raise ToolError(f"invalid search regex: {exc}") from None

        def match(line: str) -> bool:
            remaining = state.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            # A deadline between lines cannot interrupt catastrophic
            # backtracking within a line. Bound each match by the remaining
            # scan budget and release the GIL so other turns can keep running.
            return pattern.search(line, timeout=remaining, concurrent=True) is not None

        return match
    if args.ignore_case:
        needle = query.casefold()
        return lambda line: needle in line.casefold()
    return lambda line: query in line


def _glob_matches(pattern: str, relative_path: str) -> bool:
    if "/" not in pattern:
        return fnmatch.fnmatchcase(relative_path.rsplit("/", 1)[-1], pattern)
    if fnmatch.fnmatchcase(relative_path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatchcase(
        relative_path, pattern.removeprefix("**/")
    )


def _deadline_reached(state: _SearchState) -> bool:
    if state.stopped_time:
        return True
    if time.monotonic() >= state.deadline:
        state.stopped_time = True
        return True
    return False


def _bounded_entries(
    directory: ConfinedDirectory,
    options: SearchOptions,
    state: _SearchState,
) -> list[tuple[str, bool, bool, bool]]:
    count = 0

    def entries() -> Iterator[tuple[str, bool, bool, bool]]:
        nonlocal count
        for entry in directory.iter_entries():
            count += 1
            if count % _SEARCH_CHECK_BATCH == 0 and _deadline_reached(state):
                break
            yield entry

    try:
        selected = heapq.nsmallest(
            options.max_dir_entries, entries(), key=lambda entry: entry[0]
        )
    except PathEscapeError:
        state.skipped_unreadable += 1
        return []
    if count > options.max_dir_entries:
        state.hit_dir_entry_cap = True
    if state.stopped_time:
        return []
    return selected


def _render_content_file(
    path: str,
    lines: list[str],
    line_numbers: list[int],
    *,
    context: int,
    max_line_chars: int,
    omitted: int,
) -> str:
    def match_text(line: str) -> str:
        return line.strip()[:max_line_chars]

    def context_text(line: str) -> str:
        return line[:max_line_chars]

    if context == 0:
        rendered = [
            f"{path}:{lineno}: {match_text(lines[lineno - 1])}"
            for lineno in line_numbers
        ]
    else:
        intervals: list[tuple[int, int]] = []
        for lineno in line_numbers:
            start = max(1, lineno - context)
            end = min(len(lines), lineno + context)
            if intervals and start <= intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
            else:
                intervals.append((start, end))
        matches = set(line_numbers)
        groups: list[str] = []
        for start, end in intervals:
            group: list[str] = []
            for lineno in range(start, end + 1):
                separator = ":" if lineno in matches else "-"
                text = (
                    match_text(lines[lineno - 1])
                    if lineno in matches
                    else context_text(lines[lineno - 1])
                )
                group.append(f"{path}{separator}{lineno}{separator} {text}")
            groups.append("\n".join(group))
        rendered = ["\n--\n".join(groups)]
    if omitted:
        rendered.append(f"{path}: (+{omitted} more matches in this file)")
    return "\n".join(rendered)


def _record_content_matches(
    path: str,
    lines: list[str],
    matcher: Callable[[str], bool],
    *,
    context: int,
    hit_cap: int,
    options: SearchOptions,
    state: _SearchState,
) -> bool:
    line_numbers: list[int] = []
    matches_seen = 0
    stop = False
    completed = True
    for lineno, line in enumerate(lines, start=1):
        if (lineno - 1) % _SEARCH_CHECK_BATCH == 0 and _deadline_reached(state):
            stop = True
            completed = False
            break
        try:
            matched = matcher(line)
        except TimeoutError:
            state.stopped_time = True
            stop = True
            completed = False
            break
        if not matched:
            continue
        matches_seen += 1
        if len(line_numbers) >= options.max_hits_per_file:
            continue
        if state.matches >= hit_cap:
            state.hit_match_cap = True
            stop = True
            completed = False
            break
        line_numbers.append(lineno)
        state.matches += 1
    if line_numbers:
        state.matched_files += 1
        state.content_results.append(
            (
                path,
                _render_content_file(
                    path,
                    lines,
                    line_numbers,
                    context=context,
                    max_line_chars=options.max_line_chars,
                    omitted=matches_seen - len(line_numbers) if completed else 0,
                ),
            )
        )
    return stop


def _record_file_content_match(
    path: str,
    lines: list[str],
    matcher: Callable[[str], bool],
    *,
    hit_cap: int,
    options: SearchOptions,
    state: _SearchState,
) -> bool:
    for lineno, line in enumerate(lines, start=1):
        if (lineno - 1) % _SEARCH_CHECK_BATCH == 0 and _deadline_reached(state):
            return True
        try:
            matched = matcher(line)
        except TimeoutError:
            state.stopped_time = True
            return True
        if not matched:
            continue
        if state.matches >= hit_cap:
            state.hit_match_cap = True
            return True
        state.matches += 1
        state.matched_files += 1
        state.file_results.append(path)
        return False
    return False


def _count_read_skip(exc: PathEscapeError, state: _SearchState) -> None:
    message = str(exc)
    if "exceeds" in message or "changed beyond" in message:
        state.skipped_too_large += 1
    elif "not a regular file" in message:
        state.skipped_non_regular += 1
    else:
        state.skipped_unreadable += 1


def _scan_file(
    directory: ConfinedDirectory,
    name: str,
    path: str,
    args: SearchArgs,
    matcher: Callable[[str], bool] | None,
    *,
    context: int,
    hit_cap: int,
    options: SearchOptions,
    state: _SearchState,
) -> bool:
    if state.scanned_files >= options.max_files_scanned:
        state.hit_file_cap = True
        return True
    state.scanned_files += 1
    if args.mode == "files" and matcher is None:
        if state.matches >= hit_cap:
            state.hit_match_cap = True
            return True
        state.matches += 1
        state.matched_files += 1
        state.file_results.append(path)
        return False
    try:
        payload, _ = directory.read_regular_with_stat(
            name, max_bytes=options.max_file_bytes
        )
    except PathEscapeError as exc:
        _count_read_skip(exc, state)
        return False
    if is_probably_binary(payload):
        state.skipped_binary += 1
        return False
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        state.skipped_non_utf8 += 1
        return False
    lines = text.splitlines()
    if matcher is None:  # only files mode without a query reaches this branch
        return False
    if args.mode == "files":
        return _record_file_content_match(
            path,
            lines,
            matcher,
            hit_cap=hit_cap,
            options=options,
            state=state,
        )
    return _record_content_matches(
        path,
        lines,
        matcher,
        context=context,
        hit_cap=hit_cap,
        options=options,
        state=state,
    )


def _walk_search(
    directory: ConfinedDirectory,
    scope_parts: tuple[str, ...],
    workspace_parts: tuple[str, ...],
    depth: int,
    args: SearchArgs,
    matcher: Callable[[str], bool] | None,
    *,
    context: int,
    hit_cap: int,
    options: SearchOptions,
    state: _SearchState,
) -> bool:
    if _deadline_reached(state):
        return True
    state.scanned_dirs += 1
    for name, is_dir, is_file, is_symlink in _bounded_entries(
        directory, options, state
    ):
        if _deadline_reached(state):
            return True
        relative_scope = "/".join(scope_parts + (name,))
        relative_workspace = "/".join(workspace_parts + (name,))
        if is_symlink:
            if _glob_matches(args.glob, relative_scope):
                state.skipped_symlink += 1
            continue
        if is_dir:
            if name == RUNTIME_DIRNAME or name in options.exclude_dirs:
                # Offloading may create this internal directory after the walk.
                # Keep that side effect out of coverage so repeats stay stable.
                if name != RUNTIME_DIRNAME:
                    state.pruned.add(name)
                continue
            if depth >= options.max_depth:
                state.hit_depth_cap = True
                continue
            try:
                with directory.subdirectory(name) as child:
                    if _walk_search(
                        child,
                        scope_parts + (name,),
                        workspace_parts + (name,),
                        depth + 1,
                        args,
                        matcher,
                        context=context,
                        hit_cap=hit_cap,
                        options=options,
                        state=state,
                    ):
                        return True
            except PathEscapeError:
                state.skipped_unreadable += 1
            continue
        if not _glob_matches(args.glob, relative_scope):
            continue
        if not is_file:
            state.skipped_non_regular += 1
            continue
        if _scan_file(
            directory,
            name,
            relative_workspace,
            args,
            matcher,
            context=context,
            hit_cap=hit_cap,
            options=options,
            state=state,
        ):
            return True
    return state.stopped_time


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or singular + "s")


def _coverage_footer(state: _SearchState, options: SearchOptions, hit_cap: int) -> str:
    parts = [
        f"{state.matches} {_plural(state.matches, 'match', 'matches')} in "
        f"{state.matched_files} {_plural(state.matched_files, 'file')}",
        f"scanned {state.scanned_files} {_plural(state.scanned_files, 'file')}, "
        f"{state.scanned_dirs} {_plural(state.scanned_dirs, 'dir')}",
    ]
    skipped = [
        (state.skipped_too_large, "too large", "too large"),
        (state.skipped_binary, "binary", "binaries"),
        (state.skipped_non_utf8, "non-UTF-8", "non-UTF-8"),
        (state.skipped_non_regular, "non-regular", "non-regular"),
        (state.skipped_symlink, "symlink", "symlinks"),
        (state.skipped_unreadable, "unreadable", "unreadable"),
    ]
    skip_notes = [
        f"{count} {_plural(count, singular, plural)}"
        for count, singular, plural in skipped
        if count
    ]
    if skip_notes:
        parts.append("skipped " + ", ".join(skip_notes))
    if state.pruned:
        parts.append("pruned " + ", ".join(sorted(state.pruned)))
    if state.hit_match_cap:
        parts.append(f"hit the {hit_cap}-match cap")
    if state.stopped_time:
        parts.append(f"stopped after the {options.time_budget_ms} ms budget")
    if state.hit_file_cap:
        parts.append(f"hit the {options.max_files_scanned}-file scan cap")
    if state.hit_dir_entry_cap:
        parts.append(f"hit a {options.max_dir_entries}-entry directory cap")
    if state.hit_depth_cap:
        parts.append(f"hit the {options.max_depth}-level depth cap")
    return "(" + "; ".join(parts) + ")"


def _search_sync(args: SearchArgs, ctx: ToolContext, options: SearchOptions) -> str:
    state = _SearchState(deadline=time.monotonic() + options.time_budget_ms / 1_000)
    matcher = _search_matcher(args, state)
    base = ctx.workspace.resolve()
    try:
        scope_path = resolve_confined(base, args.path)
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from None
    if not scope_path.is_dir():
        raise ToolError(f"not a directory: {args.path!r}")
    hit_cap = min(args.limit or options.max_hits, options.max_hits)
    context = min(args.context, options.max_context_lines)
    try:
        with confined_directory(base, args.path) as scope:
            workspace_parts = tuple(scope.path.relative_to(base).parts)
            _walk_search(
                scope,
                (),
                workspace_parts,
                0,
                args,
                matcher,
                context=context,
                hit_cap=hit_cap,
                options=options,
                state=state,
            )
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from None
    except OSError:
        raise ToolError(f"not a directory: {args.path!r}") from None

    if args.mode == "content":
        separator = "\n--\n" if context else "\n"
        body = separator.join(rendered for _, rendered in sorted(state.content_results))
    else:
        body = "\n".join(sorted(state.file_results))
    if not body:
        body = "(no matches)"
    footer = _coverage_footer(state, options, hit_cap)
    result = body + "\n" + footer
    if options.offload_over_chars > 0 and len(result) <= options.offload_over_chars:
        return result
    # Stage/truncate only the body: coverage is the most important information
    # when output is large or partial, so it must remain visible at the end.
    body_threshold = (
        0
        if options.offload_over_chars <= 0
        else min(options.offload_over_chars, max(1, len(body) - 1))
    )
    rendered_body = offload_text(
        ctx,
        source="search",
        text=body,
        threshold=body_threshold,
    )
    return rendered_body + "\n" + footer
