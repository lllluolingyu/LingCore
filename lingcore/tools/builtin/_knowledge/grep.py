"""Bounded, cancellable regular-expression retrieval."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Sequence

import regex

from lingcore.paths import PathEscapeError
from lingcore.tools.builtin._knowledge.common import (
    _DEFAULT_GREP_TIME_BUDGET_MS,
    _GREP_LINE_CHARS,
    _GREP_MATCH_TIMEOUT_MS,
    _MAX_HITS,
    _MAX_READ_BYTES,
)
from lingcore.tools.builtin._knowledge.sources import (
    _iter_source_files,
    _read_source_bytes,
)

# A brace ``re`` accepts as a repeat count; any other ``{`` is a literal to
# ``re`` (and ``{}`` always is) but may be a fuzzy constraint to ``regex``.
_RE_REPEAT = re.compile(r"\{[0-9]*(?:,[0-9]*)?\}")


def _escape_regex_divergences(query: str) -> str:
    """Escape what ``re`` reads as literal but ``regex`` would not.

    Query semantics are defined by the stdlib ``re`` (the grep contract
    predates the timed ``regex`` engine). Two constructs ``re`` accepts as
    literal text are syntax to ``regex``: a non-repeat ``{…}`` (fuzzy
    matching, e.g. ``(?:cat){e<=1}``) and ``[`` inside a character class
    (POSIX classes such as ``[[:alpha:]]``). Escaping both keeps the literal
    meaning in either engine.
    """
    out: list[str] = []
    i, n = 0, len(query)
    in_set = False
    set_body = 0  # index of a character class's first member
    while i < n:
        c = query[i]
        if c == "\\":
            out.append(query[i : i + 2])
            i += 2
            continue
        if in_set:
            if c == "[":
                out.append("\\[")
            else:
                # A "]" leading the class (``[]a]``/``[^]a]``) is a member.
                if c == "]" and i > set_body:
                    in_set = False
                out.append(c)
            i += 1
            continue
        if c == "[":
            j = i + 1
            if j < n and query[j] == "^":
                j += 1
            out.append(query[i:j])
            in_set, set_body, i = True, j, j
            continue
        if c == "{":
            repeat = _RE_REPEAT.match(query, i)
            if repeat is not None and repeat.group() != "{}":
                out.append(repeat.group())
                i = repeat.end()
            else:
                out.append("\\{")
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _compile_grep_pattern(query: str) -> regex.Pattern[str] | None:
    """Compile ``query`` for timed matching with stdlib ``re`` semantics.

    Returns ``None`` (substring matching only) when ``re`` rejects the query,
    or when the divergence escaping cannot be shown to leave ``re``'s parse
    unchanged (e.g. a verbose-mode comment confusing the scan).
    """
    try:
        re.compile(query)
    except re.error:
        return None
    escaped = _escape_regex_divergences(query)
    if escaped != query:
        try:
            from re import _parser  # type: ignore[attr-defined]

            if repr(_parser.parse(escaped)) != repr(_parser.parse(query)):
                return None
        except ImportError:
            pass
        except re.error:
            return None
    try:
        return regex.compile(escaped, regex.VERSION0)
    except regex.error:
        return None


def _grep(
    base: Path,
    sources: Sequence[str],
    query: str,
    *,
    max_hits: int = _MAX_HITS,
    max_line_chars: int = _GREP_LINE_CHARS,
    time_budget_ms: int = _DEFAULT_GREP_TIME_BUDGET_MS,
    cancel_event: threading.Event | None = None,
) -> list[str]:
    hits: list[str] = []
    budget_ms = max(1, time_budget_ms)
    deadline = time.monotonic() + budget_ms / 1_000
    budget_note = f"... (grep timed out after {budget_ms} ms; partial results)"
    timeout_note: str | None = None
    pattern = _compile_grep_pattern(query)

    # The deadline is checked before every file and line whichever matcher
    # runs, so substring fallbacks and skipped (binary/oversized) files are
    # bounded by the same budget as regex matching.
    for full in _iter_source_files(base, sources):
        if cancel_event is not None and cancel_event.is_set():
            return hits
        if time.monotonic() >= deadline:
            timeout_note = budget_note
            break
        try:
            payload, _ = _read_source_bytes(base, full, max_bytes=_MAX_READ_BYTES)
            text = payload.decode("utf-8")
        except (UnicodeDecodeError, OSError, PathEscapeError):
            continue
        rel = full.relative_to(base)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if cancel_event is not None and cancel_event.is_set():
                return hits
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout_note = budget_note
                break
            matched = query in line
            if pattern is not None and not matched:
                match_timeout = min(remaining, _GREP_MATCH_TIMEOUT_MS / 1_000)
                try:
                    # ``concurrent=True`` releases the GIL while matching, so
                    # the worker thread cannot block Stop/other sessions.
                    matched = (
                        pattern.search(line, timeout=match_timeout, concurrent=True)
                        is not None
                    )
                except TimeoutError:
                    # Do not let a single pathological line consume the whole
                    # budget; report it as a bounded partial scan.
                    timeout_note = (
                        budget_note
                        if remaining <= _GREP_MATCH_TIMEOUT_MS / 1_000
                        else f"... (grep stopped after a single regex match "
                        f"exceeded {_GREP_MATCH_TIMEOUT_MS} ms; partial results)"
                    )
                    break
            if matched:
                hits.append(f"{rel}:{lineno}: {line.strip()[:max_line_chars]}")
                if len(hits) >= max_hits:
                    hits.append(f"... (truncated at {max_hits} hits)")
                    return hits
        if timeout_note is not None:
            break
    if timeout_note is not None:
        hits.append(timeout_note)
    return hits
