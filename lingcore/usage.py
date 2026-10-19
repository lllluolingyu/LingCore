"""Provider-reported token usage, independent of any SDK.

``LLMClient`` asks the provider to append a usage block to every streamed
response and parses it into a :class:`TokenUsage`. Every client an agent owns
(the main model, the compaction/memory summarizer that reuses it, and the
optional vision fallback) reports into one :class:`UsageMeter`; the loop drains
that meter into ``Usage`` events so a frontend can account for every model
request made on a turn's behalf, not only the visible replies. At the end of
each turn the loop also reports a :class:`TurnStats`: that turn's requests and
how long it spent generating, which session stores persist per turn.

Counts follow the OpenAI chat-completions convention: ``input_tokens``
*includes* ``cached_input_tokens`` and ``output_tokens`` *includes*
``reasoning_tokens``. The values are what the provider reported; LingCore never
estimates them, and a request whose stream failed before its usage block is
simply absent.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeGuard

UsageSink = Callable[["TokenUsage"], None]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Usage of one model request, as reported by the provider."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


def _count(value: Any) -> int:
    # bool is an int subclass but never a token count.
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def usage_from_openai(model: str, usage: Any) -> TokenUsage | None:
    """Parse an OpenAI-compatible ``usage`` object (SDK model or plain dict).

    Cached input is read from ``prompt_tokens_details.cached_tokens`` or, for
    servers such as DeepSeek that report it at the top level,
    ``prompt_cache_hit_tokens``. Returns ``None`` when no usage was reported.
    """
    if usage is None:
        return None
    details = _field(usage, "prompt_tokens_details")
    cached = _field(details, "cached_tokens")
    if cached is None:
        cached = _field(usage, "prompt_cache_hit_tokens")
    reasoning = _field(_field(usage, "completion_tokens_details"), "reasoning_tokens")
    input_tokens = _count(_field(usage, "prompt_tokens"))
    output_tokens = _count(_field(usage, "completion_tokens"))
    return TokenUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=min(_count(cached), input_tokens),
        reasoning_tokens=min(_count(reasoning), output_tokens),
    )


def usage_from_anthropic(model: str, usage: Any) -> TokenUsage | None:
    """Parse Anthropic usage (SDK model or dict) into inclusive token counts.

    Anthropic's input_tokens excludes both cache reads and cache creation;
    TokenUsage includes them. Returns ``None`` when no usage was reported.
    """
    if usage is None:
        return None
    uncached = _count(_field(usage, "input_tokens"))
    cached = _count(_field(usage, "cache_read_input_tokens"))
    created = _count(_field(usage, "cache_creation_input_tokens"))
    output_tokens = _count(_field(usage, "output_tokens"))
    thinking = _field(_field(usage, "output_tokens_details"), "thinking_tokens")
    return TokenUsage(
        model=model,
        input_tokens=uncached + cached + created,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        reasoning_tokens=min(_count(thinking), output_tokens),
    )


class UsageMeter:
    """Collects usage reported by an agent's clients until the loop drains it.

    A tally, while open, additionally keeps every record for one turn's
    :class:`TurnStats`, independent of when (or whether) the loop drains.
    """

    def __init__(self) -> None:
        self._pending: list[TokenUsage] = []
        self._tally: list[TokenUsage] | None = None

    def record(self, usage: TokenUsage) -> None:
        self._pending.append(usage)
        if self._tally is not None:
            self._tally.append(usage)

    def drain(self) -> list[TokenUsage]:
        pending, self._pending = self._pending, []
        return pending

    def open_tally(self) -> None:
        self._tally = []

    def close_tally(self) -> tuple[TokenUsage, ...]:
        tally, self._tally = self._tally or [], None
        return tuple(tally)


@dataclass(frozen=True, slots=True)
class TurnStats:
    """What one turn cost and how long it spent generating.

    ``usage`` is every request billed on the turn's behalf, in order — the
    replies and any compaction, memory-condensing or vision request.
    ``model_ms`` is the turn's wall time less the time it spent running tool
    batches (which includes waiting for approvals) and backing off between
    stream retries, so output tokens over ``model_ms`` is the turn's
    throughput. It still includes each request's time to first token.
    """

    usage: tuple[TokenUsage, ...]
    started_at: datetime
    ended_at: datetime
    elapsed_ms: int
    model_ms: int


def turn_stats_payload(stats: TurnStats) -> dict[str, Any]:
    """JSON payload persisted as a ``turn_stats`` session event."""
    return {
        "usage": [
            {
                "model": u.model,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cached_input_tokens": u.cached_input_tokens,
                "reasoning_tokens": u.reasoning_tokens,
            }
            for u in stats.usage
        ],
        "started_at": stats.started_at.isoformat(),
        "ended_at": stats.ended_at.isoformat(),
        "elapsed_ms": stats.elapsed_ms,
        "model_ms": stats.model_ms,
    }


def _is_ms(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def turn_stats_from_payload(payload: Any) -> TurnStats | None:
    """Validate a stored payload; ``None`` for anything malformed, never raises."""
    if not isinstance(payload, dict):
        return None
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, list):
        return None
    usage: list[TokenUsage] = []
    for item in raw_usage:
        if not isinstance(item, dict) or not isinstance(item.get("model"), str):
            return None
        usage.append(
            TokenUsage(
                model=item["model"],
                input_tokens=_count(item.get("input_tokens")),
                output_tokens=_count(item.get("output_tokens")),
                cached_input_tokens=_count(item.get("cached_input_tokens")),
                reasoning_tokens=_count(item.get("reasoning_tokens")),
            )
        )
    try:
        started_at = datetime.fromisoformat(payload["started_at"])
        ended_at = datetime.fromisoformat(payload["ended_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if started_at.tzinfo is None or ended_at.tzinfo is None:
        return None
    elapsed_ms = payload.get("elapsed_ms")
    model_ms = payload.get("model_ms")
    if not (_is_ms(elapsed_ms) and _is_ms(model_ms)):
        return None
    return TurnStats(
        usage=tuple(usage),
        started_at=started_at,
        ended_at=ended_at,
        elapsed_ms=elapsed_ms,
        model_ms=min(model_ms, elapsed_ms),
    )


@dataclass(slots=True)
class TurnClock:
    """Measures one turn: wall time, less the spans spent waiting."""

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _start: float = field(default_factory=lambda: time.monotonic())
    _waited: float = 0.0

    @contextmanager
    def waiting(self) -> Iterator[None]:
        """Exclude the enclosed span (a tool batch, a retry backoff)."""
        since = time.monotonic()
        try:
            yield
        finally:
            self._waited += time.monotonic() - since

    def finish(self, usage: tuple[TokenUsage, ...]) -> TurnStats:
        elapsed = max(0.0, time.monotonic() - self._start)
        elapsed_ms = round(elapsed * 1000)
        return TurnStats(
            usage=usage,
            started_at=self.started_at,
            ended_at=datetime.now(timezone.utc),
            elapsed_ms=elapsed_ms,
            model_ms=min(elapsed_ms, round(max(0.0, elapsed - self._waited) * 1000)),
        )
