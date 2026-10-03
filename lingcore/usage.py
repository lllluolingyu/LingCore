"""Provider-reported token usage, independent of any SDK.

``LLMClient`` asks the provider to append a usage block to every streamed
response and parses it into a :class:`TokenUsage`. Every client an agent owns
(the main model, the compaction/memory summarizer that reuses it, and the
optional vision fallback) reports into one :class:`UsageMeter`; the loop drains
that meter into ``Usage`` events so a frontend can account for every model
request made on a turn's behalf, not only the visible replies.

Counts follow the OpenAI chat-completions convention: ``input_tokens``
*includes* ``cached_input_tokens`` and ``output_tokens`` *includes*
``reasoning_tokens``. The values are what the provider reported; LingCore never
estimates them, and a request whose stream failed before its usage block is
simply absent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    """Collects usage reported by an agent's clients until the loop drains it."""

    def __init__(self) -> None:
        self._pending: list[TokenUsage] = []

    def record(self, usage: TokenUsage) -> None:
        self._pending.append(usage)

    def drain(self) -> list[TokenUsage]:
        pending, self._pending = self._pending, []
        return pending
