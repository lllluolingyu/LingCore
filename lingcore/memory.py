"""Short-term memory.

``WindowMemory`` keeps the system prompt plus the most recent messages within
both a message-count and a token budget. Two properties matter:

* **Block-aware trimming.** An assistant message carrying ``tool_calls`` and the
  ``tool`` messages answering them form an atomic block. OpenAI rejects a
  ``tool`` message that does not follow its matching ``tool_calls``, so trimming
  drops whole blocks rather than splitting one.
* **Prefix-stable eviction.** Eviction is *hysteretic*: a monotonic floor marks
  how many oldest blocks have been dropped, and it only advances — in one chunk,
  down to ``evict_to_ratio`` of the budget — when a hard cap is breached.
  Between evictions the rendered message list is append-only, so consecutive
  requests share a byte-identical prefix and the model's prompt cache hits. The
  old behaviour (trim to just under the cap on every render) shifted every
  surviving message's position and invalidated the cache; ``evict_to_ratio=1.0``
  restores it.

``SummarizingMemory`` layers an LLM compaction step in front of that eviction:
when the working set is nearly full it summarizes the oldest history into one
note (keeping the recent tail verbatim) and only falls back to eviction if the
result is still over the hard cap. It composes a ``WindowMemory`` so the window
itself stays free of any LLM dependency (the summarizer is duck-typed, reached
through the same ``stream`` seam as the rest of the runtime).
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import tiktoken

from lingcore.events import Compacted
from lingcore.message import Message


class ShortTermMemory(Protocol):
    def add(self, message: Message) -> None: ...

    def replace(self, messages: list[Message]) -> None: ...

    def render(self, system_prompt: str) -> list[Message]: ...

    async def maybe_compact(self, system_prompt: str = "") -> Compacted | None: ...

    @property
    def messages(self) -> list[Message]: ...


def _encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


class WindowMemory:
    """Sliding-window memory with token + message-count caps, block-aware."""

    def __init__(
        self,
        max_messages: int = 40,
        max_tokens: int = 12_000,
        model: str = "gpt-4o",
        evict_to_ratio: float = 0.5,
    ) -> None:
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        # Low-water mark (tokens) eviction drops down to when a hard cap is
        # breached. Strictly below max_tokens ⇒ hysteresis: the window refills
        # before the next eviction, so the prefix stays stable across many
        # turns. evict_to_ratio == 1.0 reproduces the legacy slide-every-render.
        self._evict_to_tokens = max(1, int(max_tokens * evict_to_ratio))
        self._enc = _encoding(model)
        self._messages: list[Message] = []
        # Lifetime count of oldest blocks physically evicted (monotonic, only
        # ever increases). Evicted messages are *removed* from ``_messages``, so
        # retention is bounded by the window rather than growing with the whole
        # conversation — the store keeps full history for resume, not this list.
        self._floor = 0

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def replace(self, messages: list[Message]) -> None:
        """Replace the retained working set and begin a fresh cache epoch."""
        self._messages = list(messages)
        self._floor = 0

    def _tokens(self, message: Message) -> int:
        # Approximate but stable: encode content plus any tool-call argument
        # text. Exact accounting is the API's job; this only drives trimming.
        n = len(self._enc.encode(message.content or ""))
        for attachment in message.attachments:
            # Floors reflect the wire cost a fallback can't capture: a native
            # image/PDF part (no fallback_text but real tokens), versus text
            # (its inlined content *is* the fallback) and binary (a tiny note).
            if attachment.kind == "image":
                flat = 1_000
            elif attachment.kind == "file":
                flat = 4_000
            elif attachment.kind == "binary":
                flat = 64
            else:  # text — dominated by the inlined fallback length
                flat = 16
            # A text fallback may be what actually goes on the wire; count
            # whichever estimate is larger (over-counting only trims earlier).
            n += max(flat, len(attachment.fallback_text or "") // 4)
        for tc in message.tool_calls:
            n += len(self._enc.encode(tc.name)) + len(self._enc.encode(str(tc.arguments)))
        return n + 4  # per-message overhead fudge

    def _blocks(self) -> list[list[Message]]:
        """Group messages into atomic blocks.

        A block starts at a user message or an assistant message and absorbs
        any trailing ``tool`` messages, so a tool_call and its results stay
        together when trimming.
        """
        blocks: list[list[Message]] = []
        for m in self._messages:
            if m.role == "tool" and blocks:
                blocks[-1].append(m)
            else:
                blocks.append([m])
        return blocks

    def render(self, system_prompt: str) -> list[Message]:
        sys_tokens = len(self._enc.encode(system_prompt))
        self._evict(sys_tokens)
        return [Message.system(system_prompt), *self._messages]

    def _evict(self, sys_tokens: int) -> None:
        """Drop whole oldest blocks from the retained set when a hard cap is
        breached — *physically*, so evicted messages (and any attachment
        payloads they carry) are released rather than retained for the life of
        the process.

        Eviction is hysteretic: nothing is dropped while the working set fits
        both caps, so between evictions ``_messages`` only grows by appends and
        the rendered prefix stays byte-stable (the prompt cache hits). When a
        cap is breached, whole oldest blocks are dropped in one chunk down to
        the token low-water mark (and within the message cap), always keeping at
        least one block. ``_floor`` accumulates the lifetime count of evicted
        blocks (monotonic); ``evict_to_ratio == 1.0`` evicts on every render.
        """
        blocks = self._blocks()
        n = len(blocks)
        if n <= 1:
            return  # keep ≥1 block; nothing is evictable
        block_toks = [sum(self._tokens(m) for m in b) for b in blocks]
        block_msgs = [len(b) for b in blocks]
        toks = sys_tokens + sum(block_toks)
        msgs = sum(block_msgs)
        if toks <= self.max_tokens and msgs <= self.max_messages:
            return  # within budget — prefix stays byte-stable, cache hits
        drop = 0
        while drop < n - 1 and (
            toks > self._evict_to_tokens or msgs > self.max_messages
        ):
            toks -= block_toks[drop]
            msgs -= block_msgs[drop]
            drop += 1
        if drop:
            self._messages = [m for block in blocks[drop:] for m in block]
            self._floor += drop

    async def maybe_compact(self, system_prompt: str = "") -> Compacted | None:
        """No-op: a plain window only evicts; compaction is SummarizingMemory."""
        return None

    # Convenience for tests / inspection.
    @property
    def messages(self) -> list[Message]:
        return list(self._messages)


# Prompt for the compaction summarizer. Kept terse and instruction-only so the
# summary preserves what an agent needs to continue (decisions, paths, tasks).
_SUMMARY_SYSTEM = (
    "You compress conversation history for an AI agent. Produce a terse, "
    "factual summary that preserves goals, decisions, file paths, code changes, "
    "important tool results, and still-open tasks. Use compact bullet points. "
    "Do not add preamble, commentary, or a closing remark."
)


class SummarizingMemory:
    """``ShortTermMemory`` that compacts old history before falling back to the
    window's eviction.

    Composes a ``WindowMemory`` (so the pure window stays LLM-free): ``add``,
    ``render`` and ``messages`` delegate straight to it. ``maybe_compact`` —
    invoked by the loop once per turn, never mid tool-loop — summarizes the
    oldest blocks via the duck-typed ``summarizer`` (anything with the
    ``stream`` shape) when the working set is nearly full, then resets the
    window's floor so the next render starts a fresh, stable prefix. If the
    summary still leaves the window over the hard cap, ``render``'s eviction
    floor finishes the job. The summarizer is reached through the same
    ``stream`` seam as the main model, so this module imports nothing from
    ``llm.py``.
    """

    def __init__(
        self,
        window: WindowMemory,
        summarizer: Any,
        *,
        compact_at_ratio: float = 0.85,
        keep_recent_ratio: float = 0.35,
        max_summary_chars: int = 4_000,
    ) -> None:
        self._w = window
        self._summarizer = summarizer
        self._compact_at = compact_at_ratio
        self._keep_recent = keep_recent_ratio
        self._max_summary_chars = max_summary_chars

    # --- ShortTermMemory delegation -----------------------------------
    def add(self, message: Message) -> None:
        self._w.add(message)

    def replace(self, messages: list[Message]) -> None:
        self._w.replace(messages)

    def render(self, system_prompt: str) -> list[Message]:
        return self._w.render(system_prompt)

    @property
    def messages(self) -> list[Message]:
        return self._w.messages

    # --- compaction ----------------------------------------------------
    async def maybe_compact(self, system_prompt: str = "") -> Compacted | None:
        msgs = self._w._messages
        if not msgs:
            return None
        max_tokens = self._w.max_tokens
        # Count the system prompt too, so the trigger measures the same footprint
        # the window's eviction floor does (a large system prompt must not let
        # the floor evict before compaction ever fires).
        reserved = len(self._w._enc.encode(system_prompt)) if system_prompt else 0
        before = reserved + sum(self._w._tokens(m) for m in msgs)
        if before < self._compact_at * max_tokens:
            return None  # not nearly full

        # Block-aware split: keep the most recent blocks (≥ keep_recent budget)
        # verbatim as the tail; the older head is what gets summarized.
        blocks = self._w._blocks()
        keep_budget = self._keep_recent * max_tokens
        split = len(blocks)  # index of the first tail block
        tail_toks = 0
        for idx in range(len(blocks) - 1, -1, -1):
            b_toks = sum(self._w._tokens(m) for m in blocks[idx])
            if split < len(blocks) and tail_toks + b_toks > keep_budget:
                break
            split = idx
            tail_toks += b_toks
        head_msgs = [m for b in blocks[:split] for m in b]
        if not head_msgs:
            return None  # nothing old enough to summarize; eviction will guard

        try:
            summary = await self._summarize(head_msgs)
        except Exception:
            # Summarizer failed (network, bad backend, ...): never crash the
            # loop — leave the messages alone and let eviction guard the cap.
            return None
        if not summary.strip():
            return None

        summary_msg = Message(
            role="user",
            name="summary",
            content=f"[Earlier conversation, summarized]\n{summary}",
        )
        tail_msgs = [m for b in blocks[split:] for m in b]
        self._w._messages = [summary_msg, *tail_msgs]
        self._w._floor = 0  # new epoch: reset the lifetime eviction counter
        after = reserved + sum(self._w._tokens(m) for m in self._w._messages)
        return Compacted(
            summarized_messages=len(head_msgs),
            before_tokens=before,
            after_tokens=after,
        )

    async def _summarize(self, head_msgs: list[Message]) -> str:
        transcript = _render_transcript(head_msgs)
        request = [
            Message.system(_SUMMARY_SYSTEM),
            Message.user(
                "Summarize this earlier conversation so the agent can continue "
                "without the full text:\n\n" + transcript
            ),
        ]
        parts: list[str] = []
        async for chunk in self._summarizer.stream(request, tools=None):
            if getattr(chunk, "text_delta", ""):
                parts.append(chunk.text_delta)
        summary = "".join(parts).strip()
        if len(summary) > self._max_summary_chars:
            summary = summary[: self._max_summary_chars] + "\n[summary truncated]"
        return summary


def _render_transcript(msgs: list[Message]) -> str:
    """Flatten messages into a plain transcript for the summarizer."""
    lines: list[str] = []
    for m in msgs:
        if m.role == "user":
            label = "Summary" if m.name == "summary" else "User"
            lines.append(f"{label}: {m.content}")
        elif m.role == "assistant":
            if m.content:
                lines.append(f"Assistant: {m.content}")
            for tc in m.tool_calls:
                lines.append(f"Assistant called {tc.name}({json.dumps(tc.arguments)})")
        elif m.role == "tool":
            lines.append(f"Tool[{m.name}]: {m.content}")
    return "\n".join(lines)
