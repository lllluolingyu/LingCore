"""Short-term memory.

``WindowMemory`` keeps the system prompt plus the most recent messages within
both a message-count and a token budget. The one subtlety that matters for
correctness: an assistant message carrying ``tool_calls`` and the ``tool``
messages answering them form an atomic block. OpenAI rejects a ``tool``
message that does not follow its matching ``tool_calls``, so trimming drops
whole blocks from the front rather than splitting one.

The ``summarize`` policy is a future implementation behind the same Protocol;
it is intentionally not built yet.
"""

from __future__ import annotations

from typing import Protocol

import tiktoken

from lingcore.message import Message


class ShortTermMemory(Protocol):
    def add(self, message: Message) -> None: ...

    def render(self, system_prompt: str) -> list[Message]: ...


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
    ) -> None:
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._enc = _encoding(model)
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

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
        blocks = self._blocks()
        kept: list[list[Message]] = []
        msg_count = 0
        tok_count = len(self._enc.encode(system_prompt))

        # Walk newest-to-oldest, keeping whole blocks until a budget is hit.
        for block in reversed(blocks):
            b_msgs = len(block)
            b_toks = sum(self._tokens(m) for m in block)
            if kept and (
                msg_count + b_msgs > self.max_messages
                or tok_count + b_toks > self.max_tokens
            ):
                break
            kept.append(block)
            msg_count += b_msgs
            tok_count += b_toks

        flat = [m for block in reversed(kept) for m in block]
        return [Message.system(system_prompt), *flat]

    # Convenience for tests / inspection.
    @property
    def messages(self) -> list[Message]:
        return list(self._messages)
