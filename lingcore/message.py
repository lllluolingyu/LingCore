"""Canonical message data model.

Everything in LingCore speaks ``Message`` — the loop, memory, the LLM client,
and frontends. This module sits at the bottom of the runtime dependency graph;
it only imports standalone validation helpers. ``Message.to_openai`` is the
single place that knows the chat-completions wire shape, keeping the OpenAI
coupling contained.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from lingcore.media_types import (
    AttachmentKind,
    FALLBACK_TEXT_MAX_CHARS,
    MAX_ATTACHMENTS,
    TOTAL_ATTACHMENT_MAX_BYTES,
    decoded_payload_size,
    kind_for_media_type,
    max_bytes_for,
    sanitize_name,
    supported_media_types,
    validate_base64_payload,
)

Role = Literal["system", "user", "assistant", "tool"]

# Attachment kinds a chat-completions request carries natively when nothing
# says otherwise. ``to_openai(attachment_modalities=...)`` narrows this set
# per model (see LLMCfg.modalities); ``None`` means "all of them".
NATIVE_MODALITIES: frozenset[str] = frozenset({"image", "file"})


class Attachment(BaseModel):
    """A user-visible media attachment carried alongside message text.

    ``data`` is the raw base64 payload without a ``data:`` URI prefix. The wire
    adapter adds that provider-specific wrapper in ``Message.to_openai``.
    ``fallback_text`` is an optional text stand-in (extracted PDF text, a
    vision model's description, or a diagnostic note) computed once when the
    attachment enters a conversation whose model lacks the native modality;
    the original payload is always kept so a later modality upgrade restores
    native delivery.
    """

    kind: AttachmentKind
    media_type: str
    data: str
    name: str | None = None
    fallback_text: str | None = None

    @model_validator(mode="after")
    def _validate_attachment(self) -> Attachment:
        expected = kind_for_media_type(self.media_type)
        if expected is None:
            raise ValueError(f"unsupported media type: {self.media_type!r}")
        if self.kind != expected:
            raise ValueError(
                f"attachment kind {self.kind!r} does not match media type "
                f"{self.media_type!r}"
            )
        normalized, _ = validate_base64_payload(
            self.data,
            media_type=self.media_type,
            max_bytes=max_bytes_for(self.kind),
        )
        self.data = normalized
        if self.name is not None:
            self.name = sanitize_name(self.name)
        # Truncate (never reject) an over-long fallback: rejecting would make a
        # stored session row unloadable over a derived, non-essential field.
        if (
            self.fallback_text is not None
            and len(self.fallback_text) > FALLBACK_TEXT_MAX_CHARS
        ):
            marker = "\n[fallback text truncated]"
            self.fallback_text = (
                self.fallback_text[: FALLBACK_TEXT_MAX_CHARS - len(marker)] + marker
            )
        return self


class UserInput(BaseModel):
    """One user turn before it is committed as a ``Message``."""

    text: str = ""
    attachments: list[Attachment] = Field(default_factory=list)

    @field_validator("attachments")
    @classmethod
    def _validate_attachments(cls, attachments: list[Attachment]) -> list[Attachment]:
        return _validate_attachment_list(attachments)


class ToolCall(BaseModel):
    """A single tool invocation requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class ToolResult(BaseModel):
    """The outcome of executing one ``ToolCall``.

    ``ok=False`` marks an in-domain failure (a ``ToolError``); the content is
    still fed back to the model so it can recover. Attachments are intentionally
    not rendered on the tool-role wire message; the agent hoists them into a
    synthetic user message because chat-completions only accepts media parts on
    user messages.
    """

    call_id: str
    name: str
    content: str
    ok: bool = True
    attachments: list[Attachment] = Field(default_factory=list)

    @field_validator("attachments")
    @classmethod
    def _validate_attachments(cls, attachments: list[Attachment]) -> list[Attachment]:
        return _validate_attachment_list(attachments)


class Message(BaseModel):
    """One turn in a conversation.

    A role="assistant" message may carry ``tool_calls``. A role="tool"
    message carries ``tool_call_id`` + ``name`` linking it to the call it
    answers — OpenAI rejects a tool message that does not follow a matching
    assistant tool_call, so the two must never be separated (see WindowMemory).
    User messages may carry media attachments; only ``to_openai`` knows how to
    render those into provider wire parts.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)

    @field_validator("attachments")
    @classmethod
    def _validate_attachments(cls, attachments: list[Attachment]) -> list[Attachment]:
        return _validate_attachment_list(attachments)

    # --- constructors -------------------------------------------------
    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(
        cls, content: str, attachments: list[Attachment] | None = None
    ) -> Message:
        return cls(role="user", content=content, attachments=attachments or [])

    @classmethod
    def assistant(
        cls, content: str = "", tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> Message:
        return cls(
            role="tool",
            content=result.content,
            tool_call_id=result.call_id,
            name=result.name,
        )

    # --- wire format --------------------------------------------------
    def to_openai(
        self, *, attachment_modalities: frozenset[str] | None = None
    ) -> dict[str, Any]:
        """Render to a chat-completions message dict.

        ``attachment_modalities`` narrows which attachment kinds may render as
        native media parts (``None`` = all). An attachment outside the set
        renders as text instead — its precomputed ``fallback_text`` when
        present, else a placeholder note — and when *no* native part remains
        the whole content collapses to a plain string, because text-only
        servers may reject a parts array outright.
        """
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content,
            }
        content: str | list[dict[str, Any]] = self.content
        if self.role == "user" and self.attachments:
            modalities = (
                NATIVE_MODALITIES
                if attachment_modalities is None
                else attachment_modalities
            )
            native_parts: list[dict[str, Any]] = []
            fallback_chunks: list[str] = []
            for attachment in self.attachments:
                if attachment.kind not in modalities:
                    fallback_chunks.append(_fallback_block(attachment))
                    continue
                data_uri = (
                    f"data:{attachment.media_type};base64,{attachment.data}"
                )
                if attachment.kind == "image":
                    native_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    })
                else:
                    native_parts.append({
                        "type": "file",
                        "file": {
                            "filename": attachment.name
                            or _default_filename(attachment),
                            "file_data": data_uri,
                        },
                    })
            text = "\n\n".join(
                chunk for chunk in (self.content, *fallback_chunks) if chunk
            )
            if native_parts:
                parts: list[dict[str, Any]] = []
                if text:
                    parts.append({"type": "text", "text": text})
                parts.extend(native_parts)
                content = parts
            else:
                content = text
        msg: dict[str, Any] = {"role": self.role, "content": content}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


def _fallback_block(attachment: Attachment) -> str:
    """Text stand-in for an attachment the model can't receive natively."""
    label = attachment.name or _default_filename(attachment)
    if attachment.fallback_text:
        return (
            f"[{attachment.kind} {label!r} ({attachment.media_type}) as text:]\n"
            f"{attachment.fallback_text}"
        )
    return (
        f"[{attachment.kind} {label!r} ({attachment.media_type}) is attached, "
        "but this model does not support that modality and no text fallback "
        "is available]"
    )


def _default_filename(attachment: Attachment) -> str:
    if attachment.media_type == "application/pdf":
        return "attachment.pdf"
    subtype = (
        attachment.media_type.split("/", 1)[-1]
        if "/" in attachment.media_type
        else "bin"
    )
    return f"attachment.{subtype or 'bin'}"


def _validate_attachment_list(attachments: list[Attachment]) -> list[Attachment]:
    if len(attachments) > MAX_ATTACHMENTS:
        raise ValueError(
            f"too many attachments ({len(attachments)}; limit {MAX_ATTACHMENTS})"
        )
    try:
        total = sum(decoded_payload_size(attachment.data) for attachment in attachments)
    except Exception:
        raise ValueError("invalid attachment base64 data") from None
    if total > TOTAL_ATTACHMENT_MAX_BYTES:
        raise ValueError(
            f"attachments too large ({total} bytes; "
            f"limit {TOTAL_ATTACHMENT_MAX_BYTES})"
        )
    known = supported_media_types()
    for attachment in attachments:
        if (attachment.kind, attachment.media_type) not in known:
            raise ValueError(f"unsupported media type: {attachment.media_type!r}")
    return attachments


class Conversation(BaseModel):
    """An ordered list of messages. A thin convenience wrapper."""

    messages: list[Message] = Field(default_factory=list)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def to_openai(self) -> list[dict[str, Any]]:
        return [m.to_openai() for m in self.messages]

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):  # type: ignore[override]
        return iter(self.messages)
