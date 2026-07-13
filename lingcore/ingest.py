"""Workspace ingest for user attachments.

Every file a user attaches is copied into ``<workspace>/attachments/`` so the
agent's workspace-confined tools (``read_file``, ``pdf2md``, ``run_shell``, …)
can reach it, and the model is told where it landed via a note line appended to
the user message. Beyond the copy, ingest computes the text stand-ins only it
can: a ``text`` file's content is inlined (capped), and a ``binary`` file gets a
short "saved here, use tools" pointer. Image/PDF fallbacks are deliberately
*not* computed here — that needs the model's vision client, so it stays with
``MediaAdapter`` (which the agent runs right after ingest); ingest only copies
those and announces the path.

``ingest_attachments`` never raises (the loop's invariant 5 extends here): a
copy that fails degrades to a note and the turn proceeds. It is synchronous
blocking I/O, so the agent calls it through ``asyncio.to_thread`` to keep the
event loop free.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from lingcore.media_types import TEXT_INLINE_MAX_CHARS, sanitize_name
from lingcore.message import Attachment
from lingcore.paths import ConfinedDirectory, PathEscapeError, confined_directory

_ATTACH_DIRNAME = "attachments"
_MAX_COLLISION_SUFFIX = 99
_DEFAULT_NAMES = {"text": "attachment.txt", "binary": "attachment.bin"}


def ingest_attachments(
    attachments: list[Attachment], workspace: Path
) -> tuple[list[Attachment], list[str]]:
    """Copy attachments into the workspace; return ``(attachments, notes)``.

    The returned list mirrors the input order. An attachment is replaced by a
    ``model_copy`` only when ingest sets its ``fallback_text`` (text/binary), so
    an image/PDF object passes through with its identity intact. ``notes`` holds
    one human-readable line per attachment, for the caller to append to the user
    message text.
    """
    if not attachments:
        return attachments, []
    out: list[Attachment] = list(attachments)
    notes: list[str] = []
    for i, attachment in enumerate(attachments):
        try:
            data = base64.b64decode(attachment.data)
        except Exception:
            notes.append(
                f"[attachment {_label(attachment)} could not be decoded; skipped]"
            )
            continue
        rel_path, error = _store(data, attachment, workspace)
        if rel_path is not None:
            notes.append(
                f"[attached: {rel_path} ({attachment.media_type}, {len(data)} bytes)]"
            )
        else:
            notes.append(
                f"[attachment {_label(attachment)} could not be saved to the "
                f"workspace ({error}); it is still attached to this message]"
            )
        if attachment.fallback_text is None:
            fallback = _fallback_for(attachment, data, rel_path)
            if fallback is not None:
                out[i] = attachment.model_copy(update={"fallback_text": fallback})
    return out, notes


def _label(attachment: Attachment) -> str:
    return repr(attachment.name or attachment.media_type)


def _store(
    data: bytes, attachment: Attachment, workspace: Path
) -> tuple[str | None, str | None]:
    """Save ``data`` under ``<workspace>/attachments/`` with a collision-safe
    name. Returns ``(workspace-relative posix path, None)`` on success or
    ``(None, error)`` on failure — never raises.

    The attachments directory is opened by descriptor with no-follow traversal,
    and the descriptor stays open through collision selection and creation. A
    parent swapped to a symlink after validation therefore cannot redirect the
    payload; a changed directory anchor makes the write fail closed.
    """
    try:
        with confined_directory(
            workspace, _ATTACH_DIRNAME, create=True
        ) as directory:
            _ensure_gitignore(directory)
            name = sanitize_name(
                attachment.name,
                fallback=_DEFAULT_NAMES.get(attachment.kind, "attachment"),
            )
            target_name = _unique_name(directory, name, data)
            rel = f"{_ATTACH_DIRNAME}/{target_name}"
            created = False
            try:
                with directory.open_exclusive(target_name) as fh:
                    created = True
                    fh.write(data)
            except FileExistsError:
                # _unique_name returns an existing name only for byte-identical
                # dedupe. O_EXCL also catches a name raced in after selection.
                if not directory.same_bytes(target_name, data):
                    return None, "destination changed while saving"
            except OSError:
                if created:
                    directory.unlink(target_name, missing_ok=True)
                raise
            try:
                directory.ensure_anchored()
            except PathEscapeError:
                if created:
                    directory.unlink(target_name, missing_ok=True)
                raise
            return rel, None
    except PathEscapeError as e:
        return None, str(e)
    except OSError as e:
        return None, _summarize(e)


def _unique_name(directory: ConfinedDirectory, name: str, data: bytes) -> str:
    """An entry name under ``directory``: reuse a byte-identical existing file,
    else append ``-2``, ``-3``, … (finally a content hash) to avoid clobbering."""
    if not directory.entry_exists(name) or directory.same_bytes(name, data):
        return name
    stem, dot, suffix = name.partition(".")
    if not stem:  # a dotfile like ".env" — keep it whole, no extension split
        stem, dot, suffix = name, "", ""
    ext = f"{dot}{suffix}" if dot else ""
    for n in range(2, _MAX_COLLISION_SUFFIX + 1):
        candidate = f"{stem}-{n}{ext}"
        if not directory.entry_exists(candidate) or directory.same_bytes(
            candidate, data
        ):
            return candidate
    digest = hashlib.sha256(data).hexdigest()[:8]
    return f"{stem}-{digest}{ext}"


def _ensure_gitignore(directory: ConfinedDirectory) -> None:
    """Best-effort: keep workspace uploads out of a user's git repo.

    Created exclusively (no-follow): a planted ``.gitignore`` symlink —
    dangling ones pass an ``exists()`` check — must never redirect this write
    outside the workspace. Any existing path (or an unwritable dir) just means
    we skip; ingest never fails over housekeeping.
    """
    try:
        with directory.open_exclusive(".gitignore") as fh:
            fh.write(b"*\n")
    except OSError:
        pass


def _fallback_for(
    attachment: Attachment, data: bytes, rel_path: str | None
) -> str | None:
    """The text stand-in for a text/binary attachment (image/PDF -> None)."""
    if attachment.kind == "text":
        return _inline_text(data, rel_path)
    if attachment.kind == "binary":
        where = f" saved to {rel_path}" if rel_path else ""
        return (
            f"[binary file{where} ({attachment.media_type}, {len(data)} bytes); "
            "not displayable inline — inspect it with workspace tools]"
        )
    return None


def _inline_text(data: bytes, rel_path: str | None) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) <= TEXT_INLINE_MAX_CHARS:
        return text
    where = f" — full file at {rel_path}" if rel_path else ""
    marker = f"\n[truncated at {TEXT_INLINE_MAX_CHARS} chars{where}]"
    return text[: TEXT_INLINE_MAX_CHARS - len(marker)] + marker


def _summarize(exc: BaseException) -> str:
    msg = " ".join(str(exc).split())
    return msg[:200] if msg else type(exc).__name__
