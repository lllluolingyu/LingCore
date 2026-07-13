"""Canvas LMS tools — shipped *inside the canvas skill* (invariant 13).

This module is loaded by ``lingcore.skills.load_skill_tools`` via importlib when
a profile engages the ``canvas`` skill; its ``@tool`` functions register into the
global ``REGISTRY``.  Nothing here lives in ``lingcore/tools/builtin`` — all
Canvas code is self-contained in this skill directory.

Design notes:
- LingCore is async-everywhere, so this is an ``httpx.AsyncClient`` port of the
  reference ``canvas_get``/``sync_canvas`` (which used synchronous ``requests``).
- The token is read from an env var named by ``tool_options.canvas.token_env``
  at call time — never stored in profile YAML (invariant 4).
- ``base_url`` is operator-configured (not model-controlled), so the SSRF surface
  is small: the model only supplies course ids / paths.  We still reject
  embedded credentials and non-http(s) schemes.
- Network failures surface as ``ToolError`` → ``ToolResult(ok=False)``; they never
  crash the loop (invariant 5).
- Downloads are confined under the workspace with the same ``_resolve`` guard the
  builtin fs tools use, so a hostile filename can't escape.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from lingcore import __version__ as _lingcore_version
from lingcore.errors import ToolError
from lingcore.paths import (
    ConfinedDirectory,
    PathEscapeError,
    confined_directory,
    resolve_confined,
)
from lingcore.tools import ToolContext, tool

_TIMEOUT = 30.0
_DOWNLOAD_TIMEOUT = 120.0
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB per file
_DEFAULT_EXT = (".pdf", ".pptx", ".ppt", ".docx")
_USER_AGENT = f"LingCore/{_lingcore_version}"  # tracks the package version
_TAG_RE = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------- #
# Config + low-level Canvas client                                            #
# --------------------------------------------------------------------------- #


def _canvas_cfg(ctx: ToolContext) -> tuple[str, str, dict[str, Any]]:
    """Return ``(base_url, token, opts)`` from ``ctx.options['canvas']``.

    Raises ``ToolError`` (recoverable by the model / surfaced to the user) when
    the base URL or token are missing or malformed.
    """
    opts = (ctx.options or {}).get("canvas", {}) or {}
    base_url = str(opts.get("base_url", "")).strip().rstrip("/")
    if not base_url:
        raise ToolError(
            "Canvas is not configured: set tool_options.canvas.base_url "
            "(e.g. ${CANVAS_URL}) in the profile."
        )
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"Canvas base_url must be http(s): {base_url!r}")
    if parsed.username or parsed.password:
        raise ToolError("Canvas base_url must not embed credentials")
    token_env = str(opts.get("token_env", "CANVAS_TOKEN"))
    token = os.environ.get(token_env, "")
    if not token:
        raise ToolError(
            f"Canvas access token env var {token_env!r} is not set; export it "
            "before launching the agent."
        )
    return base_url, token, opts


def _next_link(link_header: str) -> str | None:
    """Parse the Canvas ``Link`` header and return the ``rel="next"`` URL."""
    for part in link_header.split(","):
        if 'rel="next"' in part:
            return part.split(";")[0].strip().strip("<>").strip()
    return None


def _same_origin(url: str, base_url: str) -> bool:
    """True when ``url`` has the same scheme/host/port as the Canvas base URL."""
    def origin(value: str) -> tuple[str, str, int] | None:
        try:
            parsed = urlparse(value)
            scheme = parsed.scheme.lower()
            host = parsed.hostname
            if scheme not in ("http", "https") or host is None:
                return None
            port = parsed.port
        except ValueError:
            return None
        if port is None:
            port = 443 if scheme == "https" else 80
        return scheme, host.lower(), port

    candidate = origin(url)
    return candidate is not None and candidate == origin(base_url)


async def _canvas_get(
    base_url: str, token: str, path: str, params: dict[str, Any] | None = None
) -> Any:
    """GET ``{base_url}/api/v1{path}`` with bearer auth, following pagination.

    Accumulates list responses across ``Link rel="next"`` pages; returns a
    non-list JSON body (a dict) directly.  Wraps ``httpx`` errors as ``ToolError``.
    """
    headers = {"Authorization": f"Bearer {token}", "User-Agent": _USER_AGENT}
    items: list[Any] = []
    next_url: str | None = f"{base_url}/api/v1{path}"
    next_params = params
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        while next_url:
            try:
                resp = await client.get(next_url, headers=headers, params=next_params)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ToolError(
                    f"Canvas API error {e.response.status_code} for {path!r}"
                ) from None
            except httpx.HTTPError as e:
                raise ToolError(f"Canvas request failed for {path!r}: {e}") from None
            data = resp.json()
            if not isinstance(data, list):
                return data
            items.extend(data)
            link = resp.headers.get("Link") or resp.headers.get("link") or ""
            next_url = _next_link(link)
            # Only follow (and re-send the bearer token to) a pagination link on
            # the configured Canvas origin. A next-link pointing off-origin is
            # unexpected and would leak credentials to another host, so refuse.
            if next_url is not None and not _same_origin(next_url, base_url):
                raise ToolError(
                    f"Canvas pagination link points off-origin ({next_url!r}); "
                    "refusing to send credentials off the configured Canvas host"
                )
            next_params = None  # the next-link URL already carries the query
    return items


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _sanitize(name: str) -> str:
    return "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "untitled"


def _strip_html(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", html)).strip()


def _confined(workspace: Path, rel: str) -> Path:
    """Resolve ``rel`` under the workspace, rejecting escapes (shared guard)."""
    try:
        return resolve_confined(workspace, rel)
    except PathEscapeError as e:
        raise ToolError(str(e)) from None


def _filter_courses(courses: Any, course_ids: Any) -> list[dict[str, Any]]:
    out = [c for c in courses if isinstance(c, dict) and c.get("id") is not None]
    if course_ids:
        wanted = {str(i) for i in course_ids}
        out = [c for c in out if str(c.get("id")) in wanted]
    return out


async def _resolve_courses(
    base_url: str, token: str, opts: dict[str, Any], course_id: int | None
) -> list[dict[str, Any]]:
    """Active courses (filtered by configured ``course_ids``), optionally narrowed
    to a single ``course_id`` the model asked for."""
    courses = await _canvas_get(
        base_url, token, "/courses", {"enrollment_state": "active", "per_page": 50}
    )
    courses = _filter_courses(courses, opts.get("course_ids"))
    if course_id is not None:
        courses = [c for c in courses if str(c.get("id")) == str(course_id)]
    return courses


def _folder_path(folder: Any, all_folders: dict[Any, Any]) -> Path:
    """Reconstruct a folder's path relative to the course root."""
    parts: list[str] = []
    current = folder
    seen: set[Any] = set()
    while isinstance(current, dict):
        fid = current.get("id")
        if fid in seen:  # defensive cycle guard
            break
        seen.add(fid)
        name = current.get("name", "")
        if current.get("parent_folder_id") is None or name.lower() in (
            "course files",
            "files",
        ):
            break
        parts.append(_sanitize(name))
        current = all_folders.get(current.get("parent_folder_id"))
    parts.reverse()
    return Path(*parts) if parts else Path(".")


# --------------------------------------------------------------------------- #
# Tools                                                                       #
# --------------------------------------------------------------------------- #


class CoursesArgs(BaseModel):
    pass


@tool(description="List the user's active Canvas courses (id, name, course code).")
async def canvas_courses(args: CoursesArgs, ctx: ToolContext) -> str:
    base_url, token, opts = _canvas_cfg(ctx)
    courses = _filter_courses(
        await _canvas_get(
            base_url, token, "/courses", {"enrollment_state": "active", "per_page": 50}
        ),
        opts.get("course_ids"),
    )
    if not courses:
        return "No active courses found."
    lines = []
    for c in courses:
        cid = c.get("id")
        name = c.get("name") or f"course {cid}"
        code = c.get("course_code") or ""
        lines.append(f"- [{cid}] {name}" + (f" ({code})" if code else ""))
    return "Active courses:\n" + "\n".join(lines)


class AssignmentsArgs(BaseModel):
    course_id: int | None = Field(
        default=None,
        description="Limit to one course id; omit to span all active courses.",
    )


@tool(
    description="List Canvas assignments with due dates and points, sorted by due "
    "date, across active courses (or one course)."
)
async def canvas_assignments(args: AssignmentsArgs, ctx: ToolContext) -> str:
    base_url, token, opts = _canvas_cfg(ctx)
    courses = await _resolve_courses(base_url, token, opts, args.course_id)
    entries: list[tuple[str, str]] = []  # (sort_key, text)
    notes: list[str] = []
    for c in courses:
        cid = c.get("id")
        cname = c.get("name") or f"course {cid}"
        try:
            asn = await _canvas_get(
                base_url,
                token,
                f"/courses/{cid}/assignments",
                {"per_page": 100, "order_by": "due_at"},
            )
        except ToolError as e:
            notes.append(f"  (could not load {cname}: {e})")
            continue
        for a in asn:
            if not isinstance(a, dict) or a.get("published") is False:
                continue
            title = a.get("name") or "(untitled)"
            due = a.get("due_at")
            pts = a.get("points_possible")
            pts_s = f", {pts} pts" if pts is not None else ""
            text = f"- [{cname}] {title} — due {due or 'no due date'}{pts_s}"
            entries.append((due or "9999-12-31", text))
    if not entries:
        body = "No published assignments found."
    else:
        entries.sort(key=lambda e: e[0])
        body = "Assignments (by due date):\n" + "\n".join(t for _, t in entries)
    return body + ("\n" + "\n".join(notes) if notes else "")


class AnnouncementsArgs(BaseModel):
    course_id: int | None = Field(
        default=None,
        description="Limit to one course id; omit to span all active courses.",
    )


@tool(
    description="List recent Canvas announcements across active courses (or one "
    "course), most-recent context first."
)
async def canvas_announcements(args: AnnouncementsArgs, ctx: ToolContext) -> str:
    base_url, token, opts = _canvas_cfg(ctx)
    courses = await _resolve_courses(base_url, token, opts, args.course_id)
    out: list[str] = []
    for c in courses:
        cid = c.get("id")
        cname = c.get("name") or f"course {cid}"
        try:
            anns = await _canvas_get(
                base_url,
                token,
                f"/courses/{cid}/discussion_topics",
                {"only_announcements": True, "per_page": 20},
            )
        except ToolError:
            continue
        for a in anns:
            if not isinstance(a, dict):
                continue
            title = a.get("title") or "(untitled)"
            posted = a.get("posted_at") or a.get("created_at") or "?"
            msg = _strip_html(a.get("message") or "")
            if len(msg) > 280:
                msg = msg[:280] + "…"
            out.append(f"- [{cname}] {title} ({posted})" + (f"\n    {msg}" if msg else ""))
    return "Announcements:\n" + "\n".join(out) if out else "No announcements found."


class SyncArgs(BaseModel):
    course_id: int | None = Field(
        default=None,
        description="Sync a single course id; omit to sync all active courses.",
    )


@tool(
    description="Download Canvas course files (PDF/PPT/DOCX by default) into the "
    "workspace, mirroring the Canvas folder layout and skipping unchanged files."
)
async def canvas_sync(args: SyncArgs, ctx: ToolContext) -> str:
    base_url, token, opts = _canvas_cfg(ctx)
    exts = tuple(str(e).lower() for e in (opts.get("file_ext") or _DEFAULT_EXT))
    dl_dir = str(opts.get("download_dir", "canvas"))
    # Validate the configured root eagerly; individual file parents are created
    # later by no-follow directory-descriptor traversal.
    _confined(ctx.workspace, dl_dir)

    courses = await _resolve_courses(base_url, token, opts, args.course_id)
    downloaded: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    async with httpx.AsyncClient(
        timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True
    ) as client:
        for c in courses:
            cid = c.get("id")
            cname = _sanitize(c.get("name") or str(cid))
            try:
                folders = await _canvas_get(
                    base_url, token, f"/courses/{cid}/folders", {"per_page": 100}
                )
                files = await _canvas_get(
                    base_url, token, f"/courses/{cid}/files", {"per_page": 100}
                )
            except ToolError as e:
                errors.append(f"{cname}: {e}")
                continue
            folder_map = {
                f.get("id"): f for f in folders if isinstance(f, dict)
            }
            for f in files:
                if not isinstance(f, dict):
                    continue
                fname = f.get("display_name") or f.get("filename") or "unknown"
                if Path(fname).suffix.lower() not in exts:
                    continue
                sub = _folder_path(folder_map.get(f.get("folder_id")), folder_map)
                rel = Path(dl_dir) / cname / sub / _sanitize(fname)
                try:
                    # Resolve only the parent for containment. The final name is
                    # handled as a no-follow directory entry, so a planted link
                    # at the destination is replaced rather than traversed.
                    parent = _confined(ctx.workspace, str(rel.parent))
                    parent_rel = parent.relative_to(ctx.workspace.resolve())
                except ToolError as e:
                    errors.append(f"{fname}: {e}")
                    continue
                size = f.get("size")
                url = f.get("url") or f.get("download_url")
                if not url:
                    errors.append(f"{fname}: no download URL")
                    continue
                try:
                    with confined_directory(
                        ctx.workspace, parent_rel, create=True
                    ) as directory:
                        if (
                            size is not None
                            and directory.regular_size(rel.name) == size
                        ):
                            skipped.append(str(rel))
                            continue
                        await _download(
                            client,
                            str(url),
                            token,
                            base_url,
                            directory,
                            rel.name,
                        )
                        downloaded.append(str(rel))
                except ToolError as e:
                    errors.append(f"{fname}: {e}")
                except (OSError, PathEscapeError) as e:
                    errors.append(f"{fname}: {e}")

    summary = (
        f"Sync complete: {len(downloaded)} downloaded, "
        f"{len(skipped)} skipped, {len(errors)} errors."
    )
    detail = ""
    if downloaded:
        detail += "\nDownloaded:\n" + "\n".join(f"  + {d}" for d in downloaded[:50])
        if len(downloaded) > 50:
            detail += f"\n  … and {len(downloaded) - 50} more"
    if errors:
        detail += "\nErrors:\n" + "\n".join(f"  ! {e}" for e in errors[:20])
    return summary + detail


async def _download(
    client: httpx.AsyncClient,
    url: str,
    token: str,
    base_url: str,
    directory: ConfinedDirectory,
    dest_name: str,
) -> None:
    """Stream a Canvas file to a confined directory (bounded and atomic).

    The bearer token is attached only when ``url`` is on the Canvas origin:
    Canvas file URLs commonly point at signed S3/CDN endpoints that carry their
    own authorization, and forwarding the token there would leak it to a third
    party. The body streams into a ``.part`` sibling and is atomically renamed
    into place on success, so a failed/partial download never leaves a
    truncated file that a size-match resync would treat as complete.
    """
    headers = {"User-Agent": _USER_AGENT}
    if _same_origin(url, base_url):
        headers["Authorization"] = f"Bearer {token}"
    tmp_name = dest_name + ".part"
    try:
        written = 0
        # The .part name is predictable, so it must never be opened through a
        # pre-existing path. Every operation uses the already-validated parent
        # descriptor; replacing that parent's pathname with a symlink cannot
        # redirect the unlink, create, or final rename.
        directory.unlink(tmp_name, missing_ok=True)
        async with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            with directory.open_exclusive(tmp_name) as fh:
                async for chunk in resp.aiter_bytes(65536):
                    written += len(chunk)
                    if written > _MAX_FILE_BYTES:
                        raise ToolError(
                            f"file exceeds max size ({_MAX_FILE_BYTES} bytes)"
                        )
                    fh.write(chunk)
        directory.replace(tmp_name, dest_name)
    except ToolError:
        directory.unlink(tmp_name, missing_ok=True)
        raise
    except (httpx.HTTPError, OSError, PathEscapeError) as e:
        directory.unlink(tmp_name, missing_ok=True)
        raise ToolError(f"download failed: {e}") from None
