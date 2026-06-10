"""Tests for the Canvas skill's shipped tools (``lingcore/skills/canvas``).

The Canvas module is loaded the same way the framework loads it (importlib via
``load_skill_tools``), then exercised against a mocked Canvas API using
``httpx.MockTransport``.  ``canvas_tools`` does attribute access
``httpx.AsyncClient``, so patching that attribute injects the mock transport.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from lingcore.errors import ToolError
from lingcore.tools import ToolContext

import lingcore.tools.builtin  # noqa: F401

_CANVAS_PATH = (
    Path(__file__).parent.parent / "lingcore" / "skills" / "canvas" / "canvas_tools.py"
)


def _load_canvas():
    """Load the canvas skill the way the framework does — via ``load_skill_tools``
    — so its tools register exactly once under the canonical synthetic module
    name. A later ``from_profile`` that loads the teaching profile then
    re-imports the same path idempotently instead of colliding on the canvas
    tool names in the process-global REGISTRY (test isolation)."""
    from lingcore.skills import load_skills, load_skill_tools

    bundled = Path(__file__).parent.parent / "lingcore" / "skills"
    skills = load_skills([bundled])
    load_skill_tools({"canvas": skills["canvas"]})
    name = next(n for n in sys.modules if n.startswith("lingcore_skill_tools.canvas."))
    return sys.modules[name]


canvas = _load_canvas()


def _ctx(tmp_path, **canvas_opts):
    opts = {"base_url": "https://canvas.test", "token_env": "CANVAS_TEST_TOKEN"}
    opts.update(canvas_opts)
    return ToolContext(workspace=tmp_path, options={"canvas": opts})


def _patch_canvas(handler):
    """Patch httpx.AsyncClient so the Canvas client speaks to a MockTransport."""
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    return patch("httpx.AsyncClient", side_effect=factory)


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("CANVAS_TEST_TOKEN", "secret-token")


# --------------------------------------------------------------------------- #
# Config / credential handling                                                #
# --------------------------------------------------------------------------- #

async def test_missing_token_is_toolerror(tmp_path, monkeypatch):
    monkeypatch.delenv("CANVAS_TEST_TOKEN", raising=False)
    with pytest.raises(ToolError, match="token"):
        await canvas.canvas_courses(canvas.CoursesArgs(), _ctx(tmp_path))


async def test_missing_base_url_is_toolerror(tmp_path):
    ctx = ToolContext(workspace=tmp_path, options={"canvas": {"token_env": "CANVAS_TEST_TOKEN"}})
    with pytest.raises(ToolError, match="base_url|not configured"):
        await canvas.canvas_courses(canvas.CoursesArgs(), ctx)


async def test_network_error_is_toolerror(tmp_path):
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    with _patch_canvas(handler):
        with pytest.raises(ToolError, match="request failed|failed"):
            await canvas.canvas_courses(canvas.CoursesArgs(), _ctx(tmp_path))


async def test_bearer_token_is_sent(tmp_path):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=[])

    with _patch_canvas(handler):
        await canvas.canvas_courses(canvas.CoursesArgs(), _ctx(tmp_path))
    assert seen["auth"] == "Bearer secret-token"


# --------------------------------------------------------------------------- #
# Pagination + course filter                                                  #
# --------------------------------------------------------------------------- #

async def test_courses_pagination_follows_next(tmp_path):
    def handler(request):
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[{"id": 2, "name": "Chem", "course_code": "CHEM"}])
        return httpx.Response(
            200,
            json=[{"id": 1, "name": "Bio", "course_code": "BIO"}],
            headers={"Link": '<https://canvas.test/api/v1/courses?page=2>; rel="next"'},
        )

    with _patch_canvas(handler):
        out = await canvas.canvas_courses(canvas.CoursesArgs(), _ctx(tmp_path))
    assert "Bio" in out and "Chem" in out  # both pages collected


async def test_course_ids_filter(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json=[{"id": 1, "name": "Bio"}, {"id": 2, "name": "Chem"}],
        )

    with _patch_canvas(handler):
        out = await canvas.canvas_courses(canvas.CoursesArgs(), _ctx(tmp_path, course_ids=[2]))
    assert "Chem" in out and "Bio" not in out


# --------------------------------------------------------------------------- #
# Assignments                                                                 #
# --------------------------------------------------------------------------- #

async def test_assignments_drops_unpublished_and_sorts(tmp_path):
    def handler(request):
        if request.url.path.endswith("/courses"):
            return httpx.Response(200, json=[{"id": 1, "name": "Bio"}])
        if request.url.path.endswith("/assignments"):
            return httpx.Response(200, json=[
                {"id": 9, "name": "Hidden", "published": False, "due_at": "2026-01-01"},
                {"id": 10, "name": "Later", "published": True, "due_at": "2026-12-01", "points_possible": 5},
                {"id": 11, "name": "Sooner", "published": True, "due_at": "2026-06-01"},
            ])
        return httpx.Response(404)

    with _patch_canvas(handler):
        out = await canvas.canvas_assignments(canvas.AssignmentsArgs(), _ctx(tmp_path))
    assert "Hidden" not in out
    assert "5 pts" in out
    # Sorted by due date: Sooner (2026-06) before Later (2026-12).
    assert out.index("Sooner") < out.index("Later")


# --------------------------------------------------------------------------- #
# Sync: download, skip-if-unchanged, confinement                              #
# --------------------------------------------------------------------------- #

def test_confined_rejects_escape(tmp_path):
    with pytest.raises(ToolError, match="escapes workspace"):
        canvas._confined(tmp_path, "../escape.pdf")
    inside = canvas._confined(tmp_path, "canvas/x.pdf")
    assert str(inside).startswith(str(tmp_path.resolve()))


async def test_sync_downloads_then_skips(tmp_path):
    body = b"%PDF-1.4 fake"

    def handler(request):
        p = request.url.path
        if p.endswith("/courses"):
            return httpx.Response(200, json=[{"id": 1, "name": "Bio"}])
        if p.endswith("/folders"):
            return httpx.Response(200, json=[])
        if p.endswith("/files"):
            return httpx.Response(200, json=[
                {"id": 7, "display_name": "lecture.pdf", "folder_id": None,
                 "size": len(body), "url": "https://files.test/lecture.pdf"},
                {"id": 8, "display_name": "skip.txt", "folder_id": None,
                 "size": 3, "url": "https://files.test/skip.txt"},
            ])
        if "lecture.pdf" in p:
            return httpx.Response(200, content=body)
        return httpx.Response(404)

    ctx = _ctx(tmp_path, download_dir="canvas")
    with _patch_canvas(handler):
        first = await canvas.canvas_sync(canvas.SyncArgs(), ctx)
    dest = tmp_path / "canvas" / "Bio" / "lecture.pdf"
    assert dest.is_file() and dest.read_bytes() == body
    assert "1 downloaded" in first
    # .txt filtered out (not in default file_ext).
    assert not (tmp_path / "canvas" / "Bio" / "skip.txt").exists()

    # Re-sync: size matches → skipped, not re-downloaded.
    with _patch_canvas(handler):
        second = await canvas.canvas_sync(canvas.SyncArgs(), ctx)
    assert "0 downloaded" in second and "1 skipped" in second


async def test_sync_reports_download_error(tmp_path):
    def handler(request):
        p = request.url.path
        if p.endswith("/courses"):
            return httpx.Response(200, json=[{"id": 1, "name": "Bio"}])
        if p.endswith("/folders"):
            return httpx.Response(200, json=[])
        if p.endswith("/files"):
            return httpx.Response(200, json=[
                {"id": 7, "display_name": "broken.pdf", "folder_id": None,
                 "size": 10, "url": "https://files.test/broken.pdf"},
            ])
        return httpx.Response(500)  # the download fails

    with _patch_canvas(handler):
        out = await canvas.canvas_sync(canvas.SyncArgs(), _ctx(tmp_path))
    assert "1 errors" in out and "broken.pdf" in out
    # A failed download leaves no partial file behind.
    assert not (tmp_path / "canvas" / "Bio" / "broken.pdf").exists()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def test_next_link_parsing():
    hdr = '<https://c/api/v1/x?page=2>; rel="next", <https://c/api/v1/x?page=9>; rel="last"'
    assert canvas._next_link(hdr) == "https://c/api/v1/x?page=2"
    assert canvas._next_link('<https://c/x>; rel="last"') is None


def test_folder_path_reconstruction():
    folders = {
        1: {"id": 1, "name": "course files", "parent_folder_id": None},
        2: {"id": 2, "name": "Week 1", "parent_folder_id": 1},
        3: {"id": 3, "name": "Slides", "parent_folder_id": 2},
    }
    assert str(canvas._folder_path(folders[3], folders)) == "Week 1/Slides"
    assert str(canvas._folder_path(folders[1], folders)) == "."


# --------------------------------------------------------------------------- #
# End-to-end: the real teaching profile builds in this (already-polluted) proc #
# --------------------------------------------------------------------------- #

async def test_teaching_profile_loads_without_collision(tmp_path, monkeypatch):
    # This module already registered the canvas tools at import time (the
    # ``canvas = _load_canvas()`` above). Building the *real* teaching profile
    # must still succeed: the framework re-imports the same canvas module path
    # idempotently rather than colliding on its tool names in the global
    # REGISTRY. Doubles as the end-to-end check that the teaching profile
    # authorizes the canvas tools (their code is loaded before subset()).
    from lingcore.agent import Agent
    from lingcore.config import AgentProfile
    from tests.fakes import FakeLLMClient, ScriptedTurn

    monkeypatch.setenv("LLY_API_KEY", "dummy")
    prof = AgentProfile.load(
        Path(__file__).parent.parent / "profiles" / "teaching"
    )
    # Keep the auto-created default workspace out of the real profile dir.
    prof.workspace = str(tmp_path)
    agent = Agent.from_profile(prof, llm=FakeLLMClient([ScriptedTurn(text="hi")]))
    assert {
        "canvas_courses",
        "canvas_assignments",
        "canvas_announcements",
        "canvas_sync",
    } <= set(agent.tools.names())
