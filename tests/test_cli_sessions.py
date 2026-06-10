"""CLI composition-root behavior for session flags (list/continue/resume)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.__main__ import main
from lingcore.io.cli import CLIFrontend
from lingcore.message import Message
from lingcore.sessions import SessionStore

PROFILE_YAML = """
name: clitest
workspace: .
llm:
  model: test-model
  base_url: http://localhost:11434/v1
tools: []
"""

SID_A = "aa" + "0" * 30
SID_B = "ab" + "0" * 30


def _write_profile(tmp_path: Path) -> Path:
    d = tmp_path / "prof"
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(PROFILE_YAML, encoding="utf-8")
    return d


@pytest.fixture
def no_input(monkeypatch):
    """End the interactive session immediately (EOF) without touching the LLM."""

    async def _none(self):
        return None

    monkeypatch.setattr(CLIFrontend, "read_input", _none)


def test_list_sessions_empty(tmp_path, capsys):
    d = _write_profile(tmp_path)
    assert main(["-p", str(d), "--list-sessions"]) == 0
    assert "no stored sessions" in capsys.readouterr().out


def test_list_sessions_populated(tmp_path, capsys):
    d = _write_profile(tmp_path)
    with SessionStore(d / "sessions.db") as store:
        store.append(SID_A, Message.user("hello world"))
    assert main(["-p", str(d), "--list-sessions"]) == 0
    out = capsys.readouterr().out
    assert SID_A[:8] in out and "hello world" in out


def test_list_sessions_on_in_package_profile_prints_notice(tmp_path, monkeypatch, capsys):
    import lingcore.sessions as sessions_mod

    d = _write_profile(tmp_path)
    monkeypatch.setattr(sessions_mod, "_PACKAGE_DIR", tmp_path.resolve())
    assert main(["-p", str(d), "--list-sessions"]) == 0
    assert "inside the installed" in capsys.readouterr().out


def test_continue_with_no_sessions_fails(tmp_path, capsys):
    d = _write_profile(tmp_path)
    assert main(["-p", str(d), "-c"]) == 2
    assert "no sessions to continue" in capsys.readouterr().err


def test_continue_with_sessions_disabled_fails(tmp_path, capsys):
    d = _write_profile(tmp_path)
    (d / "config.yaml").write_text(
        PROFILE_YAML + "sessions:\n  enabled: false\n", encoding="utf-8"
    )
    assert main(["-p", str(d), "-c"]) == 2
    assert "cannot resume" in capsys.readouterr().err


def test_resume_ambiguous_prefix(tmp_path, capsys):
    d = _write_profile(tmp_path)
    with SessionStore(d / "sessions.db") as store:
        store.append(SID_A, Message.user("one"))
        store.append(SID_B, Message.user("two"))
    assert main(["-p", str(d), "--resume", "a"]) == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err and SID_A[:8] in err and SID_B[:8] in err


def test_resume_unknown_prefix(tmp_path, capsys):
    d = _write_profile(tmp_path)
    with SessionStore(d / "sessions.db") as store:
        store.append(SID_A, Message.user("one"))
    assert main(["-p", str(d), "--resume", "ff"]) == 2
    assert "no session matching" in capsys.readouterr().err


def test_resume_shows_banner_and_replay(tmp_path, capsys, no_input):
    d = _write_profile(tmp_path)
    with SessionStore(d / "sessions.db") as store:
        store.append(SID_A, Message.user("the question"))
        store.append(SID_A, Message.assistant(content="the answer"))

    assert main(["-p", str(d), "--resume", "aa"]) == 0
    out = capsys.readouterr().out
    assert "resumed" in out and SID_A[:8] in out
    assert "the question" in out and "the answer" in out
    assert "saved" in out  # the resumed session has rows, so the hint prints


def test_default_run_immediate_exit_leaves_no_rows(tmp_path, capsys, no_input):
    d = _write_profile(tmp_path)
    assert main(["-p", str(d)]) == 0
    with SessionStore(d / "sessions.db") as store:
        assert store.list() == []  # rows are lazy: nothing said, nothing stored
    assert "saved" not in capsys.readouterr().out


def test_no_session_flag_skips_store(tmp_path, capsys, no_input):
    d = _write_profile(tmp_path)
    assert main(["-p", str(d), "--no-session"]) == 0
    assert not (d / "sessions.db").exists()


def test_resume_conflicts_with_no_session(tmp_path):
    d = _write_profile(tmp_path)
    with pytest.raises(SystemExit):
        main(["-p", str(d), "--resume", "aa", "--no-session"])
