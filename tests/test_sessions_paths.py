"""Sessions tests: paths."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lingcore.config import AgentProfile, LLMCfg
from lingcore.errors import ConfigError
from lingcore.message import Message
from lingcore.sessions import (
    new_session_id,
    open_store,
)
from tests.support.sessions import PROFILE_YAML, _profile
from tests.support.sessions import store as store


def test_open_store_default_path(tmp_path: Path):
    profile = _profile(tmp_path)
    store, notice = open_store(profile)
    assert notice is None
    with store:
        assert store.db_path == (tmp_path / "prof" / "sessions.db").resolve()


def test_open_store_disabled(tmp_path: Path):
    yaml_text = PROFILE_YAML + "sessions:\n  enabled: false\n"
    store, notice = open_store(_profile(tmp_path, yaml_text))
    assert store is None and notice is None


def test_open_store_rejects_escape(tmp_path: Path):
    yaml_text = PROFILE_YAML + "sessions:\n  path: ../outside.db\n"
    with pytest.raises(ConfigError, match="escapes profile directory"):
        open_store(_profile(tmp_path, yaml_text))


def test_open_store_rejects_bare_absolute(tmp_path: Path):
    yaml_text = PROFILE_YAML + f"sessions:\n  path: {tmp_path / 'abs.db'}\n"
    with pytest.raises(ConfigError, match="allow_absolute_path"):
        open_store(_profile(tmp_path, yaml_text))


def test_open_store_absolute_with_flag(tmp_path: Path):
    yaml_text = PROFILE_YAML + (
        f"sessions:\n  path: {tmp_path / 'elsewhere' / 'abs.db'}\n"
        "  allow_absolute_path: true\n"
    )
    store, notice = open_store(_profile(tmp_path, yaml_text))
    assert notice is None
    with store:
        sid = new_session_id()
        store.append(sid, Message.user("hi"))
        assert (tmp_path / "elsewhere" / "abs.db").is_file()


def test_open_store_refuses_package_tree(tmp_path: Path, monkeypatch):
    import lingcore.sessions as sessions_mod

    monkeypatch.setattr(sessions_mod, "_PACKAGE_DIR", tmp_path.resolve())
    store, notice = open_store(_profile(tmp_path))
    assert store is None
    assert "inside the installed" in notice


def test_open_store_absolute_path_refuses_package_tree(tmp_path: Path, monkeypatch):
    import lingcore.sessions as sessions_mod

    monkeypatch.setattr(sessions_mod, "_PACKAGE_DIR", tmp_path.resolve())
    target = tmp_path / "inside-package.db"
    yaml_text = PROFILE_YAML + (
        f"sessions:\n  path: {target}\n  allow_absolute_path: true\n"
    )
    store, notice = open_store(_profile(tmp_path, yaml_text))
    assert store is None
    assert "inside the installed" in notice
    assert not target.exists()


def test_bundled_profiles_live_outside_package_and_persist(tmp_path: Path):
    import lingcore.sessions as sessions_mod

    repo_profiles = Path(__file__).parent.parent / "profiles"
    assert not repo_profiles.resolve().is_relative_to(sessions_mod._PACKAGE_DIR)
    copy = tmp_path / "daily"
    shutil.copytree(repo_profiles / "daily", copy)
    store, notice = open_store(AgentProfile.load(copy))
    assert store is not None and notice is None
    store.close()


def test_open_store_without_source_dir():
    profile = AgentProfile(llm=LLMCfg(model="m"))
    store, notice = open_store(profile)
    assert store is None and "no source directory" in notice


def test_sessions_cfg_typo_is_loud(tmp_path: Path):
    yaml_text = PROFILE_YAML + "sessions:\n  enabld: true\n"
    with pytest.raises(ConfigError, match="invalid profile"):
        _profile(tmp_path, yaml_text)
