"""Writable profiles initialized from immutable package templates."""

from __future__ import annotations

import pytest

from lingcore.__main__ import main
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.profiles import (
    PROFILE_TEMPLATE_FILES,
    default_profile_path,
    initialize_profile,
)


def test_initialize_profile_copies_only_declared_template_files(tmp_path):
    destination = tmp_path / "local-agent"
    initialized = initialize_profile("coding_ollama", destination=destination)

    assert initialized == destination
    assert {path.name for path in destination.iterdir()} == {"config.yaml"}
    assert not (destination / "sessions.db").exists()
    assert not (destination / "workspace").exists()
    assert AgentProfile.load(destination).name == "coding_ollama"


def test_initialize_profile_never_overwrites_existing_directory(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("mine", encoding="utf-8")

    with pytest.raises(ConfigError, match="already exists"):
        initialize_profile("coding", destination=destination)
    assert sentinel.read_text(encoding="utf-8") == "mine"


def test_default_profile_uses_writable_state_outside_a_checkout(tmp_path, monkeypatch):
    import lingcore.profiles as profiles_module

    monkeypatch.setattr(profiles_module, "_REPO_PROFILE_ROOT", tmp_path / "missing")
    path = default_profile_path({"LINGCORE_STATE_HOME": str(tmp_path / "state")})
    assert path == tmp_path / "state" / "profiles" / "coding"


def test_profile_cli_lists_and_initializes(tmp_path, capsys):
    assert main(["profile", "list"]) == 0
    listed = capsys.readouterr().out
    assert set(PROFILE_TEMPLATE_FILES) <= set(listed.split())

    destination = tmp_path / "ollama"
    assert (
        main(
            [
                "profile",
                "init",
                "coding_ollama",
                "--destination",
                str(destination),
            ]
        )
        == 0
    )
    assert (destination / "config.yaml").is_file()
    assert "Initialized 'coding_ollama'" in capsys.readouterr().out


def test_main_help_advertises_profile_management(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    assert "{doctor,telegram,profile}" in capsys.readouterr().out
