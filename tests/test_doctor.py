"""Offline profile-doctor and shipped ``.env.example`` coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

from lingcore.__main__ import main
from lingcore.config import AgentProfile
from lingcore.doctor import diagnose_profile, required_example_names

REPO_ROOT = Path(__file__).parent.parent


def _profile(tmp_path: Path, body: str) -> Path:
    root = tmp_path / "profile"
    root.mkdir()
    (root / "config.yaml").write_text(body, encoding="utf-8")
    return root


def test_doctor_passes_without_side_effects_or_secret_output(
    tmp_path, monkeypatch, capsys
):
    root = _profile(
        tmp_path,
        """
name: healthy
llm:
  model: test-model
  api_key_env: DOCTOR_TEST_KEY
""",
    )
    (root / ".env").write_text(
        "DOCTOR_TEST_KEY=profile-secret-sentinel\n", encoding="utf-8"
    )
    (root / ".env.example").write_text("DOCTOR_TEST_KEY=\n", encoding="utf-8")
    monkeypatch.setenv("DOCTOR_TEST_KEY", "process-secret-sentinel")

    before = {path.name for path in root.iterdir()}
    assert main(["doctor", "--profile", str(root)]) == 0
    after = {path.name for path in root.iterdir()}

    out = capsys.readouterr().out
    assert "doctor passed" in out
    assert "DOCTOR_TEST_KEY is set by profile .env" in out
    assert "profile-secret-sentinel" not in out
    assert "process-secret-sentinel" not in out
    assert after == before
    assert not (root / "sessions.db").exists()
    assert not (root / "workspace").exists()


def test_doctor_reports_missing_required_variable(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DOCTOR_MISSING_KEY", raising=False)
    root = _profile(
        tmp_path,
        """
name: missing
llm:
  model: test-model
  api_key_env: DOCTOR_MISSING_KEY
""",
    )
    (root / ".env.example").write_text("DOCTOR_MISSING_KEY=\n", encoding="utf-8")

    assert main(["--profile", str(root), "doctor"]) == 1

    out = capsys.readouterr().out
    assert "DOCTOR_MISSING_KEY is missing" in out
    assert "doctor found 1 error" in out


def test_doctor_honors_empty_profile_override(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCTOR_EMPTY_KEY", "ambient-secret-sentinel")
    root = _profile(
        tmp_path,
        """
name: empty
llm:
  model: test-model
  api_key_env: DOCTOR_EMPTY_KEY
""",
    )
    (root / ".env").write_text("DOCTOR_EMPTY_KEY=\n", encoding="utf-8")
    (root / ".env.example").write_text("DOCTOR_EMPTY_KEY=\n", encoding="utf-8")

    assert main(["doctor", "-p", str(root)]) == 1

    out = capsys.readouterr().out
    assert "DOCTOR_EMPTY_KEY is empty in profile .env" in out
    assert "ambient-secret-sentinel" not in out


def test_doctor_keeps_grep_offline_and_keyless(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    root = _profile(
        tmp_path,
        """
name: offline
llm:
  model: local-model
tools: [knowledge]
tool_options:
  knowledge:
    backend: grep
    embedding:
      enabled: false
""",
    )

    assert main(["doctor", "-p", str(root)]) == 0

    out = capsys.readouterr().out
    assert "doctor passed" in out
    assert "SILICONFLOW_API_KEY" not in out
    assert "no .env.example is required" in out


def test_doctor_rejects_invalid_search_options_at_load(tmp_path, capsys):
    root = _profile(
        tmp_path,
        """
name: invalid-search
llm:
  model: local-model
tools: [search]
tool_options:
  search:
    max_hitz: 10
""",
    )

    assert main(["doctor", "-p", str(root)]) == 2

    err = capsys.readouterr().err
    assert "tool_options.search.max_hitz is not supported" in err


def test_doctor_reports_canvas_configuration(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CANVAS_TOKEN", raising=False)
    root = _profile(
        tmp_path,
        """
name: canvas-test
llm:
  model: local-model
skills: [canvas]
tools: []
tool_options:
  canvas:
    base_url: ""
    token_env: CANVAS_TOKEN
""",
    )
    (root / ".env.example").write_text(
        "CANVAS_URL=https://school.example\nCANVAS_TOKEN=\n", encoding="utf-8"
    )

    assert main(["doctor", "-p", str(root)]) == 1

    out = capsys.readouterr().out
    assert "Canvas base_url is empty" in out
    assert "CANVAS_TOKEN is missing" in out
    assert ".env.example documents 2 variable(s)" in out


def test_every_shipped_profile_that_needs_env_has_complete_safe_example(
    monkeypatch,
):
    import lingcore.config as config_module

    # A developer may have real ignored files under profiles/. Template tests
    # must inspect repository defaults without reading or depending on them.
    monkeypatch.setattr(config_module, "_load_profile_env", lambda _: {})
    for name in (
        "LINGCORE_API_KEY_ENV",
        "LINGCORE_BASE_URL",
        "LINGCORE_MODEL",
        "LINGCORE_WORKSPACE",
        "LLY_API_KEY",
        "DEEPSEEK_API_KEY",
        "CANVAS_URL",
        "CANVAS_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    expected = {
        "coding": {"LLY_API_KEY"},
        "coding_ollama": set(),
        "daily": {"DEEPSEEK_API_KEY"},
        "teaching": {"LLY_API_KEY", "CANVAS_URL", "CANVAS_TOKEN"},
    }
    found = {path.name for path in (REPO_ROOT / "profiles").iterdir() if path.is_dir()}
    assert found == set(expected)

    for profile_name, names in expected.items():
        root = REPO_ROOT / "profiles" / profile_name
        profile = AgentProfile.load(root)
        assert required_example_names(profile) == names
        example_path = root / ".env.example"
        if not names:
            assert not example_path.exists()
            continue
        assert example_path.is_file()
        parsed = dotenv_values(example_path, interpolate=False)
        assert names <= set(parsed)
        for name in names:
            if name.endswith(("KEY", "TOKEN")):
                assert parsed[name] == ""


def test_canvas_skill_example_remains_secret_free_and_complete():
    path = REPO_ROOT / "lingcore" / "skills" / "canvas" / ".env.example"
    parsed = dotenv_values(path, interpolate=False)
    assert {"CANVAS_URL", "CANVAS_TOKEN"} <= set(parsed)
    assert parsed["CANVAS_TOKEN"] == ""


def test_doctor_probes_bubblewrap_and_tracks_passed_environment(monkeypatch):
    import lingcore.doctor as doctor_module

    monkeypatch.setattr(doctor_module.sys, "platform", "linux")
    monkeypatch.setattr(doctor_module, "_probe", lambda argv, timeout=5: True)
    monkeypatch.setenv("SANDBOX_BUILD_TOKEN", "secret-not-for-output")
    profile = AgentProfile.model_validate(
        {
            "name": "sandboxed",
            "llm": {"model": "test"},
            "tools": ["run_shell"],
            "tool_options": {
                "run_shell": {
                    "sandbox": {
                        "backend": "bubblewrap",
                        "executable": sys.executable,
                        "pass_env": ["SANDBOX_BUILD_TOKEN"],
                    }
                }
            },
        }
    )

    report = diagnose_profile(profile)
    messages = "\n".join(finding.message for finding in report.findings)
    assert not report.errors
    assert "Bubblewrap isolation probe passed" in messages
    assert "SANDBOX_BUILD_TOKEN is set by process environment" in messages
    assert "secret-not-for-output" not in messages


def test_doctor_reports_shell_options_mutated_after_validation():
    profile = AgentProfile.model_validate(
        {
            "name": "mutated-shell-options",
            "llm": {"model": "test"},
            "tools": ["run_shell"],
        }
    )
    profile.tool_options["run_shell"] = {"timeout": 0}

    report = diagnose_profile(profile)

    assert any(
        "invalid tool_options.run_shell" in finding.message for finding in report.errors
    )


def test_doctor_reports_search_options_mutated_after_validation():
    profile = AgentProfile.model_validate(
        {
            "name": "mutated-search-options",
            "llm": {"model": "test"},
            "tools": ["search"],
        }
    )
    profile.tool_options["search"] = {"max_hitz": 10}

    report = diagnose_profile(profile)

    assert any(
        "tool_options.search.max_hitz is not supported" in finding.message
        for finding in report.errors
    )


def test_doctor_checks_oci_image_without_pulling(monkeypatch):
    import lingcore.doctor as doctor_module

    calls: list[list[str]] = []

    def fake_probe(argv: list[str], *, timeout: float = 5) -> bool:
        calls.append(argv)
        return argv[1] == "info"

    monkeypatch.setattr(doctor_module, "_probe", fake_probe)
    profile = AgentProfile.model_validate(
        {
            "name": "containerized",
            "llm": {"model": "test"},
            "tools": ["run_shell"],
            "tool_options": {
                "run_shell": {
                    "sandbox": {
                        "backend": "oci",
                        "runtime": "docker",
                        "executable": sys.executable,
                        "image": "example.test/build:fixed",
                        "pull": "missing",
                        "container_os": "linux",
                        "user": "1000:1000",
                        "resources": {
                            "cpus": 1,
                            "memory_mb": 128,
                            "pids": 32,
                            "ephemeral_storage_mb": 64,
                        },
                    }
                }
            },
        }
    )

    report = diagnose_profile(profile)
    messages = "\n".join(finding.message for finding in report.findings)
    assert not report.errors
    assert "execution will pull it" in messages
    assert any(argv[1:3] == ["image", "inspect"] for argv in calls)
    assert all("pull" not in argv for argv in calls)


def test_doctor_never_executes_backend_from_writable_workspace(tmp_path, monkeypatch):
    import lingcore.doctor as doctor_module

    executable = tmp_path / "fake-bwrap"
    marker = tmp_path / "doctor-executed-backend"
    executable.write_text(
        f"#!/bin/sh\ntouch {marker}\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(doctor_module.sys, "platform", "linux")
    monkeypatch.setattr(
        doctor_module,
        "_probe",
        lambda *args, **kwargs: pytest.fail("workspace executable was probed"),
    )
    profile = AgentProfile.model_validate(
        {
            "name": "unsafe-doctor",
            "workspace": str(tmp_path),
            "llm": {"model": "test"},
            "tools": ["run_shell"],
            "tool_options": {
                "run_shell": {
                    "sandbox": {
                        "backend": "bubblewrap",
                        "executable": str(executable),
                    }
                }
            },
        }
    )

    report = diagnose_profile(profile)
    assert any("writable workspace" in finding.message for finding in report.errors)
    assert not marker.exists()


def test_doctor_rejects_oci_image_declared_volumes(monkeypatch):
    import lingcore.doctor as doctor_module

    def fake_probe(argv: list[str], *, timeout: float = 5) -> bool:
        return argv[1] == "info" or argv[1:3] == ["image", "inspect"]

    monkeypatch.setattr(doctor_module, "_probe", fake_probe)
    monkeypatch.setattr(
        doctor_module,
        "_probe_output",
        lambda argv, timeout=5: (True, b'linux\n{"/data": {}}\n'),
    )
    profile = AgentProfile.model_validate(
        {
            "name": "volume-image",
            "llm": {"model": "test"},
            "tools": ["run_shell"],
            "tool_options": {
                "run_shell": {
                    "sandbox": {
                        "backend": "oci",
                        "runtime": "docker",
                        "executable": sys.executable,
                        "image": "example.test/volume:fixed",
                        "pull": "never",
                        "container_os": "linux",
                        "user": "1000:1000",
                        "resources": {
                            "cpus": 1,
                            "memory_mb": 128,
                            "pids": 32,
                            "ephemeral_storage_mb": 64,
                        },
                    }
                }
            },
        }
    )

    report = diagnose_profile(profile)
    assert any(
        "unsupported writable volumes" in finding.message for finding in report.errors
    )
