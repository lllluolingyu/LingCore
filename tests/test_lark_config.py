"""Lark configuration, CLI aliases and SDK-free diagnostics."""

from __future__ import annotations

import builtins

import pytest

from lingcore.__main__ import _parse_args, main
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.integrations.lark.config import load_lark_config
from tests.support.lark_bridge import CONFIG, PROFILE, loaded


def test_credentials_use_profile_env_and_hide_values(tmp_path, monkeypatch):
    loaded(tmp_path)
    root = tmp_path / "profile"
    (root / ".env").write_text("LARK_APP_ID=cli_test\nLARK_APP_SECRET=profile-secret\n")
    monkeypatch.setenv("LARK_APP_SECRET", "process-secret")
    config = load_lark_config(root)
    assert config.resolve_credentials() == ("cli_test", "profile-secret")
    assert "profile-secret" not in repr(config)
    assert "profile-secret" not in config.model_dump_json()
    (root / ".env").write_text("LARK_APP_ID=cli_test\nLARK_APP_SECRET=\n")
    with pytest.raises(ConfigError, match="LARK_APP_SECRET"):
        load_lark_config(root)


@pytest.mark.parametrize(
    "extra",
    [
        "app_secret: SENTINEL",
        "app_id_env: 'SENTINEL!'",
        "domain: SENTINEL",
        "confirmation_timeout: SENTINEL",
        "unknown: SENTINEL",
    ],
)
def test_errors_never_echo_values(tmp_path, extra):
    with pytest.raises(ConfigError) as caught:
        loaded(tmp_path, config_text=CONFIG + extra)
    assert "SENTINEL" not in str(caught.value)


@pytest.mark.parametrize(
    "ids", ["[]", "[true]", "[123]", "['']", "['../escape']", "['*']"]
)
def test_reject_invalid_user_ids(tmp_path, ids):
    with pytest.raises(ConfigError, match="allowed_user_ids"):
        loaded(tmp_path, config_text=f"allowed_user_ids: {ids}\n")


def test_domains_dedup_and_state_confinement(tmp_path):
    _, config = loaded(tmp_path, config_text=CONFIG + "domain: lark\n")
    assert config.api_domain == "https://open.larksuite.com"
    _, config = loaded(tmp_path, config_text="allowed_user_ids: [ou_a, ou_a]\n")
    assert config.allowed_user_ids == ["ou_a"]
    assert config.api_domain == "https://open.feishu.cn"
    for path in ("../escape", str(tmp_path / "absolute")):
        with pytest.raises(ConfigError):
            loaded(tmp_path, config_text=CONFIG + f"state_dir: {path}\n")
    _, config = loaded(
        tmp_path,
        config_text=CONFIG
        + f"state_dir: {tmp_path / 'absolute'}\nallow_absolute_state_dir: true\n",
    )
    assert config.state_path == tmp_path / "absolute"


def test_reject_symlink_state_escape_and_package_state(tmp_path):
    loaded(tmp_path)
    (tmp_path / "profile" / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ConfigError, match="escapes"):
        loaded(tmp_path, config_text=CONFIG + "state_dir: outside/state\n")
    from pathlib import Path

    import lingcore

    with pytest.raises(ConfigError, match="package tree"):
        loaded(
            tmp_path,
            config_text=CONFIG
            + f"state_dir: {Path(lingcore.__file__).parent}\nallow_absolute_state_dir: true\n",
        )


@pytest.mark.parametrize(
    "body",
    [
        PROFILE + "sessions:\n  enabled: false\n",
        PROFILE.replace("[memory]", "[process_start]"),
        PROFILE.replace("[memory]", "[run_shell]")
        + "tool_options:\n  run_shell:\n    require_confirmation: false\n",
        PROFILE.replace("[memory]", "[run_shell]")
        + "tool_options:\n  run_shell:\n    allow_patterns: ['ls*']\n",
    ],
)
def test_reject_unattended_profile_policies(tmp_path, body):
    with pytest.raises(ConfigError, match="Lark"):
        loaded(tmp_path, profile_text=body)


@pytest.mark.parametrize("alias", ["lark", "feishu"])
def test_cli_aliases_route_to_runner(monkeypatch, alias):
    calls = []
    monkeypatch.setattr(
        "lingcore.integrations.lark.application.run_lark",
        lambda *a, **kw: calls.append((a, kw)) or 0,
    )
    assert main([alias, "-p", "/profile", "--feishu-config", "/config.yaml"]) == 0
    assert calls == [(("/profile",), {"lark_config_path": "/config.yaml"})]


@pytest.mark.parametrize(
    "args",
    [
        ["lark", "--continue"],
        ["feishu", "--resume", "abc"],
        ["lark", "--no-session"],
        ["lark", "--workspace", "/tmp"],
        ["lark", "--list-sessions"],
        ["lark", "--telegram-config", "foo"],
        ["telegram", "--lark-config", "foo"],
        ["--lark-config", "foo"],
        ["feishu", "--telegram-mode", "polling"],
    ],
)
def test_cli_rejects_incompatible_flags(args):
    with pytest.raises(SystemExit) as caught:
        _parse_args(args)
    assert caught.value.code == 2


def test_doctor_and_config_work_without_optional_sdk(tmp_path, monkeypatch, capsys):
    loaded(tmp_path)
    original = builtins.__import__

    def without_sdk(name, *args, **kwargs):
        if name == "lark_channel" or name.startswith("lark_channel."):
            raise ModuleNotFoundError("missing", name="lark_channel")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    root = tmp_path / "profile"
    assert (
        main(["doctor", "-p", str(root), "--lark-config", str(root / "lark.yaml")]) == 1
    )
    report = capsys.readouterr()
    assert "LARK_APP_ID" in report.out and "LARK_APP_SECRET" in report.out
    assert main(["lark", "-p", str(root)]) == 2
    assert 'pip install "lingcore[lark]"' in capsys.readouterr().err


def test_config_expansion_uses_selected_profile(tmp_path, monkeypatch):
    loaded(tmp_path)
    root = tmp_path / "profile"
    (root / ".env").write_text(
        "APP_ID_NAME=CUSTOM_ID\nCUSTOM_ID=cli_one\nCUSTOM_SECRET=secret\n"
    )
    (root / "lark.yaml").write_text(
        CONFIG + "app_id_env: ${APP_ID_NAME}\napp_secret_env: CUSTOM_SECRET\n"
    )
    monkeypatch.chdir(tmp_path)
    config = load_lark_config(AgentProfile.load(root))
    assert config.resolve_credentials() == ("cli_one", "secret")
