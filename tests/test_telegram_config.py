"""Offline Telegram configuration, CLI, and doctor coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.__main__ import main
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.integrations.telegram.config import load_telegram_config

PROFILE = """
name: telegram-test
llm:
  model: test-model
  base_url: http://localhost:11434/v1
tools: []
"""


def _profile(tmp_path: Path, *, suffix: str = "") -> tuple[Path, AgentProfile]:
    root = tmp_path / "profile"
    root.mkdir()
    (root / "config.yaml").write_text(PROFILE + suffix, encoding="utf-8")
    return root, AgentProfile.load(root)


def _telegram(root: Path, body: str) -> Path:
    path = root / "telegram.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_config_uses_profile_env_precedence_and_deduplicates(tmp_path, monkeypatch):
    root, _ = _profile(tmp_path)
    (root / ".env").write_text(
        "TOKEN_NAME=PROFILE_TOKEN\nPROFILE_TOKEN=profile-secret\n",
        encoding="utf-8",
    )
    profile = AgentProfile.load(root)
    monkeypatch.setenv("PROFILE_TOKEN", "process-secret")
    path = _telegram(
        root,
        """
token_env: ${TOKEN_NAME}
allowed_user_ids: [7, 7, 8]
""",
    )

    config = load_telegram_config(profile, path)

    assert config.allowed_user_ids == [7, 8]
    assert config.resolve_token() == "profile-secret"
    assert "profile-secret" not in repr(config)


def test_config_is_strict_and_does_not_echo_literal_secret(tmp_path):
    root, profile = _profile(tmp_path)
    sentinel = "literal-secret-sentinel"
    path = _telegram(
        root,
        f"""
token: {sentinel}
token_env: TELEGRAM_TOKEN
allowed_user_ids: [1]
""",
    )
    with pytest.raises(ConfigError) as caught:
        load_telegram_config(profile, path, require_secrets=False)
    assert sentinel not in str(caught.value)
    assert "literal secrets" in str(caught.value)


@pytest.mark.parametrize(
    "users",
    ("[]", "[0]", "[-1]", "[true]", '["123"]'),
)
def test_allowed_user_ids_are_nonempty_positive_strict_ints(
    tmp_path, users
):
    root, profile = _profile(tmp_path)
    path = _telegram(
        root,
        f"token_env: TELEGRAM_TOKEN\nallowed_user_ids: {users}\n",
    )
    with pytest.raises(ConfigError, match="allowed_user_ids"):
        load_telegram_config(profile, path, require_secrets=False)


def test_relative_state_may_not_escape_profile(tmp_path):
    root, profile = _profile(tmp_path)
    path = _telegram(
        root,
        """
token_env: TELEGRAM_TOKEN
allowed_user_ids: [1]
state_dir: ../escape
""",
    )
    with pytest.raises(ConfigError, match="escapes"):
        load_telegram_config(profile, path, require_secrets=False)


def test_absolute_state_requires_explicit_consent(tmp_path):
    root, profile = _profile(tmp_path)
    state = tmp_path / "state"
    path = _telegram(
        root,
        f"""
token_env: TELEGRAM_TOKEN
allowed_user_ids: [1]
state_dir: {state}
""",
    )
    with pytest.raises(ConfigError, match="allow_absolute_state_dir"):
        load_telegram_config(profile, path, require_secrets=False)

    path.write_text(path.read_text() + "allow_absolute_state_dir: true\n")
    assert (
        load_telegram_config(profile, path, require_secrets=False).state_path
        == state.resolve()
    )


def test_webhook_requires_https_named_secret_and_resolves_profile_env(tmp_path):
    root, _ = _profile(tmp_path)
    (root / ".env").write_text(
        "BOT_TOKEN=bot-secret\nHOOK_SECRET=hook-secret\n", encoding="utf-8"
    )
    profile = AgentProfile.load(root)
    path = _telegram(
        root,
        """
token_env: BOT_TOKEN
allowed_user_ids: [1]
mode: webhook
webhook:
  public_url: https://example.test/hooks/lingcore
  secret_token_env: HOOK_SECRET
""",
    )
    config = load_telegram_config(profile, path)
    assert config.webhook_path == "hooks/lingcore"
    assert config.resolve_webhook_secret() == "hook-secret"

    path.write_text(path.read_text().replace("https://", "http://"))
    with pytest.raises(ConfigError, match="HTTPS"):
        load_telegram_config(profile, path)


@pytest.mark.parametrize(
    "invalid",
    (
        "base64+/=",
        "contains whitespace",
        "x" * 257,
    ),
)
def test_webhook_secret_rejects_telegram_invalid_values_without_echoing(
    tmp_path, invalid
):
    root, profile = _profile(tmp_path)
    path = _telegram(
        root,
        """
token_env: BOT_TOKEN
allowed_user_ids: [1]
mode: webhook
webhook:
  public_url: https://example.test/hooks
  secret_token_env: HOOK_SECRET
""",
    )
    config = load_telegram_config(profile, path, require_secrets=False)

    with pytest.raises(ConfigError, match="1-256 characters") as caught:
        config.resolve_webhook_secret({"HOOK_SECRET": invalid})
    assert invalid not in str(caught.value)


def test_webhook_secret_accepts_maximum_valid_length(tmp_path):
    root, profile = _profile(tmp_path)
    path = _telegram(
        root,
        """
token_env: BOT_TOKEN
allowed_user_ids: [1]
mode: webhook
webhook:
  public_url: https://example.test/hooks
  secret_token_env: HOOK_SECRET
""",
    )
    config = load_telegram_config(profile, path, require_secrets=False)
    secret = "A_b-9" + "x" * 251

    assert config.resolve_webhook_secret({"HOOK_SECRET": secret}) == secret


def test_doctor_adds_telegram_names_without_importing_ptb(
    tmp_path, monkeypatch, capsys
):
    root, _ = _profile(tmp_path)
    path = _telegram(
        root,
        """
token_env: TELEGRAM_TEST_TOKEN
allowed_user_ids: [1]
mode: webhook
webhook:
  public_url: https://example.test/hooks
  secret_token_env: TELEGRAM_TEST_WEBHOOK
""",
    )
    (root / ".env.example").write_text(
        "TELEGRAM_TEST_TOKEN=\nTELEGRAM_TEST_WEBHOOK=\n", encoding="utf-8"
    )
    monkeypatch.delenv("TELEGRAM_TEST_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TEST_WEBHOOK", raising=False)

    assert main(["doctor", "-p", str(root), "--telegram-config", str(path)]) == 1
    output = capsys.readouterr().out
    assert "TELEGRAM_TEST_TOKEN is missing" in output
    assert "TELEGRAM_TEST_WEBHOOK is missing" in output
    assert "documents 2 variable(s)" in output
    assert not (root / ".lingcore").exists()


def test_telegram_rejects_interactive_session_flags(tmp_path):
    root, _ = _profile(tmp_path)
    for flag in (
        ["--continue"],
        ["--resume", "abc"],
        ["--no-session"],
        ["--list-sessions"],
        ["--workspace", str(tmp_path)],
    ):
        with pytest.raises(SystemExit) as caught:
            main(["telegram", "-p", str(root), *flag])
        assert caught.value.code == 2


def test_telegram_dispatch_is_synchronous(tmp_path, monkeypatch):
    root, _ = _profile(tmp_path)
    called = {}

    def fake_run(profile_path, **kwargs):
        called["profile"] = profile_path
        called.update(kwargs)
        return 17

    import lingcore.integrations.telegram.application as application

    monkeypatch.setattr(application, "run_telegram", fake_run)
    monkeypatch.setattr(
        "lingcore.__main__.asyncio.run",
        lambda _: pytest.fail("Telegram must not enter asyncio.run"),
    )

    assert (
        main(
            [
                "telegram",
                "-p",
                str(root),
                "--telegram-mode",
                "polling",
            ]
        )
        == 17
    )
    assert called == {
        "profile": str(root),
        "telegram_config_path": None,
        "mode": "polling",
    }


def test_sessions_disabled_and_unsafe_shell_are_refused(tmp_path):
    root, profile = _profile(tmp_path, suffix="sessions:\n  enabled: false\n")
    path = _telegram(
        root, "token_env: TELEGRAM_TOKEN\nallowed_user_ids: [1]\n"
    )
    with pytest.raises(ConfigError, match="sessions.enabled"):
        load_telegram_config(profile, path, require_secrets=False)

    (root / "config.yaml").write_text(
        PROFILE
        + """
tools: [run_shell]
tool_options:
  run_shell:
    require_confirmation: false
    allow_patterns: []
""",
        encoding="utf-8",
    )
    # Replace PROFILE's original tools key instead of creating a duplicate.
    text = (root / "config.yaml").read_text()
    (root / "config.yaml").write_text(
        text.replace("tools: []\n", "", 1), encoding="utf-8"
    )
    unsafe = AgentProfile.load(root)
    with pytest.raises(ConfigError, match="require_confirmation"):
        load_telegram_config(unsafe, path, require_secrets=False)
