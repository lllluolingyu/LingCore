"""Data-only plugin inspection and atomic scaffold CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.__main__ import main
from lingcore.errors import ConfigError
from lingcore.plugins.scaffold import scaffold_plugin


def profile(tmp_path: Path) -> Path:
    root = tmp_path / "profile"
    root.mkdir()
    (root / "config.yaml").write_text(
        "name: test\nworkspace: workspace\nllm:\n  model: test\ntools: []\n", "utf-8"
    )
    return root


def test_scaffold_is_complete_and_refuses_existing(tmp_path):
    root = profile(tmp_path)
    created = scaffold_plugin(root, "hello-world")
    assert created == root / "plugins" / "hello-world"
    for name in [
        "plugin.yaml",
        "plugin.py",
        "commands/hello.md",
        "skills/hello-world/skill.md",
        "README.md",
    ]:
        assert (created / name).is_file()
    assert "hello_world_echo" in (created / "plugin.yaml").read_text()
    with pytest.raises(ConfigError, match="already exists"):
        scaffold_plugin(root, "hello-world")
    assert not any(p.name.startswith(".hello-world-") for p in created.parent.iterdir())


def test_scaffold_optional_components_and_bad_name(tmp_path):
    root = profile(tmp_path)
    created = scaffold_plugin(root, "minimal", hooks=False, commands=False)
    assert not (created / "commands").exists()
    assert "hooks:" not in (created / "plugin.yaml").read_text()
    for name in ["../escape", "UPPER", "a" * 41, "bad_name"]:
        with pytest.raises(ConfigError):
            scaffold_plugin(root, name)


def test_scaffold_refuses_installed_tree(tmp_path, monkeypatch):
    import lingcore.plugins.scaffold as module

    root = profile(tmp_path)
    monkeypatch.setattr(module, "_PACKAGE_DIR", root)
    with pytest.raises(ConfigError, match="installed package"):
        scaffold_plugin(root, "hello")


def test_plugin_cli_new_list_info_do_not_import(tmp_path, capsys):
    root = profile(tmp_path)
    assert main(["plugin", "new", "hello", "-p", str(root)]) == 0
    out = capsys.readouterr().out
    assert "plugins: [hello]" in out and "hello_echo" in out
    code = root / "plugins" / "hello" / "plugin.py"
    code.write_text("raise RuntimeError('inspection executed plugin code')\n", "utf-8")
    assert main(["plugin", "list", "-p", str(root)]) == 0
    out = capsys.readouterr().out
    assert "hello" in out and "0.1.0" in out and "local" in out
    assert main(["plugin", "info", "hello", "-p", str(root)]) == 0
    out = capsys.readouterr().out
    assert "hello_echo" in out and "unauthorized" in out
    assert main(["plugin", "info", "missing", "-p", str(root)]) == 2


def test_scaffold_failure_never_publishes_partial_dir(tmp_path, monkeypatch):
    root = profile(tmp_path)
    original = Path.write_text

    def fail(self, text, *args, **kwargs):
        if self.name == "plugin.py":
            raise OSError("disk failure")
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(ConfigError):
        scaffold_plugin(root, "hello")
    assert not (root / "plugins" / "hello").exists()
    assert list((root / "plugins").iterdir()) == []
