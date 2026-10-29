from __future__ import annotations

import shutil

import pytest
import yaml

from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.identity import canonical_json
from lingcore.evaluation.lock import load_lock, resolve, validate, write_lock
from tests.test_evaluation import copied_suite, example_env


async def test_resolutions_are_stable_and_offline(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    first = await resolve(suite, tmp_path / "local.lock.json")
    second = await resolve(suite, tmp_path / "local.lock.json")
    assert first.plan_digest == second.plan_digest
    assert first.model_dump(exclude={"digest", "created_at"}) == second.model_dump(
        exclude={"digest", "created_at"}
    )
    body = canonical_json(first.model_dump(mode="json"))
    assert b"fixture-secret-only" not in body
    assert str(tmp_path).encode() not in body
    assert first.candidate.tools.initial == ["list_dir", "read_file", "write_file"]
    assert first.candidate.model.id == "fake-model"
    assert len(first.trials) == 2


@pytest.mark.parametrize(
    "change,section",
    [
        ("input", "source"),
        ("revision", "source"),
        ("grader", "grading"),
        ("world", "candidate"),
        ("include", "candidate"),
        ("model", "candidate"),
        ("endpoint", "candidate"),
        ("limit", "execution"),
        ("profile", "candidate"),
        ("fixture", "source"),
    ],
)
async def test_identity_detects_semantic_changes(
    tmp_path, monkeypatch, change, section
):
    example_env(monkeypatch)
    suite_path = copied_suite(tmp_path)
    (tmp_path / "world.md").write_text("one")
    (tmp_path / "included.md").write_text("one")
    profile_file = tmp_path / "agent.yaml"
    prof = yaml.safe_load(profile_file.read_text())
    prof["persona"]["include"] = ["included.md"]
    profile_file.write_text(yaml.safe_dump(prof))
    first = await resolve(suite_path, tmp_path / "lock.json")
    if change in {"input", "revision", "grader", "fixture"}:
        path = tmp_path / "case.yaml"
        case = yaml.safe_load(path.read_text())
        if change == "input":
            case["input"]["turns"][0]["user"] = "different"
        if change == "revision":
            case["revision"] = 2
        if change == "grader":
            case["graders"][0]["equals"] = "different"
        if change == "fixture":
            case["fixture"]["files"] = {"initial.txt": "different"}
        path.write_text(yaml.safe_dump(case))
    elif change in {"world", "include"}:
        (tmp_path / ("world.md" if change == "world" else "included.md")).write_text(
            "two"
        )
    elif change in {"model", "endpoint"}:
        monkeypatch.setenv(
            "LINGCORE_MODEL" if change == "model" else "LINGCORE_BASE_URL",
            "other-model" if change == "model" else "https://other.test/v1",
        )
    elif change == "profile":
        prof["loop"]["max_iters"] += 1
        profile_file.write_text(yaml.safe_dump(prof))
    else:
        suite = yaml.safe_load(suite_path.read_text())
        suite["execution"]["limits"]["model_requests"] += 1
        suite_path.write_text(yaml.safe_dump(suite))
    second = await resolve(suite_path, tmp_path / "lock.json")
    assert first.plan_digest != second.plan_digest
    assert getattr(first, section).digest != getattr(second, section).digest
    changed = {
        name
        for name in ("source", "candidate", "runtime", "execution", "grading")
        if getattr(first, name).digest != getattr(second, name).digest
    }
    assert changed == ({"source", "grading"} if change == "grader" else {section})


async def test_comments_and_case_tags_do_not_change_plan(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    first = await resolve(suite, tmp_path / "lock.json")
    for path in (suite, tmp_path / "agent.yaml", tmp_path / "case.yaml"):
        path.write_text("# a new comment\n" + path.read_text())
    case = yaml.safe_load((tmp_path / "case.yaml").read_text())
    case["tags"] = ["different-tag"]
    (tmp_path / "case.yaml").write_text(yaml.safe_dump(case))
    second = await resolve(suite, tmp_path / "lock.json")
    assert first.plan_digest == second.plan_digest


async def test_unset_credentials_can_be_provisioned_later(tmp_path, monkeypatch):
    example_env(monkeypatch)
    monkeypatch.delenv("LINGCORE_API_KEY")
    suite = copied_suite(tmp_path)
    first = await resolve(suite, tmp_path / "lock.json")
    assert "credential_unset" in first.limitations
    monkeypatch.setenv("LINGCORE_API_KEY", "another-fixture-key")
    assert (
        first.plan_digest == (await resolve(suite, tmp_path / "lock.json")).plan_digest
    )


async def test_secret_expansion_is_refused_before_hashing(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = copied_suite(
        tmp_path, profile_changes={"tool_options": {"something": "${LINGCORE_API_KEY}"}}
    )
    with pytest.raises(EvaluationConfigError, match="known credential"):
        await resolve(suite, tmp_path / "lock.json")


async def test_lock_paths_survive_directory_move(tmp_path, monkeypatch):
    example_env(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    suite = copied_suite(source)
    lock_path = source / "local.lock.json"
    lock = await resolve(suite, lock_path)
    write_lock(lock, lock_path)
    moved = tmp_path / "moved"
    shutil.move(source, moved)
    loaded = load_lock(moved / "local.lock.json")
    assert await validate(loaded, moved / "local.lock.json") == []


async def test_tool_description_and_runtime_files_are_measured(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    import lingcore.evaluation.identity as identity
    from lingcore.tools import REGISTRY

    first = await resolve(suite, tmp_path / "lock.json")
    tool = REGISTRY.get("read_file")
    monkeypatch.setattr(tool, "description", tool.description + " Changed")
    second = await resolve(suite, tmp_path / "lock.json")
    assert first.candidate.digest != second.candidate.digest
    assert first.runtime.digest == second.runtime.digest
    package = tmp_path / "package"
    package.mkdir()
    (package / "one.py").write_text("first")
    monkeypatch.setattr(identity, "_PACKAGE", package)
    one = identity.runtime_identity()
    (package / "one.py").write_text("second")
    assert one.digest != identity.runtime_identity().digest


@pytest.mark.parametrize("checkout", [True, False])
def test_runtime_git_metadata_requires_the_package_checkout(
    tmp_path, monkeypatch, checkout
):
    import subprocess

    import lingcore.evaluation.identity as identity

    root = tmp_path / "repo"
    package = (
        root / "lingcore" if checkout else root / ".venv/lib/site-packages/lingcore"
    )
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# fixture package")
    monkeypatch.setattr(identity, "_PACKAGE", package)
    calls = []

    def git(args, **kwargs):
        calls.append(args)
        output = (
            str(root).encode() + b"\n"
            if args[-1] == "--show-toplevel"
            else b"abc123\n"
            if args[-1] == "HEAD"
            else b""
        )
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(identity.subprocess, "run", git)
    runtime = identity.runtime_identity()
    assert runtime.git == ({"commit": "abc123", "dirty": False} if checkout else None)
    assert len(calls) == (3 if checkout else 1)
