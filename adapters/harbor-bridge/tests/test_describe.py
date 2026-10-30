from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from harbor.models.task.task import Task
from harbor.publisher.packager import Packager

from lingcore_harbor_bridge.describe import (
    MAX_JSON_BYTES,
    DescriptionError,
    canonical,
    describe,
    digest,
)


def test_describe_is_deterministic_and_matches_harbor(dataset):
    result = describe(dataset)
    assert result == describe(dataset)
    assert result["harbor"]["version"] == "0.25.0"
    assert "ATIF-v1.7" in result["atif_versions"]
    task = result["tasks"][0]
    assert task["status"] == "ok"
    assert task["name"] == "fixture/tiny"
    assert task["config"] == Task(dataset / "tiny").config.model_dump(mode="json")
    assert task["instruction_digest"] == digest(b"Write a file.\n")
    assert (
        task["harbor_digest"]
        == "sha256:" + Packager.compute_content_hash(dataset / "tiny")[0]
    )
    assert task["properties"] == {
        "os": "linux",
        "gpus": 0,
        "containers": 1,
        "cpus": 2,
        "memory_mb": 4096,
        "storage_mb": 10240,
        "task_tools": False,
        "steps": 0,
        "verifier_mode": "shared",
    }
    assert result["error_policy"]["table"]["AgentTimeoutError"]["reward_valid"]
    assert not result["error_policy"]["table"]["RewardFileEmptyError"]["reward_valid"]
    assert len(canonical(result)) < MAX_JSON_BYTES
    assert str(dataset).encode() not in canonical(result)


def test_cli_imports_without_lingcore_and_without_network_or_subprocesses(dataset):
    code = """
import json, sys
def guard(event, args):
    if event in {"socket.connect", "subprocess.Popen", "os.system"}:
        raise AssertionError("offline description attempted an external operation")
sys.addaudithook(guard)
class BlockLingCore:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lingcore" or fullname.startswith("lingcore."):
            raise AssertionError("bridge imported LingCore")
sys.meta_path.insert(0, BlockLingCore())
from lingcore_harbor_bridge.cli import main
raise SystemExit(main(["describe", "--dataset", sys.argv[1]]))
"""
    env = dict(os.environ, FIXTURE_API_KEY="fixture-secret-not-exported")
    output = subprocess.run(
        [sys.executable, "-c", code, str(dataset)],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert output.returncode == 0, output.stderr.decode()
    assert b"fixture-secret-not-exported" not in output.stdout + output.stderr
    assert json.loads(output.stdout)["tasks"][0]["status"] == "ok"


def test_imports_are_only_stdlib_harbor_and_the_bridge():
    import lingcore_harbor_bridge

    for path in Path(lingcore_harbor_bridge.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert all(
                name.split(".")[0]
                in sys.stdlib_module_names | {"harbor", "lingcore_harbor_bridge"}
                for name in names
                if name
            )


def test_bad_task_metadata_is_reported_without_input_values(dataset):
    (dataset / "tiny/task.toml").write_text('invalid = "fixture-secret-not-exported')
    task = describe(dataset)["tasks"][0]
    assert task["status"] == "metadata_unreadable"
    assert "fixture-secret" not in json.dumps(task)


def test_compose_and_separate_verifier_metadata(dataset):
    task = dataset / "tiny"
    (task / "environment/docker-compose.yaml").write_text(
        "services:\n  main: {image: fixture}\n  sidecar: {image: fixture}\n"
    )
    with (task / "task.toml").open("a") as file:
        file.write(
            '\n[verifier]\nenvironment_mode = "separate"\n[verifier.environment]\ndocker_image = "fixture@sha256:'
            + "a" * 64
            + '"\ngpus = 1\n'
        )
    value = describe(dataset)["tasks"][0]
    assert value["status"] == "ok"
    assert value["properties"]["containers"] == 2
    assert value["properties"]["gpus"] == 1
    assert value["properties"]["verifier_mode"] == "separate"
    assert value["environments"]["verifier"]["pinned"]


def test_symlink_and_oversized_metadata_are_not_read(dataset, tmp_path):
    task = dataset / "tiny"
    (tmp_path / "outside.txt").write_text("fixture-secret-not-exported")
    (task / "leak").symlink_to(tmp_path / "outside.txt")
    assert describe(dataset)["tasks"][0]["status"] == "metadata_unreadable"
    (task / "leak").unlink()
    (task / "task.toml").write_bytes(b"a" * (1024 * 1024 + 1))
    assert describe(dataset)["tasks"][0]["status"] == "metadata_unreadable"


def test_total_output_limit_is_enforced(dataset, monkeypatch):
    import importlib

    module = importlib.import_module("lingcore_harbor_bridge.describe")
    monkeypatch.setattr(module, "MAX_JSON_BYTES", 100)
    with pytest.raises(DescriptionError, match="description_too_large"):
        describe(dataset)


def test_harbor_loader_constructs_the_bridge_agent(tmp_path):
    from harbor.agents.factory import AgentFactory

    from lingcore_harbor_bridge.agent import BridgeAgent

    agent = AgentFactory.create_agent_from_import_path(
        "lingcore_harbor_bridge.agent:BridgeAgent", logs_dir=tmp_path
    )
    assert isinstance(agent, BridgeAgent)
    assert agent.name() == "lingcore-bridge" and agent.version() == "0.1.0"


def test_cli_rejects_empty_dataset_with_safe_json(tmp_path):
    output = subprocess.run(
        [
            sys.executable,
            "-m",
            "lingcore_harbor_bridge",
            "describe",
            "--dataset",
            str(tmp_path),
        ],
        capture_output=True,
        timeout=30,
    )
    assert output.returncode == 2
    assert json.loads(output.stdout) == {"error": {"code": "empty_dataset"}}


@pytest.mark.parametrize("folder", ["environment", "tests"])
def test_compose_implicit_main_is_counted(dataset, folder):
    task = dataset / "tiny"
    if folder == "tests":
        with (task / "task.toml").open("a") as file:
            file.write('\n[verifier]\nenvironment_mode = "separate"\n')
    (task / folder / "docker-compose.yaml").write_text(
        "services: {db: {image: fixture}}\n"
    )
    value = describe(dataset)["tasks"][0]
    assert value["status"] == "ok"
    assert value["properties"]["containers"] == 2
