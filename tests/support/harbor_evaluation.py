"""Fixtures and builders for harbor evaluation tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from lingcore.evaluation.lock import resolve, write_lock
from lingcore.message import ToolCall
from tests.fakes import FakeLLMClient, ScriptedTurn


@pytest.fixture
def harbor_plan(tmp_path):
    task = tmp_path / "dataset" / "tiny"
    task.mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment").mkdir()
    (task / "instruction.md").write_text("Write a file.\n")
    (task / "tests/test.sh").write_text("test -f /app/answer")
    (task / "environment/Dockerfile").write_text("FROM fixture")
    desc = dict(
        status="ok",
        config={
            "agent": {"timeout_sec": 180},
            "environment": {},
            "verifier": {"timeout_sec": 1},
            "artifacts": ["/app/"],
        },
        properties=dict(
            os="linux",
            gpus=0,
            containers=1,
            cpus=1,
            memory_mb=256,
            storage_mb=1024,
            task_tools=False,
            steps=0,
            verifier_mode="separate",
        ),
        environments={
            "agent": {"kind": "image", "pinned": True},
            "verifier": {"kind": "image", "pinned": True},
        },
        resources={
            "agent": {"cpus": 1, "memory_mb": 256, "storage_mb": 1024, "gpus": 0},
            "verifier": {"cpus": 2, "memory_mb": 512, "storage_mb": 2048, "gpus": 0},
        },
    )
    (task / "description.json").write_text(json.dumps(desc))
    (
        tmp_path / "agent.yaml"
    ).write_text("""llm: {model: fake, api_key_env: FIXTURE_API_KEY, modalities: []}
tools: [terminal_exec, terminal_read_output]
sessions: {enabled: false}
loop: {parallel_tools: false}
persona: {project_instructions: []}
""")
    suite = dict(
        schema_version="lingcore.eval.suite.v1",
        name="harbor-test",
        source={
            "kind": "harbor",
            "dataset": "synthetic",
            "path": "dataset",
            "selection": {"policy": "sorted_capability_subset", "count": 1},
        },
        agent={"profile": "agent.yaml", "tool_binding": "harbor-terminal-v1"},
        execution={
            "runner": "harbor",
            "approval_policy": "benchmark_environment",
            "harbor": {"pass_env": ["FIXTURE_SCENARIO"]},
            "limits": {"wall_seconds": 1},
        },
        grading={"authority": "harbor"},
        trace={"content": "redacted", "exports": ["atif"]},
    )
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    bridge = tmp_path / "bridge"
    bridge.write_text(
        "#!"
        + sys.executable
        + "\n"
        + (Path(__file__).parents[1] / "fake_harbor_bridge.py").read_text()
    )
    bridge.chmod(0o700)
    return path, bridge


def client(command="write"):
    return lambda _: FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="call", name="terminal_exec", arguments={"command": command}
                    )
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )


async def frozen(harbor_plan):
    path, bridge = harbor_plan
    lock_path = path.parent / "lock.json"
    lock = await resolve(path, lock_path, bridge=bridge)
    write_lock(lock, lock_path)
    return lock, lock_path, bridge


@pytest.fixture
def trusted_release(harbor_plan, monkeypatch):
    """Stand in for reviewed publisher metadata, never a suite-supplied manifest."""
    from lingcore.evaluation.harbor import provenance
    from lingcore.evaluation.harbor.process import invoke

    path, bridge = harbor_plan
    description = invoke(
        bridge, ["describe", "--dataset", str(path.parent / "dataset")], {}
    )
    manifest = {
        "dataset": "synthetic",
        "hub_version": "sha256:" + "c" * 64,
        "hub_version_id": "fixture-version",
        "release_source": {
            "repository": "https://example.test/release",
            "commit": "a" * 40,
        },
        "release_url": "https://example.test/release/v1",
        "hub_url": "https://example.test/dataset/v1",
        "policy_url": "https://example.test/rules",
        "policy": {
            "repetitions": 5,
            "reward": {"name": "reward", "pass_at": 1, "values": [0, 1]},
        },
        "tasks": {t["id"]: t["harbor_digest"] for t in description["tasks"]},
    }
    monkeypatch.setattr(
        provenance,
        "release_manifest",
        lambda ref: manifest if ref == "synthetic" else None,
    )
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret-never-forwarded")
    return manifest


def benchmark_suite(harbor_plan):
    path, _ = harbor_plan
    suite = yaml.safe_load(path.read_text())
    suite["track"] = "benchmark"
    suite["source"]["training_use"] = "prohibited"
    suite["execution"]["repetitions"] = 5
    suite["execution"]["limits"] = {
        "wall_seconds": None,
        "model_requests": None,
        "tool_calls": None,
        "reported_tokens": None,
        "usage_missing": "ignore",
        "automatic_turns": 0,
    }
    path.write_text(yaml.safe_dump(suite))
    return suite
