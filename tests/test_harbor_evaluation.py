"""Harbor evaluation tests: core behavior."""

from __future__ import annotations

import json

import pytest

from lingcore.evaluation.harbor.source import select
from lingcore.evaluation.identity import identity_body
from lingcore.evaluation.lock import validate
from lingcore.evaluation.models import HarborSource
from lingcore.evaluation.runner import resume, run_locked
from tests.support.harbor_evaluation import client, frozen
from tests.support.harbor_evaluation import harbor_plan as harbor_plan
from tests.support.harbor_evaluation import trusted_release as trusted_release


async def test_harbor_lock_drift_split_and_binding(harbor_plan):
    lock, path, bridge = await frozen(harbor_plan)
    assert not await validate(lock, path, bridge=bridge)
    assert lock.candidate.tools.binding_id == "harbor-terminal-v1"
    assert "path" not in identity_body(lock)["source"]["dataset"]
    (path.parent / "dataset/tiny/tests/test.sh").write_text("changed test")
    drift = await validate(lock, path, bridge=bridge)
    assert any(d.field == "source.tasks.tiny.verifier_digest" for d in drift)
    assert not any("input_digest" in d.field for d in drift)


@pytest.mark.parametrize(
    "scenario", ["", "wait_interrupt", "malformed", "unsupported", "before_hello"]
)
async def test_runner_outcomes_and_real_environment_filter(
    harbor_plan, monkeypatch, scenario
):
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret-never-forwarded")
    monkeypatch.setenv("FIXTURE_SCENARIO", scenario)
    _, path, bridge = await frozen(harbor_plan)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    results = list(job.glob("executions/*/result.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_bytes())
    if scenario == "":
        assert (
            result["grade"]["passed"]
            and report["mean_reward"] == 1
            and report["harbor_accuracy"] == 1
        )
    elif scenario == "wait_interrupt":
        assert result["execution"]["status"] == "timed_out"
        assert result["grade"]["status"] == "valid"
        frames = json.loads(
            next(job.glob("executions/*/harbor/job/trial/frames.json")).read_bytes()
        )
        assert [f["type"] for f in frames if f["type"] != "usage"] == [
            "exec",
            "interrupt",
            "finish",
        ]
    elif scenario == "unsupported":
        assert result["unsupported"] and report["coverage"]["unsupported"] == 1
        assert result["infrastructure_failure"] is None
        assert report["trials"][0]["reason"] == "unsupported"
    else:
        assert result["execution"]["cause"] == "runner"
    for file in job.rglob("*"):
        if file.is_file():
            assert b"fixture-secret-never-forwarded" not in file.read_bytes()


async def test_regrade_preserves_original_records_and_never_constructs_agent(
    harbor_plan, monkeypatch
):
    _, path, bridge = await frozen(harbor_plan)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client("fail")
    )
    assert report["mean_reward"] == 0
    before = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}
    (path.parent / "dataset/tiny/tests/test.sh").write_text("corrected")
    from lingcore.agent import Agent
    from lingcore.evaluation.regrade import regrade

    monkeypatch.setattr(
        Agent,
        "from_profile",
        lambda *a, **k: pytest.fail("Agent created during regrade"),
    )
    result = await regrade(job, bridge=bridge)
    assert result["mean_reward"] == 1
    from lingcore.evaluation.compare import _load

    loaded = _load(job, "latest")
    assert loaded.report["cases"] == result["cases"]
    assert all(
        r.rewards == {"reward": 1.0} and r.reward_valid for r in loaded.results.values()
    )
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("command,passed", [("write", True), ("fail", False)])
async def test_oversized_checkpoint_preserves_verifier_grade(
    harbor_plan, monkeypatch, command, passed
):
    from lingcore.evaluation.harbor import runner
    from lingcore.evaluation.identity import digest

    collect = runner.collect_evidence

    def with_checkpoint(root, limit):
        # Match the CPU-10 failure without allocating a large in-memory fixture.
        with (root / "harbor/model.safetensors").open("wb") as stream:
            stream.truncate(186721248)
        (root / "harbor/at-limit.bin").write_bytes(b"x" * limit)
        collect(root, limit)

    monkeypatch.setattr(runner, "collect_evidence", with_checkpoint)
    lock, path, bridge = await frozen(harbor_plan)
    job, report = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client(command)
    )
    assert report["complete"] and report["coverage"]["valid"] == 1
    result_path = next(job.glob("executions/*/result.json"))
    result = json.loads(result_path.read_bytes())
    assert result["grade"]["passed"] is passed
    assert result["grade"]["scores"]["reward"] == float(passed)
    manifest = json.loads((result_path.parent / "evidence.json").read_bytes())
    files = {row["path"]: row for row in manifest["files"]}
    assert files["harbor/model.safetensors"] == {
        "path": "harbor/model.safetensors",
        "status": "oversized",
    }
    boundary = files["harbor/at-limit.bin"]
    assert boundary["status"] == "captured"
    assert boundary["bytes"] == lock.trace.artifact_max_bytes
    assert boundary["digest"] == digest(b"x" * boundary["bytes"])
    assert files["harbor/summary.json"]["status"] == "captured"
    assert (result_path.parent / "harbor/model.safetensors").stat().st_size == 186721248


def test_evidence_does_not_misclassify_unsafe_read_as_oversized(tmp_path, monkeypatch):
    from lingcore.evaluation.harbor import runner
    from lingcore.paths import PathEscapeError, ReadLimitError

    harbor = tmp_path / "harbor"
    harbor.mkdir()
    target = harbor / "artifact"
    target.write_bytes(b"safe")
    outside = tmp_path / "outside"
    outside.write_bytes(b"must not read")
    read = runner.read_bytes

    def swap_then_read(path, *, max_bytes):
        target.unlink()
        target.symlink_to(outside)
        return read(path, max_bytes=max_bytes)

    monkeypatch.setattr(runner, "read_bytes", swap_then_read)
    with pytest.raises(PathEscapeError) as error:
        runner.collect_evidence(tmp_path, 4)
    assert not isinstance(error.value, ReadLimitError)
    assert not (tmp_path / "evidence.json").exists()


async def test_resume_refuses_running_harness(harbor_plan):
    import fcntl

    from lingcore.evaluation.config import EvaluationConfigError

    _, path, bridge = await frozen(harbor_plan)
    job, _ = await run_locked(
        path, path.parent / "jobs", bridge=bridge, client_factory=client()
    )
    result = next(job.glob("executions/*/result.json"))
    result.unlink()
    with (result.parent / "harbor.lock").open("w") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        with pytest.raises(EvaluationConfigError, match="still running"):
            await resume(job, bridge=bridge, client_factory=client())
    report = await resume(job, bridge=bridge, client_factory=client())
    assert report["complete"] and report["coverage"]["replaced"] == 1


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("gpus", 1, "gpus"),
        ("containers", 2, "containers"),
        ("steps", 1, "steps"),
        ("task_tools", True, "task_tools"),
        ("cpus", 100, "cpus"),
    ],
)
def test_selection_uses_metadata(field, value, reason):
    source = HarborSource.model_validate(
        {
            "kind": "harbor",
            "dataset": "synthetic",
            "path": ".",
            "selection": {
                "policy": "sorted_capability_subset",
                "count": 1,
                "required": {"max_cpus": 4},
            },
        }
    )
    properties = dict(
        os="linux", gpus=0, containers=1, steps=0, task_tools=False, cpus=1
    )
    selected, excluded = select(
        source,
        {
            "tasks": [
                {"id": "a", "status": "ok", "properties": properties},
                {"id": "b", "status": "ok", "properties": properties | {field: value}},
            ]
        },
    )
    assert [t["id"] for t in selected] == ["a"]
    assert reason in excluded[0]["reasons"]
