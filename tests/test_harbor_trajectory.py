import json
from pathlib import Path
from types import SimpleNamespace

from lingcore.evaluation.harbor.trajectory import export_trajectory
from lingcore.tracing.models import TraceHealth

FIXTURE = Path(__file__).parents[1] / "adapters/harbor-bridge/protocol/atif"


def test_atif_golden_preserves_calls_and_accounts_without_inventing_usage(tmp_path):
    source = json.loads((FIXTURE / "input.json").read_bytes())
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in source["records"])
    )
    (tmp_path / "summary.json").write_text(TraceHealth().model_dump_json())
    (tmp_path / "artifacts/content.json").write_text(json.dumps(source["content"]))
    plan = SimpleNamespace(
        trace=SimpleNamespace(
            content="redacted", max_bytes=1048576, artifact_max_bytes=1048576
        ),
        runtime=SimpleNamespace(lingcore="fixture"),
        candidate=SimpleNamespace(model=SimpleNamespace(id="fake")),
    )
    value = export_trajectory(tmp_path, plan, "execution")
    assert value == json.loads((FIXTURE / "trajectory.json").read_bytes())
    assert value["final_metrics"]["total_prompt_tokens"] == 19
    assert value["steps"][2]["metrics"]["prompt_tokens"] == 10
    assert value["steps"][3]["metrics"]["prompt_tokens"] is None
    assert value["extra"]["lingcore"]["unattributed_request_ids"] == [
        "retry",
        "summary",
    ]
    assert [
        r["source_call_id"] for r in value["steps"][2]["observation"]["results"]
    ] == ["b", "a"]
    plan.trace.content = "metadata"
    assert export_trajectory(tmp_path, plan, "execution") is None
