from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lingcore.evaluation.report import exit_code
from lingcore.evaluation.runner import run_suite
from lingcore.message import ToolCall
from lingcore.usage import TokenUsage
from tests.fakes import FakeLLMClient, ScriptedTurn

EXAMPLE = Path(__file__).resolve().parents[1] / "docs/examples/evaluation/local.yaml"


def greeting_client(spec):
    usage = TokenUsage("m", 10, 2)
    return FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="w",
                        name="write_file",
                        arguments={
                            "path": "./hello.txt",
                            "content": "Hello, LingCore!\n",
                        },
                    )
                ],
                usage=usage,
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="r", name="read_file", arguments={"path": "hello.txt"})
                ],
                usage=usage,
            ),
            ScriptedTurn(text="complete", usage=usage),
        ]
    )


def example_env(monkeypatch):
    monkeypatch.setenv("LINGCORE_MODEL", "fake-model")
    monkeypatch.setenv("LINGCORE_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LINGCORE_API_KEY", "fixture-secret-only")


def copied_suite(
    tmp_path, *, suite_changes=None, profile_changes=None, case_changes=None
):
    root = EXAMPLE.parent
    suite = yaml.safe_load(EXAMPLE.read_text())
    profile = yaml.safe_load((root / "local-agent.yaml").read_text())
    case = yaml.safe_load((root / "cases/write-greeting.yaml").read_text())
    suite.update(suite_changes or {})
    profile.update(profile_changes or {})
    case.update(case_changes or {})
    suite["source"]["cases"] = ["case.yaml"]
    suite["agent"]["profile"] = "agent.yaml"
    for name, value in [
        ("suite.yaml", suite),
        ("agent.yaml", profile),
        ("case.yaml", case),
    ]:
        (tmp_path / name).write_text(yaml.safe_dump(value))
    return tmp_path / "suite.yaml"


async def test_supplied_local_example_runs_with_fake_client_and_fresh_state(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    seen_workspaces = []

    class CheckingClient(FakeLLMClient):
        async def stream(self, messages, tools=None):
            # A model never sees evaluator-only assertions or expected answers.
            assert all(
                "artifact_text" not in m.content and "all_required" not in m.content
                for m in messages
            )
            async for chunk in super().stream(messages, tools):
                yield chunk

    def factory(spec):
        raw = greeting_client(spec)
        return CheckingClient(raw._turns)

    from lingcore.agent import Agent

    original = Agent.from_profile.__func__

    def assemble(cls, *args, **kwargs):
        agent = original(cls, *args, **kwargs)
        if kwargs.get("execution_budget") is None:
            return agent  # offline identity assembly is not a trial
        assert not (agent.tool_ctx.workspace / "hello.txt").exists()
        seen_workspaces.append(agent.tool_ctx.workspace)
        return agent

    monkeypatch.setattr(Agent, "from_profile", classmethod(assemble))
    job, report = await run_suite(EXAMPLE, tmp_path, client_factory=factory)
    assert exit_code(report) == 0
    assert {
        key: report["coverage"][key]
        for key in (
            "planned",
            "started",
            "valid",
            "invalid",
            "unsupported",
            "cancelled",
            "missing",
        )
    } == {
        "planned": 2,
        "started": 2,
        "valid": 2,
        "invalid": 0,
        "unsupported": 0,
        "cancelled": 0,
        "missing": 0,
    }
    assert report["mean_case_success"] == 1
    assert len(set(seen_workspaces)) == 2 and all(
        not p.exists() for p in seen_workspaces
    )
    for result_file in job.glob("executions/*/result.json"):
        result = json.loads(result_file.read_text())
        assert result["locked"] is False and result["benchmark_compatible"] is False
        assert result["grade"]["scores"] == {"artifact": 1.0, "tool_policy": 1.0}
        assert (
            result["usage"]["requests"] == 3
            and result["usage"]["coverage"] == "complete"
        )
        assert (result_file.parent / "trace/events.jsonl").exists()
        assert not result["cleanup_failed"]
    assert "fixture-secret-only" not in "".join(
        p.read_text() for p in job.rglob("*.json*")
    )


async def test_metadata_only_argument_assertion_fails_preflight(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite_path = copied_suite(
        tmp_path, suite_changes={"trace": {"required": True, "content": "metadata"}}
    )
    job, report = await run_suite(
        suite_path, tmp_path / "out", client_factory=greeting_client
    )
    assert exit_code(report) == 2
    assert "redacted" in report["configuration_error"]
    assert not (job / "executions").exists()


async def test_forbidden_call_outside_ceiling_is_still_a_valid_failure(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)

    def factory(spec):
        client = greeting_client(spec)
        client._turns.insert(
            0,
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="denied", name="run_shell", arguments={"command": "ignored"}
                    )
                ],
                usage=TokenUsage("m", 1, 1),
            ),
        )
        return client

    _, report = await run_suite(EXAMPLE, tmp_path, client_factory=factory)
    assert exit_code(report) == 1 and report["coverage"]["valid"] == 2
    assert (
        report["cases"][0]["results"][0]["grade"]["graders"][1]["reason"]
        == "forbidden_tool_attempted"
    )


async def test_parallel_write_read_does_not_imply_causal_order(tmp_path, monkeypatch):
    example_env(monkeypatch)
    # Read first in the same parallel batch. It cannot be evidence that a
    # successful write completed before that read began.
    suite_path = copied_suite(
        tmp_path, profile_changes={"loop": {"parallel_tools": True}}
    )

    def factory(spec):
        return FakeLLMClient(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="r", name="read_file", arguments={"path": "hello.txt"}
                        ),
                        ToolCall(
                            id="w",
                            name="write_file",
                            arguments={
                                "path": "hello.txt",
                                "content": "Hello, LingCore!\n",
                            },
                        ),
                    ],
                    usage=TokenUsage("m", 1, 1),
                ),
                ScriptedTurn(text="done", usage=TokenUsage("m", 1, 1)),
            ]
        )

    _, report = await run_suite(suite_path, tmp_path / "out", client_factory=factory)
    assert exit_code(report) == 1


@pytest.mark.parametrize("change", ["lock", "required_tools", "host_shell"])
async def test_preflight_rejects_without_client_or_workspace(
    tmp_path, monkeypatch, change
):
    example_env(monkeypatch)
    suite_path = copied_suite(tmp_path)
    file = (
        suite_path
        if change == "lock"
        else tmp_path / "case.yaml"
        if change == "required_tools"
        else tmp_path / "agent.yaml"
    )
    data = yaml.safe_load(file.read_text())
    if change == "lock":
        data["source"]["lock"]["required"] = True
    elif change == "required_tools":
        data["requires"]["tools"] = ["missing_binding"]
    else:
        data["tools"].append("run_shell")
    file.write_text(yaml.safe_dump(data))

    def forbidden(spec):
        raise AssertionError("preflight must not construct clients")

    job, report = await run_suite(
        suite_path, tmp_path / "out", client_factory=forbidden
    )
    assert exit_code(report) == 2 and report["configuration_error"]
    assert not (job / "executions").exists()
    assert (job / "report.json").exists()


async def test_symlink_artifact_never_reads_host_file(tmp_path, monkeypatch):
    example_env(monkeypatch)
    secret = tmp_path / "secret"
    secret.write_text("Hello, LingCore!\n")
    from lingcore.agent import Agent

    original = Agent.from_profile.__func__

    def assemble(cls, *args, **kwargs):
        agent = original(cls, *args, **kwargs)
        (agent.tool_ctx.workspace / "hello.txt").symlink_to(secret)
        return agent

    monkeypatch.setattr(Agent, "from_profile", classmethod(assemble))

    def client(spec):
        return FakeLLMClient([ScriptedTurn(text="done", usage=TokenUsage("m", 1, 1))])

    _, report = await run_suite(EXAMPLE, tmp_path / "out", client_factory=client)
    assert exit_code(report) == 1
    assert report["cases"][0]["results"][0]["grade"]["scores"]["artifact"] == 0


def test_cli_eval_and_trace_commands(tmp_path, monkeypatch, capsys):
    example_env(monkeypatch)
    import lingcore.evaluation.runner as runner
    from lingcore.__main__ import main

    original = runner.run_suite

    async def scripted(path, output, **kwargs):
        return await original(path, output, client_factory=greeting_client, **kwargs)

    monkeypatch.setattr(runner, "run_suite", scripted)
    assert main(["eval", "run", str(EXAMPLE), "--output", str(tmp_path)]) == 0
    job = next(tmp_path.iterdir())
    assert main(["eval", "report", str(job)]) == 0
    trace = next(job.glob("executions/*/trace"))
    assert main(["trace", "show", str(trace)]) == 0
    assert "write-greeting" in capsys.readouterr().out


async def test_preflight_interruption_preserves_report(tmp_path, monkeypatch):
    import asyncio

    import lingcore.evaluation.runner as runner

    def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(runner, "load_yaml", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await run_suite(EXAMPLE, tmp_path)
    job = next(tmp_path.iterdir())
    report = json.loads((job / "report.json").read_text())
    assert exit_code(report) == 130 and not report["complete"]
    assert (job / "report.json").exists()


async def test_preflight_unexpected_failure_preserves_report(tmp_path, monkeypatch):
    import lingcore.evaluation.runner as runner

    def broken(*args, **kwargs):
        raise RuntimeError("sensitive failure detail")

    monkeypatch.setattr(runner, "load_yaml", broken)
    job, report = await run_suite(EXAMPLE, tmp_path)
    assert exit_code(report) == 2
    assert "sensitive failure detail" not in (job / "report.json").read_text()
