from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lingcore.evaluation.lock import resolve, validate, write_lock
from lingcore.evaluation.runner import run_locked
from lingcore.message import ToolCall
from lingcore.usage import TokenUsage
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.test_evaluation import example_env

EXAMPLES = Path(__file__).resolve().parents[1] / "docs/examples/evaluation"


def call(name, **arguments):
    return ScriptedTurn(
        tool_calls=[ToolCall(id="call", name=name, arguments=arguments)],
        usage=TokenUsage("fake-model", 10, 2),
    )


def done():
    return ScriptedTurn(text="done", usage=TokenUsage("fake-model", 10, 2))


def script(case):
    if case == "write-greeting":
        return [
            call("write_file", path="hello.txt", content="Hello, LingCore!\n"),
            call("read_file", path="hello.txt"),
            done(),
        ]
    if case == "retrieval-citation":
        return [
            call("knowledge", action="query", query="Release owner"),
            call("read_file", path="notes/current.md"),
            call(
                "write_file",
                path="owner.txt",
                content="Mira Chen\nnotes/current.md:2\n",
            ),
            done(),
        ]
    if case == "retrieval-refresh":
        return [
            call("knowledge", action="query", query="Launch code"),
            call("write_file", path="before.txt", content="AMBER\n"),
            done(),
            call("write_file", path="notes/launch.md", content="Launch code: COBALT\n"),
            call("knowledge", action="query", query="Launch code"),
            call("write_file", path="after.txt", content="COBALT\n"),
            done(),
        ]
    if case == "calendar-dst":
        return [
            call(
                "calendar_add",
                events=[
                    {
                        "title": "Standup",
                        "start": "2026-10-31T09:00",
                        "duration_minutes": 30,
                        "repeat": {"frequency": "daily", "count": 2},
                    }
                ],
            ),
            call("calendar_list", start="2026-10-31", end="2026-11-01"),
            call(
                "write_file",
                path="agenda.txt",
                content="2026-10-31T09:00-04:00|Standup\n2026-11-01T09:00-05:00|Standup\n",
            ),
            done(),
        ]
    assert case == "invoice-update"
    return [
        call("read_file", path="orders.csv"),
        call(
            "write_file", path="invoice.txt", content="blue=500\nred=375\ntotal=875\n"
        ),
        call("read_file", path="invoice.txt"),
        done(),
        call(
            "write_file",
            path="orders.csv",
            content="sku,quantity,unit_cents\nred,4,125\nblue,2,250\n",
        ),
        call("read_file", path="orders.csv"),
        call(
            "write_file", path="invoice.txt", content="blue=500\nred=500\ntotal=1000\n"
        ),
        call("read_file", path="invoice.txt"),
        done(),
    ]


def regression_client(spec):
    class CheckingClient(FakeLLMClient):
        async def stream(self, messages, tools=None):
            index = len(self.calls)
            if spec.case_id == "retrieval-refresh":
                if index == 1:
                    assert "AMBER" in messages[-1].content
                elif index == 5:
                    assert "AMBER" not in messages[-1].content
                    assert "COBALT" in messages[-1].content
            elif spec.case_id == "calendar-dst" and index == 2:
                text = messages[-1].content
                assert text.count("Standup") == 2
                assert "2026-10-31 09:00" in text and "2026-11-01 09:00" in text
            async for item in super().stream(messages, tools):
                yield item

    return CheckingClient(script(spec.case_id))


def suite_at(path, *, cases=None, repetitions=3):
    suite = yaml.safe_load((EXAMPLES / "regression.yaml").read_text())
    suite["agent"]["profile"] = str(EXAMPLES / "regression-agent.yaml")
    suite["execution"]["repetitions"] = repetitions
    suite["source"]["cases"] = [
        str(EXAMPLES / p)
        for p in suite["source"]["cases"]
        if cases is None or Path(p).stem in cases
    ]
    path.write_text(yaml.safe_dump(suite))
    return path


async def test_regression_suite_with_real_tools_fresh_state_and_locked_repeats(
    tmp_path, monkeypatch
):
    example_env(monkeypatch)
    suite = suite_at(tmp_path / "suite.yaml")
    lock_path = tmp_path / "lock.json"
    lock = await resolve(suite, lock_path)
    assert len(lock.trials) == 15
    write_lock(lock, lock_path)
    assert await validate(lock, lock_path) == []
    _, report = await run_locked(
        lock_path, tmp_path / "out", client_factory=regression_client
    )
    assert report["complete"], report
    assert report["coverage"]["valid"] == 15
    failures = [
        (row["case_id"], r["execution"], r["grade"])
        for row in report["cases"]
        for r in row["results"]
        if r["grade"]["passed"] is not True
    ]
    assert not failures, failures
    assert all(row["passed"] == 3 for row in report["cases"])
    assert report["usage"]["all"]["coverage"] == "complete"


@pytest.mark.parametrize(
    "case",
    [
        "write-greeting",
        "retrieval-citation",
        "retrieval-refresh",
        "calendar-dst",
        "invoice-update",
    ],
)
async def test_each_case_detects_wrong_artifacts(tmp_path, monkeypatch, case):
    example_env(monkeypatch)
    suite = suite_at(tmp_path / "suite.yaml", cases=[case], repetitions=1)
    lock_path = tmp_path / "lock.json"
    write_lock(await resolve(suite, lock_path), lock_path)

    def wrong(spec):
        turns = script(spec.case_id)
        for turn in reversed(turns):
            if turn.tool_calls and turn.tool_calls[0].name == "write_file":
                turn.tool_calls[0].arguments["content"] = "incorrect\n"
                break
        return FakeLLMClient(turns)

    _, report = await run_locked(lock_path, tmp_path / "out", client_factory=wrong)
    assert report["complete"]
    assert report["valid_failures"] == 1


@pytest.mark.parametrize("mistake", ["skip_tools", "duration", "start", "recurrence"])
async def test_correct_calendar_text_with_incorrect_calendar_actions_fails(
    tmp_path, monkeypatch, mistake
):
    example_env(monkeypatch)
    suite = suite_at(tmp_path / "suite.yaml", cases=["calendar-dst"], repetitions=1)
    lock_path = tmp_path / "lock.json"
    write_lock(await resolve(suite, lock_path), lock_path)
    turns = script("calendar-dst")
    if mistake == "skip_tools":
        turns = turns[2:]
    else:
        event = turns[0].tool_calls[0].arguments["events"][0]
        if mistake == "duration":
            event["duration_minutes"] = 60
        elif mistake == "start":
            event["start"] = "2026-10-31T10:00"
        else:
            event["repeat"]["count"] = 3
    _, report = await run_locked(
        lock_path,
        tmp_path / "out",
        client_factory=lambda _: FakeLLMClient(turns),
    )
    assert report["complete"] and report["valid_failures"] == 1
    grade = report["cases"][0]["results"][0]["grade"]
    assert grade["scores"]["dst-occurrences"] == 1
    assert grade["scores"]["calendar-policy"] == 0
