from __future__ import annotations

import json

import pytest
import yaml

from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.lock import load_lock, resolve, validate, write_lock
from tests.test_evaluation import copied_suite, example_env


async def test_write_load_validate_and_replace(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    path = tmp_path / "local.lock.json"
    lock = await resolve(suite, path)
    write_lock(lock, path)
    assert load_lock(path) == lock
    assert await validate(lock, path) == []
    with pytest.raises(EvaluationConfigError, match="already exists"):
        write_lock(lock, path)
    write_lock(lock, path, replace=True)
    assert path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    "corruption", ["truncate", "value", "coercion", "major", "extra"]
)
async def test_corrupt_or_unknown_lock_is_rejected(tmp_path, monkeypatch, corruption):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    path = tmp_path / "local.lock.json"
    lock = await resolve(suite, path)
    write_lock(lock, path)
    data = json.loads(path.read_text())
    if corruption == "truncate":
        path.write_bytes(path.read_bytes()[:-10])
    else:
        if corruption == "value":
            data["candidate"]["model"]["id"] = "edited-model"
        if corruption == "coercion":
            data["trace"]["required"] = int(data["trace"]["required"])
        if corruption == "major":
            data["schema_version"] = "lingcore.eval-lock.v2"
        if corruption == "extra":
            data["future"] = True
        if corruption in {"major", "extra"}:
            from lingcore.evaluation.identity import value_digest

            data["digest"] = value_digest(
                {k: v for k, v in data.items() if k != "digest"}
            )
        path.write_text(json.dumps(data))
    with pytest.raises(EvaluationConfigError, match="corrupt"):
        load_lock(path)


async def test_drift_is_named_and_never_contains_values(tmp_path, monkeypatch):
    example_env(monkeypatch)
    suite = copied_suite(tmp_path)
    path = tmp_path / "local.lock.json"
    lock = await resolve(suite, path)
    case = tmp_path / "case.yaml"
    value = yaml.safe_load(case.read_text())
    value["input"]["turns"][0]["user"] = "a private changed instruction"
    case.write_text(yaml.safe_dump(value))
    drift = await validate(lock, path)
    assert any(d.field == "source.cases.write-greeting.input_digest" for d in drift)
    assert "private changed" not in str(drift)
    case.unlink()
    assert (await validate(lock, path))[0].kind == "missing"
