from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from lingcore.delegation.config import RunLimits
from lingcore.delegation.models import Envelope
from lingcore.errors import ConfigError
from lingcore.workflows.expressions import resolve
from lingcore.workflows.spec import ValidatedSpec, WorkflowSpec


def validate(raw, **kwargs):
    return ValidatedSpec(
        WorkflowSpec.model_validate(raw),
        RunLimits(),
        lambda role, **kw: Envelope(
            tools=frozenset(), workspace=kw.get("workspace") or "scratch"
        ),
        **kwargs,
    )


def example():
    return yaml.safe_load(
        Path("profiles/coding/workflows/review-changes.yaml").read_text()
    )


def test_review_bound_exactly_40_and_old_184_rejected():
    raw = example()
    assert validate(raw).bound == 40
    raw["schemas"]["findings"]["properties"]["findings"]["maxItems"] = 20
    raw["steps"][0]["map"]["stages"][1]["map"]["max_items"] = 20
    with pytest.raises(ConfigError, match="184"):
        validate(raw)


def test_retries_multiply_bound_before_admission():
    raw = {
        "name": "retry",
        "steps": [
            {"agent": {"prompt": "hi", "repeat": 5, "retry_on": ["provider_transient"]}}
        ],
    }
    assert validate(raw).bound == 15


@pytest.mark.parametrize(
    "path",
    ["later.result", "__class__", "inputs.run()", "inputs[1+1]", "inputs.x | shell"],
)
def test_no_forward_references_calls_or_arithmetic(path):
    with pytest.raises(ConfigError):
        validate({"name": "bad", "steps": [{"agent": {"prompt": "${" + path + "}"}}]})


def test_value_resolution_preserves_typed_exact_reference():
    context = {"inputs": {"array": [{"name": "hello"}]}}
    assert resolve("${inputs.array}", context) == [{"name": "hello"}]
    assert resolve("hi ${inputs.array[0].name}", context) == "hi hello"


def test_nested_and_child_bounds_share_ceiling():
    child = WorkflowSpec.model_validate(
        {"name": "child", "steps": [{"agent": {"prompt": "hi", "repeat": 5}}]}
    )
    raw = {
        "name": "parent",
        "steps": [
            {
                "map": {
                    "over": list(range(11)),
                    "as": "item",
                    "stages": [{"workflow": {"name": "child"}}],
                }
            }
        ],
    }
    with pytest.raises(ConfigError, match="55"):
        validate(raw, catalog={"child": child})


def test_plugin_effecting_retry_refused_before_run():
    raw = {
        "name": "bad",
        "steps": [{"agent": {"prompt": "hi", "retry_on": ["provider_transient"]}}],
    }
    with pytest.raises(ConfigError, match="effecting"):
        ValidatedSpec(
            WorkflowSpec.model_validate(raw),
            RunLimits(),
            lambda *a, **k: Envelope(tools=frozenset({"plugin_write"})),
        )


def test_model_override_requires_declared_alias():
    raw = {"name": "model", "steps": [{"agent": {"prompt": "hi", "model": "missing"}}]}
    with pytest.raises(ConfigError, match="model alias"):
        validate(raw)
    assert validate(deepcopy(raw), models=frozenset({"missing"})).bound == 1
