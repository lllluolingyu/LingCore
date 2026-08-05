"""Profile-selected guardrail loading."""

from __future__ import annotations

import pytest

from lingcore.errors import ConfigError
from lingcore.guardrails import NoopGuardrail, build_guardrail


class PrefixGuardrail:
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    async def pre_input(self, text: str) -> str:
        return self.prefix + text

    async def post_output(self, text: str) -> str:
        return text + self.prefix


async def test_noop_guardrail_remains_builtin():
    guardrail = build_guardrail("noop")
    assert isinstance(guardrail, NoopGuardrail)
    assert await guardrail.pre_input("hello") == "hello"


async def test_dotted_guardrail_class_receives_profile_options():
    guardrail = build_guardrail(
        "tests.test_guardrails:PrefixGuardrail", {"prefix": "checked: "}
    )
    assert await guardrail.pre_input("hello") == "checked: hello"
    assert await guardrail.post_output("done") == "donechecked: "


def test_invalid_guardrail_target_is_a_config_error():
    with pytest.raises(ConfigError, match="could not load guardrail"):
        build_guardrail("tests.test_guardrails:Missing")
