"""SDK-free validation and translation of native Responses settings."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def responses_sampling(sampling: dict[str, Any]) -> dict[str, Any]:
    """Translate common Chat knobs without silently overriding native knobs."""
    result = deepcopy(sampling)
    if "extra_body" in result and not isinstance(
        result["extra_body"], (dict, type(None))
    ):
        raise ValueError("sampling.extra_body must be a mapping")
    if "reasoning" in result and not isinstance(result["reasoning"], dict):
        raise ValueError("sampling.reasoning must be a mapping")
    for alias in ("max_tokens", "max_completion_tokens"):
        if alias in result:
            value = result.pop(alias)
            if "max_output_tokens" in result and result["max_output_tokens"] != value:
                raise ValueError("conflicting output token limits in sampling")
            result["max_output_tokens"] = value
    if "reasoning_effort" in result:
        effort = result.pop("reasoning_effort")
        reasoning = result.setdefault("reasoning", {})
        if not isinstance(reasoning, dict):
            raise ValueError("sampling.reasoning must be a mapping")
        if "effort" in reasoning and reasoning["effort"] != effort:
            raise ValueError("conflicting reasoning effort settings in sampling")
        reasoning["effort"] = effort
    choice = result.get("tool_choice")
    if choice is not None and not isinstance(choice, (str, dict)):
        raise ValueError("sampling.tool_choice must be a string or mapping")
    if isinstance(choice, dict):
        if not isinstance(choice.get("type"), str):
            raise ValueError("sampling.tool_choice must declare a type")
        if choice["type"] == "function":
            function = choice.get("function", choice)
            if (
                not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
            ):
                raise ValueError("sampling.tool_choice function must declare a name")
            result["tool_choice"] = {"type": "function", "name": function["name"]}
    # Stateful continuation would bypass local rollback/compaction. Body
    # overrides must not smuggle it in after our normal request construction.
    reserved = {
        "input",
        "messages",
        "model",
        "tools",
        "stream",
        "store",
        "previous_response_id",
        "conversation",
        "background",
    }
    if reserved.intersection(result) or (reserved | {"include"}).intersection(
        result.get("extra_body") or {}
    ):
        raise ValueError(
            "sampling cannot override Responses conversation/transport fields"
        )
    include = result.setdefault("include", [])
    if not isinstance(include, list):
        raise ValueError("sampling.include must be a list")
    if "reasoning.encrypted_content" not in include:
        include.append("reasoning.encrypted_content")
    return result
