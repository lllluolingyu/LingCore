"""A data lookup language, deliberately without eval, calls or arithmetic."""

from __future__ import annotations

import json
import re
from typing import Any

from lingcore.errors import ConfigError

REFERENCE = re.compile(r"\$\{([^{}]+)\}")
PATH = re.compile(
    r"[A-Za-z][A-Za-z0-9_-]*(?:(?:\.[A-Za-z][A-Za-z0-9_-]*)|(?:\[\d+\]))*\Z"
)


class BlockedDependency(Exception):
    """A consumer must not run after an unsafe-to-replay dependency."""


BLOCKED = object()


def contains_blocked(value: Any) -> bool:
    if value is BLOCKED:
        return True
    if isinstance(value, dict):
        return any(contains_blocked(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_blocked(v) for v in value)
    return False


def parts(path: str) -> list[str | int]:
    if not PATH.fullmatch(path):
        raise ConfigError(f"invalid workflow reference: {path}")
    return [
        int(p) if p.isdigit() else p
        for p in re.findall(r"[A-Za-z][A-Za-z0-9_-]*|\d+", path)
    ]


def lookup(path: str, context: dict[str, Any]) -> Any:
    value: Any = context
    for part in parts(path):
        if value is BLOCKED:
            raise BlockedDependency(path)
        if isinstance(value, dict) and isinstance(part, str):
            if part not in value:
                raise ConfigError(f"missing workflow value: {path}")
            value = value[part]
        elif isinstance(value, list) and isinstance(part, int) and part < len(value):
            value = value[part]
        elif value is None:
            return None
        else:
            raise ConfigError(f"invalid lookup in workflow value: {path}")
    if contains_blocked(value):
        raise BlockedDependency(path)
    return value


def references(value: Any) -> list[str]:
    if isinstance(value, str):
        found = REFERENCE.findall(value)
        if "${" in REFERENCE.sub("", value):
            raise ConfigError("malformed workflow reference")
        for ref in found:
            parts(ref)
        return found
    if isinstance(value, list):
        return [r for item in value for r in references(item)]
    if isinstance(value, dict):
        return [r for item in value.values() for r in references(item)]
    return []


def resolve(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        exact = REFERENCE.fullmatch(value)
        if exact:
            return lookup(exact[1], context)

        def substitute(match: re.Match[str]) -> str:
            item = lookup(match[1], context)
            return (
                item
                if isinstance(item, str)
                else json.dumps(item, ensure_ascii=False, sort_keys=True)
            )

        return REFERENCE.sub(substitute, value)
    if isinstance(value, list):
        return [resolve(v, context) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, context) for k, v in value.items()}
    return value


def predicate(value: dict[str, Any], context: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or len(value) != 1:
        raise ConfigError("predicate requires exactly one operator")
    op, arg = next(iter(value.items()))
    if op == "not":
        return not predicate(arg, context)
    arg = resolve(arg, context)
    if op == "nonempty":
        return bool(arg)
    if op == "equals":
        return bool(arg[0] == arg[1])
    if op == "count_at_least":
        return bool(len(array(arg[0])) >= arg[1])
    if op == "votes":
        return bool(
            sum(
                isinstance(v, dict) and v.get(arg["field"]) == arg["equals"]
                for v in array(arg["over"])
            )
            >= arg["at_least"]
        )
    raise ConfigError("unknown workflow predicate")


def transform(value: dict[str, Any], context: dict[str, Any]) -> Any:
    if not isinstance(value, dict) or len(value) != 1:
        raise ConfigError("transform requires exactly one operator")
    op, arg = next(iter(resolve(value, context).items()))
    if op == "flatten":
        return [
            item
            for batch in array(arg)
            for item in (batch if isinstance(batch, list) else [batch])
            if item is not None
        ]
    if op == "count":
        return len(array(arg))
    if op == "take":
        return array(arg["over"])[: arg["n"]]
    if op == "filter":
        return [
            v
            for v in array(arg["over"])
            if isinstance(v, dict) and v.get(arg["field"]) == arg["equals"]
        ]
    if op == "pick":
        over = arg["over"]
        if over is None:
            return None
        if not isinstance(over, dict):
            raise ConfigError("pick.over must be an object or null")
        return {k: over.get(k) for k in arg["fields"]}
    if op == "dedupe":
        return dedupe(array(arg["over"]), arg.get("by"))
    raise ConfigError("unknown workflow transform")


def array(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError("workflow collection must be an array or null")
    return value


def dedupe(values: list[Any], by: str | None) -> list[Any]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = json.dumps(
            value.get(by) if by and isinstance(value, dict) else value, sort_keys=True
        )
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result
