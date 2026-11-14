"""Static workflow validation and worst-case worker admission bounds."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import Field

from lingcore.delegation.config import RunLimits, StrictModel
from lingcore.delegation.envelope import node_class
from lingcore.delegation.models import Envelope
from lingcore.delegation.schema import compile_schema
from lingcore.errors import ConfigError
from lingcore.workflows.expressions import REFERENCE, parts, references

KINDS = {"agent", "map", "parallel", "loop", "keep", "transform", "log", "workflow"}


def input_values(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Optional inputs resolve to null consistently at every invocation level."""
    parsed = compile_schema(schema).validate_python(args, strict=True)
    result: dict[str, Any] = parsed.model_dump(mode="json")
    return result


class WorkflowSpec(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,79}$")
    description: str = Field(default="", max_length=2000)
    when_to_use: str = Field(default="", max_length=2000)
    inputs: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    limits: dict[str, Any] = Field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    phases: list[dict[str, str]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(min_length=1, max_length=64)
    result: Any = None


class ValidatedSpec:
    def __init__(
        self,
        spec: WorkflowSpec,
        limits: RunLimits,
        envelope: Callable[..., Envelope],
        catalog: dict[str, WorkflowSpec] | None = None,
        *,
        child: bool = False,
        models: frozenset[str] = frozenset(),
    ) -> None:
        self.spec, self.envelope = spec, envelope
        self.limits = limits.narrow(spec.limits)
        self.catalog = catalog or {}
        self.child = child
        self.models = models
        self.count = 0
        self.roles: dict[str, Envelope] = {}
        compile_schema(spec.inputs)
        if spec.inputs.get("type") != "object":
            raise ConfigError("workflow inputs must be an object schema")
        for schema in spec.schemas.values():
            compile_schema(schema)
        self.phases = {p.get("title") for p in spec.phases}
        try:
            self.bound, scope, _ = self.sequence(
                spec.steps, {"inputs", "request", "budget"}, {"inputs": spec.inputs}, 0
            )
        except (TypeError, AttributeError, KeyError) as exc:
            raise ConfigError(f"invalid workflow value: {type(exc).__name__}") from None
        self.check_refs(spec.result, scope)
        if self.bound > self.limits.max_agents:
            raise ConfigError(
                f"workflow admits at most {self.bound} agents, above max_agents={self.limits.max_agents}"
            )

    @staticmethod
    def check_refs(value: Any, scope: set[str]) -> None:
        for path in references(value):
            if parts(path)[0] not in scope:
                raise ConfigError(f"forward or unknown workflow reference: {path}")

    @staticmethod
    def shape(value: Any, shapes: dict[str, Any]) -> Any:
        if isinstance(value, list):
            return {"type": "array", "maxItems": len(value)}
        if isinstance(value, str) and (match := REFERENCE.fullmatch(value)):
            path = parts(match[1])
            current = shapes.get(str(path[0]), {})
            for part in path[1:]:
                current = (
                    current.get("items", {})
                    if isinstance(part, int)
                    else current.get("properties", {}).get(part, {})
                )
            return current
        return {}

    def sequence(
        self,
        steps: list[dict[str, Any]],
        scope: set[str],
        shapes: dict[str, Any],
        depth: int,
    ) -> tuple[int, set[str], dict[str, Any]]:
        if not isinstance(steps, list) or not steps:
            raise ConfigError("workflow sequences must be nonempty lists")
        scope, shapes = set(scope), dict(shapes)
        total = 0
        ids: set[str] = set()
        for step in steps:
            bound, shape = self.step(step, scope, shapes, depth)
            total += bound
            name = step.get("id")
            if name:
                if (
                    not isinstance(name, str)
                    or name in scope
                    or name in ids
                    or not name.replace("-", "_").isidentifier()
                ):
                    raise ConfigError("step ids must be unique public identifiers")
                ids.add(name)
                scope.add(name)
                shapes[name] = shape
            scope.add("prev")
            shapes["prev"] = shape
        return total, scope, shapes

    def step(
        self, step: dict[str, Any], scope: set[str], shapes: dict[str, Any], depth: int
    ) -> tuple[int, Any]:
        self.count += 1
        if self.count > 64 or depth > 4 or not isinstance(step, dict):
            raise ConfigError("workflow exceeds step or nesting cap")
        kinds = set(step) & KINDS
        if len(kinds) != 1 or set(step) - KINDS - {"id", "phase", "when"}:
            raise ConfigError("step needs exactly one supported kind")
        if "phase" in step and (
            not isinstance(step["phase"], str) or step["phase"] not in self.phases
        ):
            raise ConfigError("unknown workflow phase")
        self.check_refs(step.get("when"), scope)
        if "when" in step:
            self.predicate(step["when"])
        kind = next(iter(kinds))
        value = step[kind]
        if kind == "agent":
            if not isinstance(value, dict) or set(value) - {
                "role",
                "label",
                "prompt",
                "output",
                "repeat",
                "workspace",
                "tools",
                "retry_on",
                "retries",
                "model",
            }:
                raise ConfigError("invalid agent step")
            self.check_refs(value, scope)
            if not isinstance(value.get("prompt"), str):
                raise ConfigError("agent.prompt must be a string")
            role = value.get("role", "general")
            if value.get("model") is not None and value["model"] not in self.models:
                raise ConfigError("unknown workflow model alias")
            env = self.envelope(
                role, tools=value.get("tools"), workspace=value.get("workspace")
            )
            self.roles[role + ":" + env.workspace + ":" + env.identity()[:8]] = env
            repeat = value.get("repeat", 1)
            retries = value.get("retries", 2 if value.get("retry_on") else 0)
            if (
                type(repeat) is not int
                or not 1 <= repeat <= 5
                or type(retries) is not int
                or not 0 <= retries <= 2
            ):
                raise ConfigError("agent repeat/retry cap exceeded")
            retry_on = value.get("retry_on", [])
            if (
                not isinstance(retry_on, list)
                or any(v != "provider_transient" for v in retry_on)
                or (retries and not value.get("retry_on"))
            ):
                raise ConfigError(
                    "only explicit provider_transient retries are supported"
                )
            if value.get("retry_on") and node_class(env) == "effecting":
                raise ConfigError("effecting nodes cannot retry")
            output = value.get("output")
            shape = self.spec.schemas.get(output) if isinstance(output, str) else output
            if output is not None:
                if shape is None:
                    raise ConfigError("unknown output schema")
                compile_schema(shape)
            return repeat * (1 + retries), {
                "type": "array",
                "maxItems": repeat,
                "items": shape or {},
            } if repeat > 1 else (shape or {})
        if kind == "map":
            self.keys(
                value, {"over", "as", "stages", "max_items"}, {"over", "as", "stages"}
            )
            self.check_refs(value["over"], scope)
            cap = value.get("max_items", 100)
            if type(cap) is not int or not 0 <= cap <= 4096:
                raise ConfigError("invalid map item cap")
            known = self.shape(value["over"], shapes)
            cap = min(cap, known.get("maxItems", cap))
            variable = value["as"]
            if (
                not isinstance(variable, str)
                or variable in scope
                or not variable.isidentifier()
            ):
                raise ConfigError("map variable shadows a scope name")
            bound, _, inner = self.sequence(
                value["stages"],
                scope | {variable, "index", "prev"},
                shapes
                | {variable: known.get("items", {}), "prev": known.get("items", {})},
                depth + 1,
            )
            return cap * bound, {
                "type": "array",
                "maxItems": cap,
                "items": inner.get("prev", {}),
            }
        if kind == "parallel":
            if not isinstance(value, list) or any(not s.get("id") for s in value):
                raise ConfigError("parallel branches need ids")
            ids = [s["id"] for s in value]
            if len(set(ids)) != len(ids):
                raise ConfigError("parallel ids must be unique")
            evaluated = [self.step(s, scope, shapes, depth + 1) for s in value]
            return sum(n for n, _ in evaluated), {
                "type": "object",
                "properties": {s["id"]: v[1] for s, v in zip(value, evaluated)},
            }
        if kind == "loop":
            self.keys(
                value,
                {"max_rounds", "until", "collect", "steps"},
                {"max_rounds", "steps", "collect"},
            )
            rounds = value["max_rounds"]
            if type(rounds) is not int or not 1 <= rounds <= 4096:
                raise ConfigError("invalid loop round cap")
            until = value.get("until", {})
            if set(until) - {"dry_rounds", "count_at_least", "budget_below"}:
                raise ConfigError("unknown loop stopping rule")
            if "budget_below" in until and self.limits.reported_tokens is None:
                raise ConfigError("budget_below requires a run reported_tokens limit")
            for key in {"dry_rounds", "budget_below"} & set(until):
                if type(until[key]) is not int or until[key] < 0:
                    raise ConfigError(
                        "loop stopping thresholds must be nonnegative integers"
                    )
            self.keys(value["collect"], {"from", "dedupe_by"}, {"from"})
            bound, inner_scope, _ = self.sequence(
                value["steps"],
                scope | {"acc", "index", "prev"},
                shapes
                | {
                    "acc": {"type": "array", "maxItems": 4096},
                    "prev": {"type": "array", "maxItems": 4096},
                },
                depth + 1,
            )
            self.check_refs(value["collect"], inner_scope)
            self.check_refs(until, inner_scope)
            return rounds * bound, {"type": "array", "maxItems": 4096}
        if kind == "workflow":
            self.keys(value, {"name", "args"}, {"name"})
            self.check_refs(value.get("args", {}), scope)
            if self.child or value["name"] not in self.catalog:
                raise ConfigError("unknown child workflow or nesting above one")
            child = ValidatedSpec(
                self.catalog[value["name"]],
                self.limits,
                self.envelope,
                self.catalog,
                child=True,
                models=self.models,
            )
            self.roles.update(child.roles)
            self.count += child.count
            if self.count > 64:
                raise ConfigError("expanded workflow exceeds 64 steps")
            return child.bound, {}
        self.check_refs(value, scope)
        if kind == "keep":
            self.keys(value, {"if", "value"}, {"if", "value"})
            self.predicate(value["if"])
        elif kind == "transform":
            if (
                not isinstance(value, dict)
                or len(value) != 1
                or not set(value)
                <= {"flatten", "dedupe", "filter", "take", "count", "pick"}
            ):
                raise ConfigError("invalid transform")
            op, args = next(iter(value.items()))
            if op in {"take", "filter", "pick", "dedupe"}:
                fields = {
                    "take": {"over", "n"},
                    "filter": {"over", "field", "equals"},
                    "pick": {"over", "fields"},
                    "dedupe": {"over", "by"},
                }[op]
                self.keys(args, fields, fields - {"by"})
                if op == "take" and (
                    type(args["n"]) is not int or not 0 <= args["n"] <= 4096
                ):
                    raise ConfigError("take.n must be between 0 and 4096")
                if op == "pick" and (
                    not isinstance(args["fields"], list)
                    or len(args["fields"]) > 64
                    or not all(isinstance(v, str) for v in args["fields"])
                ):
                    raise ConfigError(
                        "pick.fields must be a bounded list of field names"
                    )
        elif kind == "log" and not isinstance(value, str):
            raise ConfigError("log must be text")
        return 0, {}

    @staticmethod
    def keys(value: Any, allowed: set[str], required: set[str]) -> None:
        if (
            not isinstance(value, dict)
            or set(value) - allowed
            or not required <= set(value)
        ):
            raise ConfigError("invalid workflow operator fields")

    def predicate(self, value: Any) -> None:
        if (
            not isinstance(value, dict)
            or len(value) != 1
            or not set(value)
            <= {"nonempty", "equals", "count_at_least", "votes", "not"}
        ):
            raise ConfigError("invalid predicate")
        if "not" in value:
            self.predicate(value["not"])
        for op in {"equals", "count_at_least"} & set(value):
            if not isinstance(value[op], list) or len(value[op]) != 2:
                raise ConfigError(f"{op} requires a pair")
            if op == "count_at_least" and (
                type(value[op][1]) is not int or value[op][1] < 0
            ):
                raise ConfigError("count threshold must be nonnegative")
        if "votes" in value:
            args = value["votes"]
            self.keys(
                args,
                {"over", "field", "equals", "at_least"},
                {"over", "field", "equals", "at_least"},
            )
            if (
                not isinstance(args["field"], str)
                or type(args["at_least"]) is not int
                or args["at_least"] < 0
            ):
                raise ConfigError("invalid vote threshold or field")
