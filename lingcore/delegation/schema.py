"""Bounded data-only schema compiler; no refs, regexes or executable expressions."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
)

from lingcore.errors import ConfigError

ALLOWED = {
    "type",
    "enum",
    "required",
    "properties",
    "items",
    "maxItems",
    "maxLength",
    "minimum",
    "maximum",
    "description",
}


def compile_schema(schema: dict[str, Any]) -> TypeAdapter[Any]:
    properties = 0

    def build(spec: dict[str, Any], depth: int) -> Any:
        nonlocal properties
        if depth > 5 or not isinstance(spec, dict) or set(spec) - ALLOWED:
            raise ConfigError("unsupported result schema or depth above 5")
        if "enum" in spec:
            values = spec["enum"]
            if (
                not isinstance(values, list)
                or not values
                or len(values) > 100
                or any(
                    not isinstance(v, (str, int, float, bool, type(None)))
                    for v in values
                )
            ):
                raise ConfigError("invalid schema enum")

            def member(value: Any) -> Any:
                if not any(
                    value == item
                    and (
                        type(value) is type(item)
                        or type(value) in {int, float}
                        and type(item) in {int, float}
                    )
                    for item in values
                ):
                    raise ValueError("value is not in schema enum")
                return value

            if "type" in spec:
                base = build({k: v for k, v in spec.items() if k != "enum"}, depth)
                return Annotated[base, AfterValidator(member)]
            return Annotated[Literal[tuple(values)], BeforeValidator(member)]
        kind = spec.get("type")
        if isinstance(kind, list):
            if not kind or len(kind) > 7:
                raise ConfigError("invalid schema union")
            return Union[tuple(build(spec | {"type": t}, depth) for t in kind)]
        if kind == "object":
            props = spec.get("properties", {})
            required = spec.get("required", [])
            if (
                not isinstance(props, dict)
                or not isinstance(required, list)
                or not set(required) <= props.keys()
            ):
                raise ConfigError("invalid object properties/required")
            properties += len(props)
            if properties > 64:
                raise ConfigError("result schema exceeds 64 properties")
            fields: dict[str, Any] = {}
            for key, value in props.items():
                if (
                    not isinstance(key, str)
                    or not key.isidentifier()
                    or key.startswith("_")
                ):
                    raise ConfigError(
                        "schema property names must be public identifiers"
                    )
                annotation = build(value, depth + 1)
                fields[key] = (annotation, ... if key in required else None)
            return create_model(
                f"Result{properties}",
                __config__=ConfigDict(extra="forbid", strict=True),
                **fields,
            )
        if kind == "array":
            cap = spec.get("maxItems", 200)
            if (
                not isinstance(cap, int)
                or isinstance(cap, bool)
                or not 0 <= cap <= 4096
            ):
                raise ConfigError("array maxItems must be between 0 and 4096")
            return Annotated[
                list.__class_getitem__(build(spec.get("items", {}), depth + 1)),
                Field(max_length=cap),
            ]
        types = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "null": type(None),
        }
        if kind not in types:
            raise ConfigError("schema must declare a supported type")
        constraints = {}
        for source, target in (
            ("maxLength", "max_length"),
            ("minimum", "ge"),
            ("maximum", "le"),
        ):
            if source in spec:
                constraints[target] = spec[source]
        return (
            Annotated[types[kind], Field(**constraints)] if constraints else types[kind]
        )

    try:
        return TypeAdapter(build(schema, 0))
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"invalid result schema: {type(exc).__name__}") from None


def plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def validate_normalized(schema: dict[str, Any], value: Any) -> Any:
    """Revalidate guarded JSON while preserving normalized missing optionals."""

    def original(data: Any, spec: dict[str, Any]) -> Any:
        if isinstance(data, dict):
            props, required = spec.get("properties", {}), spec.get("required", [])
            return {
                k: original(v, props.get(k, {}))
                for k, v in data.items()
                if not (v is None and k in props and k not in required)
            }
        if isinstance(data, list):
            return [original(v, spec.get("items", {})) for v in data]
        return data

    return plain(
        compile_schema(schema).validate_python(original(value, schema), strict=True)
    )
