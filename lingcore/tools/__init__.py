"""The tool plugin contract.

Third-party tools import only from this module. A tool is an async function
whose first parameter is annotated with a pydantic ``BaseModel`` (its argument
schema) and whose second parameter is a ``ToolContext`` (scoped state such as
the workspace path and a confirmation callback). The ``@tool`` decorator turns
it into a ``Tool`` and registers it in the process-global ``REGISTRY``.

Tools receive their context as a parameter — never via globals — so that
concurrent sessions with different workspaces stay isolated.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Protocol,
    get_type_hints,
    runtime_checkable,
)

from pydantic import BaseModel

from lingcore.errors import ConfigError, ToolError
from lingcore.message import Attachment


@runtime_checkable
class ConfirmFn(Protocol):
    """A frontend-supplied callback gating risky actions (e.g. run_shell)."""

    def __call__(self, prompt: str) -> Awaitable[bool]: ...


@dataclass(slots=True)
class ToolContext:
    """Scoped state handed to every tool invocation."""

    workspace: Path
    confirm: ConfirmFn | None = None
    options: dict[str, Any] = field(default_factory=dict)
    profile_dir: Path | None = None
    # Parsed values from the selected profile's .env. Kept out of ``options``
    # so callers cannot accidentally serialize or echo secrets with tool config.
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)

    def getenv(self, name: str, default: str | None = None) -> str | None:
        """Resolve this profile's ``.env``, then the process environment.

        An empty profile value blocks the exported variable but is treated as
        unset (invariant 4): the caller's ``default`` is returned instead.
        """
        if name in self.environment:
            return self.environment[name] or default
        return os.environ.get(name, default)


@dataclass(slots=True)
class ToolOutput:
    """Structured tool output with optional media attachments."""

    text: str
    attachments: list[Attachment] = field(default_factory=list)


ToolFn = Callable[[BaseModel, ToolContext], Awaitable[str | ToolOutput]]


@dataclass(slots=True)
class Tool:
    """A registered tool: a name, description, args schema, and async impl."""

    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFn

    def json_schema(self) -> dict[str, Any]:
        """Render to an OpenAI ``function`` tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    def validate_args(self, raw: dict[str, Any]) -> BaseModel:
        return self.args_model.model_validate(raw)

    async def run(self, args: BaseModel, ctx: ToolContext) -> str | ToolOutput:
        return await self.fn(args, ctx)

    async def __call__(self, args: BaseModel, ctx: ToolContext) -> str | ToolOutput:
        """A decorated tool stays directly awaitable: ``await read_file(a, c)``."""
        return await self.fn(args, ctx)


class ToolRegistry:
    """A name -> Tool map. The global ``REGISTRY`` collects all built-ins;
    a profile builds a ``subset`` exposing only its enabled tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool: {name}") from None

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: list[str]) -> "ToolRegistry":
        sub = ToolRegistry()
        for n in names:
            if n not in self._tools:
                raise ConfigError(
                    f"profile enables unknown tool {n!r}; "
                    f"available: {sorted(self._tools)}"
                )
            sub.register(self._tools[n])
        return sub

    def schemas(self) -> list[dict[str, Any]]:
        return [t.json_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)


REGISTRY = ToolRegistry()


def tool(
    name: str | None = None,
    description: str | None = None,
    registry: ToolRegistry | None = None,
) -> Callable[[ToolFn], Tool]:
    """Decorator: register an async function as a Tool.

    The function's first parameter must be annotated with a pydantic model;
    that model becomes the argument schema advertised to the LLM.
    """

    def decorator(fn: ToolFn) -> Tool:
        try:
            hints = get_type_hints(fn)
        except NameError as e:
            raise ConfigError(
                f"could not resolve type hints for tool {fn.__name__!r}: {e}. "
                "Define the args model at module scope, not inside a function."
            ) from None
        params = [p for p in hints if p != "return"]
        if not params:
            raise ConfigError(
                f"tool {fn.__name__!r} must take an args model as its first parameter"
            )
        args_model = hints[params[0]]
        if not (isinstance(args_model, type) and issubclass(args_model, BaseModel)):
            raise ConfigError(
                f"tool {fn.__name__!r} first parameter must be annotated with a "
                f"pydantic BaseModel, got {args_model!r}"
            )
        built = Tool(
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
            args_model=args_model,
            fn=fn,
        )
        (registry if registry is not None else REGISTRY).register(built)
        return built

    return decorator
