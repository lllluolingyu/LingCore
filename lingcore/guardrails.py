"""Guardrails: optional pre-input / post-output hooks.

Core to the framework but a no-op by default. Profiles can select an installed
``lingcore.guardrails`` entry point by name or a Python target such as
``my_package.guardrails:PsychGuardrail``. The target may be a class, factory,
or ready-made object implementing :class:`Guardrail`.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

from lingcore.errors import ConfigError

ENTRY_POINT_GROUP = "lingcore.guardrails"


@runtime_checkable
class Guardrail(Protocol):
    async def pre_input(self, text: str) -> str:
        """Inspect/transform user text before it reaches the model.

        V1 guardrails inspect text only. Multimodal attachment bytes are
        validated at frontend/tool boundaries and then passed through to the
        model without guardrail inspection.
        """
        ...

    async def post_output(self, text: str) -> str:
        """Inspect/transform the model's final reply before it is shown."""
        ...


class NoopGuardrail:
    """Pass-through guardrail. The default for tool-only agents like coding."""

    async def pre_input(self, text: str) -> str:
        return text

    async def post_output(self, text: str) -> str:
        return text


def _entry_point_target(policy: str) -> Any | None:
    matches = [
        entry
        for entry in metadata.entry_points(group=ENTRY_POINT_GROUP)
        if entry.name == policy
    ]
    if len(matches) > 1:
        raise ConfigError(
            f"multiple {ENTRY_POINT_GROUP!r} entry points are named {policy!r}"
        )
    if not matches:
        return None
    try:
        return matches[0].load()
    except Exception as exc:
        raise ConfigError(
            f"could not load guardrail entry point {policy!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from None


def _dotted_target(policy: str) -> Any:
    if ":" in policy:
        module_name, _, attribute = policy.partition(":")
    else:
        module_name, _, attribute = policy.rpartition(".")
    if not module_name or not attribute:
        raise ConfigError(f"unknown guardrail policy: {policy!r}")
    try:
        target: Any = importlib.import_module(module_name)
        for part in attribute.split("."):
            target = getattr(target, part)
        return target
    except (ImportError, AttributeError) as exc:
        raise ConfigError(
            f"could not load guardrail {policy!r}: {type(exc).__name__}: {exc}"
        ) from None


def build_guardrail(policy: str, options: dict[str, Any] | None = None) -> Guardrail:
    """Build a profile-selected guardrail without changing LingCore code.

    ``noop`` is the built-in default. Other short names resolve through the
    ``lingcore.guardrails`` entry-point group; ``module:attribute`` and
    ``module.attribute`` values load a Python target directly. Classes and
    factories receive ``guardrail.options`` as keyword arguments.
    """
    if policy == "noop":
        if options:
            raise ConfigError("the noop guardrail does not accept options")
        return NoopGuardrail()

    target = _entry_point_target(policy)
    if target is None:
        target = _dotted_target(policy)
    kwargs = options or {}
    try:
        if isinstance(target, type) or not isinstance(target, Guardrail):
            if not callable(target):
                raise TypeError("target is neither a guardrail nor a factory")
            candidate = target(**kwargs)
        else:
            if kwargs:
                raise TypeError("a guardrail object cannot receive options")
            candidate = target
    except Exception as exc:
        raise ConfigError(
            f"could not construct guardrail {policy!r}: {type(exc).__name__}: {exc}"
        ) from None
    if not isinstance(candidate, Guardrail):
        raise ConfigError(
            f"guardrail {policy!r} must provide pre_input() and post_output()"
        )
    return candidate
