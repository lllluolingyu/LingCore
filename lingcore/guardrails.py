"""Guardrails: optional pre-input / post-output hooks.

Core to the framework but a no-op by default. The coding agent runs
``NoopGuardrail``; a psych-consultant profile would supply a real one
(crisis-phrase detection, disclaimers, refusal of out-of-scope advice).
Kept a single module deliberately — promote to a package only when a real
guardrail set ships.
"""

from __future__ import annotations

from typing import Protocol


class Guardrail(Protocol):
    async def pre_input(self, text: str) -> str:
        """Inspect/transform user input before it reaches the model."""
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
