"""Built-in tools. Importing this package registers them in the global REGISTRY."""

from lingcore.tools.builtin import fs, shell  # noqa: F401  (registration side effect)

__all__ = ["fs", "shell"]
