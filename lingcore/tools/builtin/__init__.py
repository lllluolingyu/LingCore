"""Built-in tools. Importing this package registers them in the global REGISTRY."""

from lingcore.tools.builtin import fs, patch, shell, web  # noqa: F401  (registration side effect)

__all__ = ["fs", "patch", "shell", "web"]
