"""Atomic, profile-owned plugin skeleton creation."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import yaml

from lingcore.errors import ConfigError
from lingcore.plugins.manifest import PLUGIN_NAME

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def scaffold_plugin(
    profile_dir: Path,
    name: str,
    *,
    hooks: bool = True,
    commands: bool = True,
) -> Path:
    """Create a complete local plugin, refusing existing/immutable destinations."""
    if not PLUGIN_NAME.fullmatch(name):
        raise ConfigError("plugin name must match [a-z0-9][a-z0-9-]{0,39}")
    target = (profile_dir / "plugins" / name).resolve()
    if target.is_relative_to(_PACKAGE_DIR.resolve()):
        raise ConfigError("plugin destination is inside the installed package tree")
    if target.exists() or (profile_dir / "plugins" / name).is_symlink():
        raise ConfigError(f"plugin destination already exists: {target}")
    prefix = name.replace("-", "_")
    manifest: dict[str, object] = {
        "name": name,
        "version": "0.1.0",
        "api": 1,
        "min_lingcore": "0.4.0",
        "description": f"A local {name} plugin.",
        "module": "plugin.py",
        "provides": [f"{prefix}_echo"],
        "skills": "skills",
    }
    if hooks:
        manifest["hooks"] = "Hooks"
    if commands:
        manifest["commands"] = "commands"
    code = (
        '"""Tools and optional per-Agent hooks for this plugin."""\n\n'
        "from __future__ import annotations\n\n"
        "from pydantic import BaseModel\n\n"
        + (
            "from lingcore.plugins import PluginHooks, ToolCallEvent, ToolDecision\n"
            if hooks
            else ""
        )
        + "from lingcore.tools import ToolContext, tool\n\n\n"
        "class EchoArgs(BaseModel):\n"
        "    text: str\n\n\n"
        f'@tool(name="{prefix}_echo")\n'
        "async def echo(args: EchoArgs, ctx: ToolContext) -> str:\n"
        '    """Echo the supplied text."""\n'
        "    return args.text\n"
    )
    if hooks:
        code += (
            "\n\nclass Hooks(PluginHooks):\n"
            '    """One instance per Agent; parallel tool hooks may overlap."""\n\n'
            "    async def before_tool(self, event: ToolCallEvent) -> ToolDecision | None:\n"
            "        # Example policy: uncomment to block shell calls.\n"
            '        # if event.name == "run_shell":\n'
            '        #     return ToolDecision.deny("Shell is blocked by this plugin.")\n'
            "        return None\n"
        )
    files = {
        "plugin.yaml": yaml.safe_dump(manifest, sort_keys=False),
        "plugin.py": code,
        f"skills/{name}/skill.md": (
            "---\n"
            f"name: {name}\n"
            f"description: Echo text with the {name} plugin.\n"
            f"requested_tools: [{prefix}_echo]\n"
            "---\n\n"
            f"Use `{prefix}_echo` to echo text when the user asks.\n"
        ),
        "README.md": (
            f"# {name}\n\n"
            "Enable this plugin in the profile's config.yaml:\n\n"
            "```yaml\n"
            f"plugins: [{name}]\n"
            f"tools: [{prefix}_echo]\n"
            "```\n\n"
            "Merge these entries with your existing lists. Registering a tool does "
            "not authorize it; the profile's tools list remains the ceiling.\n\n"
            + (
                f"Try `/{name}:hello world` (or `/hello world` when unambiguous).\n\n"
                if commands
                else ""
            )
            + (
                "The hooks class is a no-op. Its commented policy demonstrates how "
                "to deny a tool; hook instances belong to one Agent and must protect "
                "their own state when parallel_tools is enabled.\n\n"
                if hooks
                else ""
            )
            + "Store secret values in the profile .env. Declare variable names in "
            "plugin.yaml environment entries. See docs/plugins.md for the stable "
            "API and pip entry-point packaging.\n"
        ),
    }
    if commands:
        files["commands/hello.md"] = (
            "---\n"
            "description: Greet someone using the echo tool.\n"
            "argument_hint: '[name]'\n"
            "---\n\n"
            f"Greet $ARGUMENTS using the {prefix}_echo tool.\n"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=target.parent))
        try:
            for relative, text in files.items():
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, "utf-8")
            if target.exists():
                raise ConfigError(f"plugin destination already exists: {target}")
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    except OSError as exc:
        raise ConfigError(
            f"could not create plugin at {target}: {type(exc).__name__}"
        ) from None
    return target
