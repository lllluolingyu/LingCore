"""Named profile and explicitly consented plugin workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from lingcore.delegation.roles import read_asset
from lingcore.errors import ConfigError
from lingcore.workflows.spec import WorkflowSpec

if TYPE_CHECKING:
    from lingcore.config import AgentProfile


def load_catalog(profile: AgentProfile) -> dict[str, WorkflowSpec]:
    catalog: dict[str, WorkflowSpec] = {}
    roots: list[tuple[Path, str]] = []
    source = getattr(profile, "_source_dir", None)
    if source:
        roots.append((source, profile.workflows.dir))
    from lingcore.plugins.discovery import discover_plugins

    found = discover_plugins(source)
    for name in profile.plugins:
        plugin = found.get(name)
        if plugin is None:
            raise ConfigError(f"missing workflow plugin: {name}")
        directory = getattr(plugin.manifest, "workflows", None)
        if directory:
            roots.append((plugin.root, directory))
    for root, relative in roots:
        directory = root / relative
        if not directory.exists():
            continue
        if directory.is_symlink():
            raise ConfigError("workflow directory cannot be a symlink")
        for path in sorted(directory.glob("*.yaml")):
            spec = parse_spec(read_asset(root, str(path.relative_to(root)), 262144))
            if spec.name in catalog:
                raise ConfigError(f"duplicate workflow: {spec.name}")
            catalog[spec.name] = spec
    return catalog


def discovery_cards(profile: AgentProfile) -> list[dict[str, Any]]:
    """Runtime discovery may import explicitly consented operator Python only."""
    from typing import get_type_hints

    from lingcore.workflows.api import plugin_workflows

    cards = {
        name: {
            "workflow": name,
            "description": spec.description[:200],
            "when_to_use": spec.when_to_use[:200],
            "inputs": spec.inputs,
        }
        for name, spec in load_catalog(profile).items()
    }
    for name, definition in plugin_workflows(profile).items():
        if name in cards:
            raise ConfigError(f"duplicate workflow: {name}")
        model = get_type_hints(definition.fn).get("args")
        cards[name] = {
            "workflow": name,
            "description": definition.description[:200],
            "inputs": model.model_json_schema()
            if model and hasattr(model, "model_json_schema")
            else {"type": "object"},
        }
    return [cards[name] for name in sorted(cards)]


def parse_spec(value: str | dict[str, Any]) -> WorkflowSpec:
    if isinstance(value, str):
        if len(value) > 262144:
            raise ConfigError("inline workflow exceeds 256 KiB")
        value = yaml.safe_load(value)
    count = 0
    active: set[int] = set()

    def bounded(item: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if (
            depth > 24
            or count > 20000
            or (isinstance(item, str) and len(item) > 262144)
        ):
            raise ConfigError("workflow data exceeds structural bounds")
        if isinstance(item, (dict, list)):
            if id(item) in active:
                raise ConfigError("recursive workflow YAML is forbidden")
            active.add(id(item))
            for child in item.values() if isinstance(item, dict) else item:
                bounded(child, depth + 1)
            active.remove(id(item))

    bounded(value, 0)
    return WorkflowSpec.model_validate(value)
