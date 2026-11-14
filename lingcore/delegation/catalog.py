"""Compact static discovery cards for the model-facing delegation tools."""

from __future__ import annotations

import json
from typing import Any

from lingcore.delegation.roles import load_roles
from lingcore.workflows.catalog import discovery_cards


def prompt_catalog(profile: Any) -> str:
    lines = [
        "Delegation catalog (role tools still intersect the current dispatch authorization):"
    ]
    for name, role in sorted(load_roles(profile).items()):
        lines.append(
            json.dumps(
                {
                    "role": name,
                    "description": role.description[:200],
                    "workspace": role.workspace,
                    "model": role.model,
                },
                ensure_ascii=False,
            )
        )
    for card in discovery_cards(profile):
        lines.append(json.dumps(card, ensure_ascii=False))
    lines.append("Model aliases: " + ", ".join(sorted(profile.delegation.models)))
    result = []
    size = 0
    for line in lines:
        if size + len(line) > 16000:
            result.append(
                "Additional catalog entries omitted; use an explicitly supplied role/workflow name."
            )
            break
        result.append(line)
        size += len(line)
    return "\n".join(result)
