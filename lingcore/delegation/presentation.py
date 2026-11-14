"""Compact shared run progress, keeping delegated spend out of context gauges."""

from __future__ import annotations

from typing import Any


def progress_text(value: dict[str, Any]) -> str:
    done, total = value.get("done", 0), value.get("total", 0)
    phase = value.get("phase") or value.get("latest", "")
    text = f"{value.get('run_id', 'run')} · {phase} · {done}/{total} workers · {value.get('status', '')}"
    if value.get("failed"):
        text += f" · {value['failed']} failed/blocked"
    text += f" · workers {value.get('reported_tokens', 0):,} tokens"
    if not value.get("usage_complete", True):
        text += " (usage incomplete)"
    return text[:1000]
