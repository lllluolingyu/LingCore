"""Offline delegation/service findings, without constructing providers or stores."""

from __future__ import annotations

from typing import Any

from lingcore._doctor_models import DoctorFinding
from lingcore.delegation.retention import retained_size
from lingcore.delegation.validation import validate_definitions
from lingcore.delegation.workspaces import WorkspaceManager


def findings(profile: Any) -> list[DoctorFinding]:
    result = []
    try:
        roles, workflows = validate_definitions(profile)
        result.append(
            DoctorFinding(
                "ok", f"delegation: {roles} roles, {len(workflows)} workflows"
            )
        )
        for name, bound in workflows.items():
            result.append(
                DoctorFinding("info", f"workflow {name}: at most {bound} workers")
            )
        size = retained_size(
            WorkspaceManager(profile.workspace_path(), profile.delegation.workspace)
        )
        if size["files"]:
            result.append(
                DoctorFinding(
                    "info",
                    f"retained runs: {size['files']} artifact files, {size['bytes']} bytes; prune with lingcore run prune",
                )
            )
        if profile.superagent:
            result.append(
                DoctorFinding("ok", "superagent executor sandbox and roles validated")
            )
            for name, source in profile.superagent.sources.items():
                if source.enabled and not source.allowlist:
                    result.append(
                        DoctorFinding(
                            "warning",
                            f"superagent {name}: empty actor allowlist admits no events",
                        )
                    )
    except Exception as exc:
        result.append(DoctorFinding("error", f"delegation configuration: {exc}"))
    return result


def environment(profile: Any) -> dict[str, str]:
    result = {}
    for name, model in profile.delegation.models.items():
        if model.get("api_key_env"):
            result[model["api_key_env"]] = f"delegation.models.{name}"
    if profile.superagent:
        root = getattr(profile, "_source_dir", None)
        for name, source in profile.superagent.sources.items():
            if not source.enabled:
                continue
            path = root / source.config if root and source.config else None
            if name == "lark":
                from lingcore.integrations.lark.config import load_lark_config

                config = load_lark_config(
                    profile, path, require_secrets=False, transport_only=True
                )
                result[config.app_id_env] = "superagent.lark"
                result[config.app_secret_env] = "superagent.lark"
            elif name == "telegram":
                from lingcore.integrations.telegram.config import load_telegram_config

                telegram = load_telegram_config(
                    profile, path, require_secrets=False, transport_only=True
                )
                result[telegram.token_env] = "superagent.telegram"
            elif name == "github":
                result[source.options.get("token_env", "GITHUB_TOKEN")] = (
                    "superagent.github"
                )
    return result
