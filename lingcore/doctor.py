"""Offline, secret-safe diagnostics for a loaded agent profile.

``lingcore doctor`` validates the configuration needed to *start* a profile
without opening its session database, creating a workspace, building an agent,
or contacting any provider. Diagnostics name variables and their source but
never include values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.knowledge import (
    DEFAULT_API_KEY_ENV,
    embedding_options,
    reranker_options,
)

DoctorLevel = Literal["ok", "info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class DoctorFinding:
    level: DoctorLevel
    message: str


@dataclass(frozen=True, slots=True)
class DoctorReport:
    profile_name: str
    findings: tuple[DoctorFinding, ...]

    @property
    def errors(self) -> tuple[DoctorFinding, ...]:
        return tuple(finding for finding in self.findings if finding.level == "error")

    @property
    def warnings(self) -> tuple[DoctorFinding, ...]:
        return tuple(
            finding for finding in self.findings if finding.level == "warning"
        )

    @property
    def exit_code(self) -> int:
        return 1 if self.errors else 0


def _add_requirement(
    requirements: dict[str, set[str]], name: str | None, consumer: str
) -> None:
    if name is None or not name.strip():
        return
    requirements.setdefault(name.strip(), set()).add(consumer)


def _add_provider_requirement(
    requirements: dict[str, set[str]],
    config_errors: list[str],
    options: Mapping[str, object],
    consumer: str,
) -> None:
    raw_name = options.get("api_key_env", DEFAULT_API_KEY_ENV)
    if not isinstance(raw_name, str) or not raw_name.strip():
        config_errors.append(f"{consumer} must name an environment variable")
        return
    _add_requirement(requirements, raw_name, consumer)


def _profile_requirements(
    profile: AgentProfile,
) -> tuple[dict[str, set[str]], set[str], list[str]]:
    """Return runtime env requirements, example keys, and config errors."""
    requirements: dict[str, set[str]] = {}
    example_names: set[str] = set()
    config_errors: list[str] = []

    _add_requirement(requirements, profile.llm.api_key_env, "llm.api_key_env")
    vision = profile.media_fallback.image
    if vision is not None:
        _add_requirement(
            requirements, vision.api_key_env, "media_fallback.image.api_key_env"
        )

    canvas_enabled = "canvas" in profile.skills or any(
        name.startswith("canvas_") for name in profile.tools
    )
    if canvas_enabled:
        example_names.add("CANVAS_URL")
        raw_canvas = profile.tool_options.get("canvas", {})
        if not isinstance(raw_canvas, Mapping):
            config_errors.append("tool_options.canvas must be a mapping")
        else:
            base_url = raw_canvas.get("base_url", "")
            if not isinstance(base_url, str) or not base_url.strip():
                config_errors.append(
                    "Canvas base_url is empty; set CANVAS_URL in the profile "
                    ".env or configure tool_options.canvas.base_url"
                )
            raw_token_env = raw_canvas.get("token_env", "CANVAS_TOKEN")
            if not isinstance(raw_token_env, str) or not raw_token_env.strip():
                config_errors.append(
                    "tool_options.canvas.token_env must name an environment variable"
                )
            else:
                _add_requirement(
                    requirements, raw_token_env, "tool_options.canvas.token_env"
                )

    if "knowledge" in profile.tools:
        raw_knowledge = profile.tool_options.get("knowledge", {})
        if not isinstance(raw_knowledge, Mapping):
            config_errors.append("tool_options.knowledge must be a mapping")
        else:
            backend = raw_knowledge.get("backend", "grep")
            if backend not in {"grep", "index", "hybrid"}:
                config_errors.append(
                    "tool_options.knowledge.backend must be grep, index, or hybrid"
                )
            elif backend in {"index", "hybrid"}:
                try:
                    embedding = embedding_options(raw_knowledge)
                    reranker = reranker_options(raw_knowledge)
                except ConfigError as exc:
                    config_errors.append(str(exc))
                else:
                    if not embedding.get("enabled", False):
                        config_errors.append(
                            "knowledge backend requires embedding.enabled: true"
                        )
                    else:
                        _add_provider_requirement(
                            requirements,
                            config_errors,
                            embedding,
                            "tool_options.knowledge.embedding.api_key_env",
                        )
                    if reranker.get("enabled", False):
                        _add_provider_requirement(
                            requirements,
                            config_errors,
                            reranker,
                            "tool_options.knowledge.reranker.api_key_env",
                        )

    example_names.update(requirements)
    return requirements, example_names, config_errors


def required_example_names(profile: AgentProfile) -> frozenset[str]:
    """Names a profile's ``.env.example`` must document."""
    _, example_names, _ = _profile_requirements(profile)
    return frozenset(example_names)


def _example_findings(path: Path, required: set[str]) -> list[DoctorFinding]:
    if not required:
        return [DoctorFinding("info", "no .env.example is required for this profile")]
    if not path.is_file():
        return [
            DoctorFinding(
                "warning",
                f"missing {path}; add a secret-free template documenting "
                + ", ".join(sorted(required)),
            )
        ]
    try:
        parsed = dotenv_values(path, interpolate=False, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [
            DoctorFinding(
                "warning", f"could not read .env.example at {path}: {exc}"
            )
        ]
    missing = sorted(required - set(parsed))
    if missing:
        return [
            DoctorFinding(
                "warning",
                f"{path} does not document: " + ", ".join(missing),
            )
        ]
    return [DoctorFinding("ok", f".env.example documents {len(required)} variable(s)")]


def diagnose_profile(profile: AgentProfile) -> DoctorReport:
    """Inspect one loaded profile without side effects or provider calls."""
    source_dir = getattr(profile, "_source_dir", None)
    profile_env: Mapping[str, str] = getattr(profile, "_profile_env", {})
    findings: list[DoctorFinding] = [
        DoctorFinding("ok", f"profile {profile.name!r} loaded and validated")
    ]

    if source_dir is None:
        findings.append(
            DoctorFinding("warning", "profile has no source directory")
        )
    else:
        env_path = source_dir / ".env"
        if env_path.is_file():
            findings.append(DoctorFinding("ok", f"profile environment: {env_path}"))
        else:
            findings.append(
                DoctorFinding(
                    "info", "profile .env is absent; exported values/defaults apply"
                )
            )

    requirements, example_names, config_errors = _profile_requirements(profile)
    findings.extend(DoctorFinding("error", message) for message in config_errors)
    for name, consumers in sorted(requirements.items()):
        used_by = ", ".join(sorted(consumers))
        if name in profile_env:
            if profile_env[name]:
                findings.append(
                    DoctorFinding(
                        "ok", f"{name} is set by profile .env (used by {used_by})"
                    )
                )
            else:
                findings.append(
                    DoctorFinding(
                        "error",
                        f"{name} is empty in profile .env (required by {used_by})",
                    )
                )
        elif os.environ.get(name):
            findings.append(
                DoctorFinding(
                    "ok", f"{name} is set by process environment (used by {used_by})"
                )
            )
        elif name in os.environ:
            findings.append(
                DoctorFinding(
                    "error",
                    f"{name} is empty in process environment (required by {used_by})",
                )
            )
        else:
            findings.append(
                DoctorFinding(
                    "error", f"{name} is missing (required by {used_by})"
                )
            )

    if source_dir is not None:
        findings.extend(
            _example_findings(source_dir / ".env.example", example_names)
        )
    return DoctorReport(profile_name=profile.name, findings=tuple(findings))


def print_doctor_report(report: DoctorReport) -> None:
    """Render a deterministic, value-free report to stdout."""
    print(f"LingCore doctor · profile {report.profile_name}")
    for finding in report.findings:
        print(f"[{finding.level}] {finding.message}")
    if report.errors:
        print(
            f"doctor found {len(report.errors)} error(s) and "
            f"{len(report.warnings)} warning(s)"
        )
    else:
        print(f"doctor passed with {len(report.warnings)} warning(s)")
