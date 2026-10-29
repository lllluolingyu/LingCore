"""Offline suite resolution and preflight. Never starts a model or container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.evaluation.models import ArtifactText, Case, EvaluationDiagnostic, Suite
from lingcore.sandbox import parse_shell_options


class EvaluationConfigError(ConfigError):
    """An actionable preflight diagnostic that contains no raw input values."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.diagnostic = EvaluationDiagnostic(
            phase="preflight", code=code, message=message
        )


def load_yaml(path: Path, model: type[BaseModel]) -> Any:
    try:
        raw = yaml.safe_load(path.read_text("utf-8"))
        return model.model_validate(raw)
    except ValidationError as exc:
        fields = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'root'}: {error['type']}"
            for error in exc.errors(include_input=False, include_context=False)[:8]
        )
        raise EvaluationConfigError(
            f"invalid {model.__name__} configuration: {path.name} ({fields})",
            "schema_validation",
        ) from None
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # Never echo parser errors: their messages can include input values.
        raise EvaluationConfigError(
            f"cannot load {model.__name__} configuration: {path.name} ({type(exc).__name__})",
            "configuration_read",
        ) from None


@dataclass(frozen=True)
class ResolvedSuite:
    suite: Suite
    cases: tuple[Case, ...]
    profile: AgentProfile


def resolve_suite(path: Path, *, require_credentials: bool = True) -> ResolvedSuite:
    suite: Suite = load_yaml(path, Suite)
    if suite.source.lock.required:
        raise EvaluationConfigError(
            "this suite requires a locked plan; locked execution is milestone 2",
            "locked_plan_required",
        )
    cases = tuple(load_yaml(path.parent / item, Case) for item in suite.source.cases)
    if len({case.id for case in cases}) != len(cases):
        raise EvaluationConfigError("case ids must be unique", "duplicate_case_id")
    try:
        profile = AgentProfile.load(path.parent / suite.agent.profile)
    except ConfigError:
        raise EvaluationConfigError(
            "cannot load the agent profile; run lingcore doctor -p <profile> to inspect its configuration",
            "agent_profile",
        ) from None
    ceiling = set(profile.tools)
    if "run_shell" in ceiling or any(name.startswith("process_") for name in ceiling):
        try:
            sandbox = parse_shell_options(
                profile.tool_options.get("run_shell", {})
            ).sandbox
        except (TypeError, ValueError, ConfigError):
            raise EvaluationConfigError(
                "invalid run_shell sandbox configuration", "sandbox_configuration"
            ) from None
        if sandbox is None:
            raise EvaluationConfigError(
                "native evaluation requires a configured run_shell sandbox",
                "sandbox_required",
            )
    for case in cases:
        missing = set(case.requires.tools) - ceiling
        if missing:
            raise EvaluationConfigError(
                f"case {case.id} requires missing tools: {', '.join(sorted(missing))}",
                "required_tools",
            )
        for grader in case.graders:
            if isinstance(grader, ArtifactText):
                raw = Path(grader.path)
                if raw.is_absolute() or ".." in raw.parts or not raw.name:
                    raise EvaluationConfigError(
                        "artifact paths must stay inside the trial workspace",
                        "artifact_path",
                    )
    # Discover/load only the explicitly engaged code, then validate bindings.
    import lingcore.tools.builtin  # noqa: F401
    from lingcore.plugins.loader import load_plugins
    from lingcore.tools import REGISTRY

    try:
        load_plugins(profile)
        REGISTRY.subset(profile.tools)
    except Exception as exc:
        raise EvaluationConfigError(
            f"cannot load candidate tool or plugin bindings ({type(exc).__name__}); run lingcore doctor -p <profile>",
            "tool_bindings",
        ) from None
    if require_credentials:
        for purpose, cfg in (
            ("main", profile.llm),
            ("vision", profile.media_fallback.image),
        ):
            if cfg is not None:
                try:
                    cfg.resolve_api_key(getattr(profile, "_profile_env", {}))
                except ConfigError:
                    raise EvaluationConfigError(
                        f"{purpose} model credential is unavailable; set the variable named by api_key_env in the profile environment",
                        "model_credentials",
                    ) from None
    return ResolvedSuite(suite, cases, profile)
