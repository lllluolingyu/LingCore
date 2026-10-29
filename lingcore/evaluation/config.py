"""Offline suite resolution and preflight. Never starts a model or container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.evaluation.graders import evidence_needs
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


def resolve_suite(
    path: Path, *, require_credentials: bool = True, allow_locked: bool = False
) -> ResolvedSuite:
    suite: Suite = load_yaml(path, Suite)
    if suite.source.lock.required and not allow_locked:
        raise EvaluationConfigError(
            "this suite requires a locked plan; use eval resolve and eval run --lock",
            "locked_plan_required",
        )
    cases = tuple(load_yaml(path.parent / item, Case) for item in suite.source.cases)
    unknown = set(suite.source.lock.capture) - {
        "tasks",
        "candidate",
        "grader",
        "runtime",
        "execution",
        "trace",
        "grading",
        "source",
        "trials",
    }
    if unknown:
        raise EvaluationConfigError(
            "unknown native lock capture requirement", "lock_capture"
        )
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
        if (
            evidence_needs(case.graders)["content"]
            and suite.trace.content != "redacted"
        ):
            raise EvaluationConfigError(
                f"case {case.id}: tool_policy argument predicates require redacted trace content",
                "content_required",
            )
        for path in [*case.evidence.artifacts, *case.fixture.files]:
            raw_path = Path(path)
            if raw_path.is_absolute() or ".." in raw_path.parts or not raw_path.name:
                raise EvaluationConfigError(
                    "evidence and fixture paths must stay inside the workspace",
                    "artifact_path",
                )
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
        import os

        from lingcore.doctor import required_environment_names

        environment = {**os.environ, **getattr(profile, "_profile_env", {})}
        missing_environment = sorted(
            name
            for name in required_environment_names(profile)
            if not environment.get(name)
        )
        if missing_environment:
            from lingcore.tracing.config import known_secrets
            from lingcore.tracing.content import redact_known_secrets

            raise EvaluationConfigError(
                "required environment variables are unavailable: "
                + ", ".join(
                    redact_known_secrets(name, known_secrets(profile))
                    for name in missing_environment
                ),
                "model_credentials",
            )
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
