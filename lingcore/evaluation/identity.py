"""Measured offline plan identity. Secrets are checked before any hashing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field

import lingcore
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.doctor import required_environment_names
from lingcore.evaluation.config import EvaluationConfigError, ResolvedSuite
from lingcore.evaluation.graders import GRADERS, evidence_needs
from lingcore.evaluation.models import (
    RetryCause,
    StrictModel,
    Suite,
    TracePolicy,
    TrialSpec,
    default_retry_on,
)
from lingcore.evaluation.storage import canonical_json, directory, read_bytes
from lingcore.plugins.discovery import discover_plugins, engaged_plugins, local_skills
from lingcore.tracing import NoopRecorder
from lingcore.tracing.config import known_secrets

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
_PACKAGE = Path(lingcore.__file__).resolve().parent


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def value_digest(value: Any) -> str:
    return digest(canonical_json(value))


def assert_secret_free(data: bytes, secrets: tuple[str, ...]) -> None:
    """Long secrets match substrings; short ones match complete JSON strings."""
    text = data.decode("utf-8", errors="replace")
    try:
        value = json.loads(text)
    except ValueError:
        value = text

    def strings(item: Any) -> Any:
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for key, child in item.items():
                yield str(key)
                yield from strings(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                yield from strings(child)

    if any(
        secret and (value == secret or len(secret) >= 8 and secret in value)
        for value in strings(value)
        for secret in secrets
    ):
        raise EvaluationConfigError(
            "known credential value is present in evaluation inputs", "secret_in_plan"
        )


class GraderIdentity(StrictModel):
    id: str
    kind: str
    version: int


class EvidenceNeeds(StrictModel):
    artifacts: list[str] = Field(default_factory=list)
    content: bool = False
    metadata: bool = False


class CaseIdentity(StrictModel):
    id: str
    revision: int | str
    path: str = ""
    input_digest: Digest
    graders_digest: Digest
    graders: list[GraderIdentity]
    evidence: EvidenceNeeds


class LocalSourceIdentity(StrictModel):
    kind: Literal["local"] = "local"
    digest: Digest
    cases: list[CaseIdentity]


class ModelIdentity(StrictModel):
    backend: str
    id: str
    endpoint: str


class ProfileIdentity(StrictModel):
    path: str
    config_digest: Digest


class ToolIdentity(StrictModel):
    name: str
    schema_digest: Digest


class ToolsIdentity(StrictModel):
    ceiling: list[ToolIdentity]
    initial: list[str]


class CredentialIdentity(StrictModel):
    name: str
    set: bool


class PluginIdentity(StrictModel):
    version: str | None
    root_digest: Digest


class CandidateIdentity(StrictModel):
    digest: Digest
    profile: ProfileIdentity
    model: ModelIdentity
    vision_model: ModelIdentity | None
    sampling: dict[str, Any]
    prompt_digest: Digest
    tools: ToolsIdentity
    sources: dict[str, str]
    plugins: dict[str, PluginIdentity] = Field(default_factory=dict)
    credentials: list[CredentialIdentity]


class RuntimeIdentity(StrictModel):
    digest: Digest
    lingcore: str
    package_digest: Digest
    git: dict[str, Any] | None
    python: str
    dependencies: dict[str, str]


class SuiteIdentity(StrictModel):
    name: str
    track: str
    path: str
    digest: Digest


class RetryPolicy(StrictModel):
    infrastructure_retries: int = Field(default=0, ge=0, le=5)
    retry_on: list[RetryCause] = Field(default_factory=default_retry_on)


class ExecutionIdentity(StrictModel):
    digest: Digest
    runner: Literal["native"] = "native"
    state: Literal["fresh_per_execution"] = "fresh_per_execution"
    approval_policy: Literal["deny"] = "deny"
    repetitions: int
    order: Literal["declared"] = "declared"
    concurrency: Literal[1] = 1
    retry: RetryPolicy
    limits: dict[str, Any]


class GradingIdentity(StrictModel):
    digest: Digest
    authority: Literal["local"] = "local"
    success: Literal["all_required"] = "all_required"


class PlanIdentity(StrictModel):
    suite: SuiteIdentity
    source: Annotated[LocalSourceIdentity, Field(discriminator="kind")]
    candidate: CandidateIdentity
    runtime: RuntimeIdentity
    execution: ExecutionIdentity
    trace: TracePolicy
    grading: GradingIdentity
    trials: list[TrialSpec]
    limitations: list[str] = Field(default_factory=list)


def identity_body(plan: PlanIdentity) -> dict[str, Any]:
    value = {
        key: value
        for key, value in plan.model_dump(mode="json").items()
        if key in PlanIdentity.model_fields
    }
    value.pop("limitations")
    value["suite"].pop("path")
    value["candidate"]["profile"].pop("path")
    for credential in value["candidate"]["credentials"]:
        credential.pop("set")
    value["runtime"].pop("git")
    for case in value["source"]["cases"]:
        case.pop("path")
    return value


def case_identity(case: Any) -> CaseIdentity:
    versions = [
        GraderIdentity(id=g.id, kind=g.kind, version=GRADERS[g.kind].version)
        for g in case.graders
    ]
    return CaseIdentity(
        id=case.id,
        revision=case.revision,
        input_digest=value_digest(
            case.model_dump(mode="json", exclude={"tags", "graders", "evidence"})
        ),
        graders_digest=value_digest(
            {
                "specs": [g.model_dump(mode="json") for g in case.graders],
                "versions": [g.model_dump() for g in versions],
                "extra": case.evidence.model_dump(),
            }
        ),
        graders=versions,
        evidence=EvidenceNeeds(**evidence_needs(case.graders, case.evidence.artifacts)),
    )


def endpoint(url: str, alias: str | None = None) -> str:
    if alias is not None:
        return alias
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    return host + (f":{parsed.port}" if parsed.port else "")


def _profile_values(profile: AgentProfile, suite: Suite) -> dict[str, Any]:
    values = profile.model_dump(mode="json", exclude={"workspace", "tracing"})

    def normalize(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "base_url" and isinstance(child, str):
                    value[key] = endpoint(child, suite.agent.endpoint_alias)
                else:
                    normalize(child)
        elif isinstance(value, list):
            for child in value:
                normalize(child)

    normalize(values)
    return values


def _tree(root: Path, secrets: tuple[str, ...] = ()) -> dict[str, str]:
    result: dict[str, str] = {}

    def walk(current: Any, prefix: str = "") -> None:
        for name, is_dir, is_file, is_link in sorted(current.iter_entries()):
            if name == "__pycache__" or name == ".env" or name.startswith(".git"):
                continue
            if is_link:
                raise EvaluationConfigError(
                    "identity source contains a symlink", "unsafe_source"
                )
            relative = prefix + name
            if is_dir:
                with current.subdirectory(name) as child:
                    walk(child, relative + "/")
            elif is_file:
                data = current.read_regular(name, max_bytes=16 * 1024 * 1024)
                assert_secret_free(data, secrets)
                result[relative] = digest(data)

    with directory(root) as current:
        walk(current)
    return result


class _OfflineClient:
    def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("identity resolution must never invoke a model")


async def candidate_identity(
    profile: AgentProfile, suite: Suite, *, profile_path: Path | None = None
) -> CandidateIdentity:
    secrets = known_secrets(profile)
    values = _profile_values(profile, suite)
    assert_secret_free(canonical_json(values), secrets)
    sources: dict[str, str] = {}
    plugins: dict[str, PluginIdentity] = {}
    source = getattr(profile, "_source_dir", None)
    if source is not None:
        # YAML is parsed input: comments/ordering are not candidate drift.
        sources[(profile_path or source / "config.yaml").name] = value_digest(values)
        for name in ["world.md", "role.md", "workflow.md", *profile.persona.include]:
            path = source / name
            if path.is_file():
                data = read_bytes(path)
                assert_secret_free(data, secrets)
                sources[os.path.relpath(path, source)] = digest(data)
        if (source / "commands").exists():
            sources.update(
                {
                    "commands/" + k: v
                    for k, v in _tree(source / "commands", secrets).items()
                }
            )
        for name, skill in local_skills(profile).items():
            if skill.source_dir and (
                "activate_skill" in profile.tools
                or name in profile.skills
                or set(skill.provides) & set(profile.tools)
            ):
                prefix = skill.source_dir.relative_to(source).as_posix()
                sources.update(
                    {
                        prefix + "/" + k: v
                        for k, v in _tree(skill.source_dir, secrets).items()
                    }
                )
        discovered = discover_plugins(source)
        for name in engaged_plugins(profile, discovered):
            plugin = discovered[name]
            if plugin.source == "local":
                prefix = plugin.root.relative_to(source).as_posix()
                sources.update(
                    {
                        prefix + "/" + k: v
                        for k, v in _tree(plugin.root, secrets).items()
                    }
                )
            elif plugin.source == "installed":
                version = next(
                    (
                        ep.dist.version
                        for ep in metadata.entry_points(group="lingcore.plugins")
                        if ep.name == name and ep.dist is not None
                    ),
                    None,
                )
                plugins[name] = PluginIdentity(
                    version=version,
                    root_digest=value_digest(_tree(plugin.root, secrets)),
                )
    credentials = [
        CredentialIdentity(
            name=name,
            set=bool(
                getattr(profile, "_profile_env", {}).get(name, os.environ.get(name))
            ),
        )
        for name in sorted(required_environment_names(profile))
    ]
    with tempfile.TemporaryDirectory(prefix="lingcore-identity-") as temporary:
        measured = profile.model_copy(deep=True)
        measured.workspace = str(Path(temporary) / "workspace")
        measured.tracing = measured.tracing.model_copy(update={"enabled": False})
        agent = Agent.from_profile(
            measured,
            llm=_OfflineClient(),
            vision_llm=_OfflineClient(),
            state_dir=Path(temporary) / "state",
            recorder=NoopRecorder(),
        )
        try:
            prompt = await agent.initial_system_prompt()
            schemas = agent.ceiling_tool_schemas()
            assert_secret_free(
                canonical_json({"prompt": prompt, "schemas": schemas}), secrets
            )
            tools = ToolsIdentity(
                ceiling=[
                    ToolIdentity(
                        name=s["function"]["name"], schema_digest=value_digest(s)
                    )
                    for s in sorted(schemas, key=lambda s: s["function"]["name"])
                ],
                initial=sorted(agent.initial_tools),
            )
        finally:
            await agent.aclose()
    model = ModelIdentity(
        backend=profile.llm.backend,
        id=profile.llm.model,
        endpoint=endpoint(profile.llm.base_url, suite.agent.endpoint_alias),
    )
    vision = profile.media_fallback.image
    fields: dict[str, Any] = dict(
        profile=ProfileIdentity(path="", config_digest=value_digest(values)),
        model=model,
        vision_model=ModelIdentity(
            backend=vision.backend,
            id=vision.model,
            endpoint=endpoint(vision.base_url, suite.agent.endpoint_alias),
        )
        if vision
        else None,
        sampling=profile.llm.sampling.model_dump(mode="json"),
        prompt_digest=digest(prompt.encode()),
        tools=tools,
        sources=sources,
        plugins=plugins,
        credentials=credentials,
    )
    # Pydantic serializes nested model collections consistently before hashing.
    provisional = CandidateIdentity(digest="sha256:" + "0" * 64, **fields)
    body = provisional.model_dump(
        mode="json", exclude={"digest", "credentials", "profile"}
    ) | {
        "config_digest": provisional.profile.config_digest,
        "credentials": [c.name for c in credentials],
    }
    return provisional.model_copy(update={"digest": value_digest(body)})


def runtime_identity() -> RuntimeIdentity:
    dependencies: dict[str, str] = {}
    for requirement in metadata.requires("lingcore") or ():
        if "extra ==" in requirement:
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if match:
            name = match[0]
            try:
                dependencies[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                dependencies[name] = "unavailable"
    git = None
    try:
        top = (
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=_PACKAGE,
                capture_output=True,
                timeout=2,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        if not top or Path(top).resolve() != _PACKAGE.parent.resolve():
            raise ValueError("package is not at the checkout root")
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=_PACKAGE,
                capture_output=True,
                timeout=2,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "lingcore"],
            cwd=_PACKAGE.parent,
            capture_output=True,
            timeout=2,
            check=True,
        ).stdout
        git = {"commit": commit, "dirty": bool(dirty)}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    fields: dict[str, Any] = dict(
        lingcore=lingcore.__version__,
        package_digest=value_digest(sorted(_tree(_PACKAGE).items())),
        python=f"{platform.python_implementation()} {platform.python_version()}",
        dependencies=dependencies,
    )
    return RuntimeIdentity(digest=value_digest(fields), git=git, **fields)


async def build_identity(
    resolved: ResolvedSuite, suite_path: Path, base_dir: Path
) -> PlanIdentity:
    suite = resolved.suite
    secrets = known_secrets(resolved.profile)
    assert_secret_free(canonical_json(suite.model_dump(mode="json")), secrets)
    cases = []
    for case, path in zip(resolved.cases, suite.source.cases, strict=True):
        assert_secret_free(canonical_json(case.model_dump(mode="json")), secrets)
        cases.append(
            case_identity(case).model_copy(
                update={"path": os.path.relpath(suite_path.parent / path, base_dir)}
            )
        )
    profile_path = (suite_path.parent / suite.agent.profile).resolve()
    if profile_path.is_dir():
        profile_path /= "config.yaml"
    candidate = await candidate_identity(
        resolved.profile, suite, profile_path=profile_path
    )
    candidate = candidate.model_copy(
        update={
            "profile": candidate.profile.model_copy(
                update={"path": os.path.relpath(profile_path, base_dir)}
            )
        }
    )
    runtime = await asyncio.to_thread(runtime_identity)
    execution = suite.execution.model_dump(
        mode="json", exclude={"infrastructure_retries"}
    )
    execution["retry"] = {
        "infrastructure_retries": suite.execution.infrastructure_retries,
        "retry_on": execution.pop("retry_on", ["runner", "environment"]),
    }
    execution["order"] = "declared"
    limitations = (
        ["git_unavailable"]
        if runtime.git is None
        else ["runtime_dirty"]
        if runtime.git["dirty"]
        else []
    )
    if any(not c.set for c in candidate.credentials):
        limitations.append("credential_unset")
    if any(p.version is None for p in candidate.plugins.values()):
        limitations.append("plugin_unversioned")
    source_body = [c.model_dump(exclude={"path"}) for c in cases]
    grading = suite.grading.model_dump(mode="json")
    plan = PlanIdentity(
        suite=SuiteIdentity(
            name=suite.name,
            track=suite.track,
            path=os.path.relpath(suite_path, base_dir),
            digest=value_digest(suite.model_dump(mode="json")),
        ),
        source=LocalSourceIdentity(digest=value_digest(source_body), cases=cases),
        candidate=candidate,
        runtime=runtime,
        execution=ExecutionIdentity(digest=value_digest(execution), **execution),
        trace=suite.trace,
        grading=GradingIdentity(
            digest=value_digest(
                {"policy": grading, "cases": [c.graders_digest for c in cases]}
            ),
            **grading,
        ),
        trials=[
            TrialSpec(trial_id=f"{c.id}--candidate--{i}", case_id=c.id, repetition=i)
            for c in cases
            for i in range(1, suite.execution.repetitions + 1)
        ],
        limitations=limitations,
    )
    assert_secret_free(canonical_json(plan.model_dump(mode="json")), secrets)
    return plan
