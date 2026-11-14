"""Explicitly consented Python workflows, with the same runtime admission caps."""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable

from lingcore.delegation.models import RunRecord, WorkspaceMode
from lingcore.delegation.supervisor import RunHandle, Supervisor
from lingcore.errors import ConfigError, ToolError
from lingcore.workflows.interpreter import gather_owned


@dataclass(frozen=True)
class PythonWorkflow:
    name: str
    description: str
    phases: tuple[str, ...]
    fn: Callable[..., Any]


def workflow(
    *, name: str, description: str = "", phases: list[str] | None = None
) -> Callable[..., Any]:
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,79}", name):
        raise ConfigError("invalid Python workflow name")
    if (
        len(description) > 2000
        or len(phases or []) > 64
        or any(not isinstance(p, str) or len(p) > 200 for p in phases or [])
    ):
        raise ConfigError("Python workflow metadata exceeds catalog bounds")

    def decorate(fn: Callable[..., Any]) -> PythonWorkflow:
        if not inspect.iscoroutinefunction(fn):
            raise ConfigError("Python workflows must be async functions")
        return PythonWorkflow(name, description, tuple(phases or []), fn)

    return decorate


class WorkflowAPI:
    def __init__(
        self,
        supervisor: Supervisor,
        handle: RunHandle,
        snapshot: frozenset[str],
        args: dict[str, Any],
    ) -> None:
        self.supervisor, self.handle, self.snapshot, self.args = (
            supervisor,
            handle,
            snapshot,
            args,
        )
        self.budget = handle.budget
        self._phase = ""
        self._children = 0

    async def agent(
        self,
        prompt: str,
        *,
        role: str = "general",
        output: dict[str, Any] | None = None,
        workspace: WorkspaceMode | None = None,
        tools: list[str] | None = None,
        label: str | None = None,
        model: str | None = None,
    ) -> Any:
        envelope = self.supervisor.envelope(
            role,
            self.snapshot,
            workspace=workspace,
            tools=tools,
        )
        report = await self.supervisor.worker(
            self.handle, prompt, role, envelope, schema=output, label=label, model=model
        )
        return report.result if report.status == "completed" else None

    async def parallel(self, *calls: Any) -> list[Any]:
        # Worker failures are already represented by agent() as None. Errors
        # in orchestration/configuration must remain visible to its caller.
        return await gather_owned(list(calls))

    async def pipeline(
        self, items: list[Any], *stages: Callable[..., Any]
    ) -> list[Any]:
        if len(items) > 4096:
            raise ToolError("Python workflow item cap exceeded")

        async def run(item: Any, index: int) -> Any:
            for stage in stages:
                item = await stage(item, index)
                if item is None:
                    break
            return item

        return await self.parallel(*(run(item, i) for i, item in enumerate(items)))

    def phase(self, title: str) -> None:
        self._phase = title
        self.supervisor.progress(self.handle, title, "phase", title)

    def log(self, text: str) -> None:
        self.supervisor.progress(self.handle, text[:2000], "log", self._phase)

    async def workflow(self, name: str, args: dict[str, Any] | None = None) -> Any:
        from lingcore.workflows.catalog import load_catalog
        from lingcore.workflows.children import child_handle
        from lingcore.workflows.interpreter import Interpreter
        from lingcore.workflows.spec import ValidatedSpec, input_values

        catalog = load_catalog(self.supervisor.profile)
        if name not in catalog:
            raise ToolError("unknown child workflow")
        validated = ValidatedSpec(
            catalog[name],
            self.supervisor.profile.delegation.limits,
            lambda role, **kw: self.supervisor.envelope(role, self.snapshot, **kw),
            catalog,
            child=True,
            models=frozenset(self.supervisor.profile.delegation.models),
        )
        self._children += 1
        prefix = f"/python-child:{self._children}/"
        nested = child_handle(
            self.supervisor, self.handle, validated.limits, prefix, resume=False
        )
        interpreter = Interpreter(self.supervisor, nested, validated, self.snapshot)
        await interpreter.prepare()
        from lingcore.workflows.expressions import (
            BlockedDependency,
            contains_blocked,
            resolve,
        )

        local = {
            "inputs": input_values(validated.spec.inputs, args or {}),
            "request": self.handle.record.request,
        }
        await interpreter.sequence(validated.spec.steps, local, prefix, {})
        try:
            result = (
                resolve(validated.spec.result, interpreter.context(local))
                if validated.spec.result is not None
                else local.get("prev")
            )
        except BlockedDependency:
            result = None
        progress = self.handle.record.progress.setdefault("children", [])
        progress.append(
            {
                "name": name,
                "dropped": interpreter.dropped,
                "failed": interpreter.failed,
                "blocked": interpreter.blocked,
            }
        )
        if interpreter.blocked or contains_blocked(result):
            self.handle.record.status = "blocked"
            return None
        return result


def plugin_workflows(profile: Any) -> dict[str, PythonWorkflow]:
    from lingcore.delegation.roles import read_asset
    from lingcore.plugins.discovery import discover_plugins
    from lingcore.plugins.manifest import component_path

    catalog = {}
    for name, plugin in discover_plugins(getattr(profile, "_source_dir", None)).items():
        if name not in profile.plugins or not plugin.manifest.workflows:
            continue
        directory = component_path(plugin.root, plugin.manifest.workflows)
        for path in sorted(directory.glob("*.py")):
            component_path(plugin.root, str(path.relative_to(plugin.root)))
            source = read_asset(plugin.root, str(path.relative_to(plugin.root)), 262144)
            revision = sha256((str(path) + "\0" + source).encode()).hexdigest()
            module_name = f"lingcore_workflow_{revision}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ConfigError("cannot load Python workflow")
            module = sys.modules.get(module_name)
            if module is None:
                module = importlib.util.module_from_spec(spec)
                # Dataclasses and postponed annotations resolve their owning
                # module during execution. Cache by source revision, not name.
                sys.modules[module_name] = module
                try:
                    exec(compile(source, str(path), "exec"), module.__dict__)
                except BaseException:
                    sys.modules.pop(module_name, None)
                    raise
            for value in vars(module).values():
                if isinstance(value, PythonWorkflow):
                    if value.name in catalog:
                        raise ConfigError("duplicate Python workflow")
                    catalog[value.name] = value
    return catalog


async def run_python(
    supervisor: Supervisor,
    snapshot: frozenset[str],
    definition: PythonWorkflow,
    args: dict[str, Any],
    *,
    background: bool = False,
    resumed: RunRecord | None = None,
) -> Any:
    # Validate typed operator arguments before admitting or confirming a run.
    from typing import get_type_hints

    hints = get_type_hints(definition.fn)
    model = hints.get("args")
    parsed = (
        model.model_validate(args, strict=True)
        if model and hasattr(model, "model_validate")
        else args
    )
    handle = (
        supervisor.new_run(kind="workflow", name=definition.name, background=background)
        if resumed is None
        else _resume_handle(supervisor, resumed)
    )
    handle.record.args = args
    envelopes = {r: supervisor.envelope(r, snapshot) for r in supervisor.roles}
    await supervisor.confirm_run(
        handle,
        envelopes,
        supervisor.profile.delegation.limits.max_agents,
        force=True,
        preserve_status=resumed is not None,
    )
    if resumed is not None:
        supervisor.journal.revive(handle.record)
    api = WorkflowAPI(supervisor, handle, snapshot, args)
    return await supervisor.execute(handle, lambda: definition.fn(api, parsed))


async def resume_python(
    supervisor: Supervisor, record: RunRecord, snapshot: frozenset[str]
) -> Any:
    if any(
        n.classification == "effecting" for n in supervisor.journal.nodes(record.id)
    ):
        raise ToolError("Python workflow with a started effecting worker cannot resume")
    catalog = plugin_workflows(supervisor.profile)
    if record.name not in catalog:
        raise ToolError("Python workflow is no longer installed")
    return await run_python(
        supervisor, snapshot, catalog[record.name], record.args, resumed=record
    )


def _resume_handle(supervisor: Supervisor, record: RunRecord) -> RunHandle:
    from lingcore.workflows.runtime import resume_handle

    return resume_handle(supervisor, record, supervisor.profile.delegation.limits)
