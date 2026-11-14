"""Deterministic bounded pipelines and safe, value-based journal resume."""

from __future__ import annotations

import asyncio
from typing import Any

from lingcore.delegation.envelope import node_class
from lingcore.delegation.models import digest
from lingcore.delegation.supervisor import RunHandle, Supervisor
from lingcore.delegation.worker_run import owned_io
from lingcore.errors import ToolError
from lingcore.workflows.expressions import (
    BLOCKED,
    BlockedDependency,
    contains_blocked,
    dedupe,
    lookup,
    predicate,
    references,
    resolve,
    transform,
)
from lingcore.workflows.spec import KINDS, ValidatedSpec, input_values


class Interpreter:
    def __init__(
        self,
        supervisor: Supervisor,
        handle: RunHandle,
        validated: ValidatedSpec,
        snapshot: frozenset[str],
        *,
        resume: bool = False,
    ) -> None:
        self.supervisor, self.handle, self.validated, self.snapshot = (
            supervisor,
            handle,
            validated,
            snapshot,
        )
        self.resume = resume
        self.dropped = 0
        self.failed = 0
        self.blocked: list[dict[str, Any]] = []
        self.previous = {
            n.identity: n for n in supervisor.journal.nodes(handle.record.id)
        }

    def context(self, values: dict[str, Any]) -> dict[str, Any]:
        return values | {
            "budget": {
                "remaining_tokens": self.handle.budget.remaining_tokens,
                "agents_left": max(
                    0, self.validated.limits.max_agents - self.handle.budget.agents
                ),
            }
        }

    async def prepare(self) -> None:
        if not any(e.workspace == "frozen" for e in self.validated.roles.values()):
            return
        async with self.handle.frozen_lock:
            await self._prepare_input()

    async def _prepare_input(self) -> None:
        if self.handle.record.input_revision in self.handle.verified_inputs:
            return
        record = self.handle.record
        if record.input_revision is None:
            if self.resume:
                raise ToolError("frozen input is missing; start a new workflow run")
            record.source_revision = await owned_io(
                self.supervisor.workspace.source_revision
            )
            _, revision, skipped = await owned_io(
                self.supervisor.workspace.capture, record.id, "input", frozen=True
            )
            from lingcore.delegation.models import now

            record.input_revision, record.input_captured_at, record.skipped = (
                revision,
                now(),
                skipped,
            )
            self.supervisor.journal.save(record)
        else:
            path = self.supervisor.workspace.path(record.id, "input")
            if (
                await owned_io(self.supervisor.workspace.revision, path, record.skipped)
                != record.input_revision
            ):
                raise ToolError("frozen input was altered; start a new workflow run")
        assert record.input_revision is not None
        self.handle.verified_inputs.add(record.input_revision)

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.prepare()
        args = input_values(self.validated.spec.inputs, args)
        values = {"inputs": args, "request": self.handle.record.request}
        await self.sequence(self.validated.spec.steps, values, "", {})
        try:
            result = (
                resolve(self.validated.spec.result, self.context(values))
                if self.validated.spec.result is not None
                else values.get("prev")
            )
            if contains_blocked(result):
                result = None
        except BlockedDependency:
            result = None
        if self.blocked:
            self.handle.record.status = "blocked"
        current_revision = (
            await owned_io(self.supervisor.workspace.source_revision)
            if self.handle.record.input_revision
            else None
        )
        return {
            "result": result,
            "dropped": self.dropped,
            "failed": self.failed,
            "blocked": self.blocked,
            "input_revision": self.handle.record.input_revision,
            "input_captured_at": self.handle.record.input_captured_at,
            "parent_workspace_changed": current_revision
            != self.handle.record.source_revision
            if current_revision and self.handle.record.source_revision
            else None,
        }

    async def sequence(
        self,
        steps: list[dict[str, Any]],
        values: dict[str, Any],
        path: str,
        control: dict[str, Any],
    ) -> Any:
        for index, step in enumerate(steps):
            identity = path + "/" + str(step.get("id", index))
            result = await self.step(step, values, identity, control)
            values["prev"] = result
            if step.get("id"):
                values[step["id"]] = result
        return values.get("prev")

    async def step(
        self,
        step: dict[str, Any],
        values: dict[str, Any],
        path: str,
        control: dict[str, Any],
    ) -> Any:
        try:
            return await self._step(step, values, path, control)
        except BlockedDependency as exc:
            self.blocked.append(
                {"node": path, "reason": "blocked dependency: " + str(exc)}
            )
            self.dropped += 1
            return BLOCKED

    async def _step(
        self,
        step: dict[str, Any],
        values: dict[str, Any],
        path: str,
        control: dict[str, Any],
    ) -> Any:
        if self.handle.cancelled.is_set():
            raise asyncio.CancelledError
        context = self.context(values)
        if "when" in step and not predicate(step["when"], context):
            return None
        if step.get("phase"):
            self.supervisor.progress(self.handle, path, "running", step["phase"])
        kind = next(iter(set(step) & KINDS))
        value = step[kind]
        if kind == "agent":

            async def occurrence(index: int) -> Any:
                return await self.agent(step, context, f"{path}#{index}", control)

            results = await gather_owned(
                [occurrence(i) for i in range(value.get("repeat", 1))]
            )
            return results if value.get("repeat", 1) > 1 else results[0]
        if kind == "map":
            items = resolve(value["over"], context) or []
            if not isinstance(items, list):
                raise ToolError("map input must be a list")
            cap = value.get("max_items", 100)
            self.dropped += max(0, len(items) - cap)

            async def pipeline(index: int, item: Any) -> Any:
                local = values | {value["as"]: item, "index": index, "prev": item}
                controls = control | {"map_item": item, "map_index": index}
                for stage_index, stage in enumerate(value["stages"]):
                    result = await self.step(
                        stage,
                        local,
                        f"{path}/item:{index}/{stage.get('id', stage_index)}",
                        controls,
                    )
                    if result is BLOCKED:
                        return BLOCKED
                    if result is None:
                        self.dropped += 1
                        return None
                    local["prev"] = result
                    if stage.get("id"):
                        local[stage["id"]] = result
                return local["prev"]

            # Bounded task fanout independent of worker slots. Each item reaches
            # the next stage immediately; gather only fixes final output order.
            results = []
            for start in range(0, min(len(items), cap), 4096):
                results.extend(
                    await gather_owned(
                        [
                            pipeline(i, items[i])
                            for i in range(start, min(start + 4096, len(items), cap))
                        ]
                    )
                )
            return (
                BLOCKED
                if contains_blocked(results)
                else [v for v in results if v is not None]
            )
        if kind == "parallel":
            results = await gather_owned(
                [
                    self.step(s, dict(values), path + "/" + s["id"], control)
                    for s in value
                ]
            )
            return {s["id"]: r for s, r in zip(value, results)}
        if kind == "loop":
            acc: list[Any] = []
            dry = 0
            for index in range(value["max_rounds"]):
                local = values | {"acc": list(acc), "index": index, "prev": list(acc)}
                until = value.get("until", {})
                threshold = until.get("budget_below")
                remaining = self.handle.budget.remaining_tokens
                if (
                    threshold is not None
                    and remaining is not None
                    and remaining < threshold
                ):
                    break
                await self.sequence(
                    value["steps"],
                    local,
                    f"{path}/round:{index}",
                    control | {"acc": list(acc), "round": index},
                )
                additions = resolve(value["collect"]["from"], self.context(local)) or []
                if not isinstance(additions, list):
                    additions = [additions]
                combined = dedupe(acc + additions, value["collect"].get("dedupe_by"))
                self.dropped += max(0, len(combined) - 4096)
                combined = combined[:4096]
                dry = dry + 1 if len(combined) == len(acc) else 0
                acc = combined
                if ("dry_rounds" in until and dry >= until["dry_rounds"]) or (
                    "count_at_least" in until
                    and predicate(
                        {"count_at_least": until["count_at_least"]},
                        self.context(local | {"acc": acc}),
                    )
                ):
                    break
            return acc
        if kind == "keep":
            return (
                resolve(value["value"], context)
                if predicate(value["if"], context)
                else None
            )
        if kind == "transform":
            return transform(value, context)
        if kind == "log":
            self.supervisor.progress(
                self.handle, str(resolve(value, context))[:2000], "log"
            )
            return values.get("prev")
        if kind == "workflow":
            child = self.validated.catalog[value["name"]]
            child_valid = ValidatedSpec(
                child,
                self.validated.limits,
                self.validated.envelope,
                self.validated.catalog,
                child=True,
                models=self.validated.models,
            )
            child_args = input_values(
                child.inputs, resolve(value.get("args", {}), context)
            )
            local = {"inputs": child_args, "request": self.handle.record.request}
            from lingcore.workflows.children import child_handle

            nested = child_handle(
                self.supervisor,
                self.handle,
                child_valid.limits,
                path + "/child",
                resume=self.resume,
            )
            interpreter = Interpreter(
                self.supervisor,
                nested,
                child_valid,
                self.snapshot,
                resume=self.resume,
            )
            await interpreter.prepare()
            await interpreter.sequence(
                child.steps, local, path + "/child", control | {"args": child_args}
            )
            self.dropped += interpreter.dropped
            self.failed += interpreter.failed
            self.blocked.extend(interpreter.blocked)
            return (
                resolve(child.result, interpreter.context(local))
                if child.result is not None
                else local.get("prev")
            )
        raise ToolError("unknown workflow operator")

    async def agent(
        self,
        step: dict[str, Any],
        context: dict[str, Any],
        identity: str,
        control: dict[str, Any],
    ) -> Any:
        spec = step["agent"]
        role = spec.get("role", "general")
        envelope = self.supervisor.envelope(
            role,
            self.snapshot,
            tools=spec.get("tools"),
            workspace=spec.get("workspace"),
        )
        classification = node_class(envelope)
        schema = spec.get("output")
        if isinstance(schema, str):
            schema = self.validated.spec.schemas[schema]
        prompt = str(resolve(spec["prompt"], context))
        dependencies = {ref: lookup(ref, context) for ref in references(step)}
        definition = self.supervisor.role(role)
        if spec.get("model"):
            definition = definition.model_copy(update={"model": spec["model"]})
        profile = self.supervisor.profile
        model = (
            profile.delegation.models.get(definition.model)
            if definition.model
            else profile.llm.model_dump()
        )
        from lingcore.workflows.cache import worker_context

        layers, context_reusable = worker_context(
            self.supervisor, self.handle, definition, envelope
        )
        key = digest(
            {
                "version": 1,
                "identity": identity,
                "definition": step,
                "prompt": prompt,
                "dependencies": dependencies,
                "control": control,
                "role": definition.model_dump(),
                "envelope": envelope.identity(),
                "context": layers,
                "schema": schema,
                "model": model,
                "input_revision": self.handle.record.input_revision
                if classification == "frozen"
                else "",
            }
        )
        old = self.previous.get(identity)
        if self.resume and old:
            try:
                intact = (
                    await owned_io(
                        self.supervisor.workspace.verify_artifacts, old.artifacts_digest
                    )
                    and old.report is not None
                    and old.report_digest == digest(old.report.model_dump(mode="json"))
                )
            except (OSError, ToolError, ValueError):
                intact = False
            if old.classification == "effecting" and (
                old.key != key or old.status != "completed" or not intact
            ):
                self.blocked.append(
                    {
                        "node": identity,
                        "reason": "effecting node cannot execute again; start a new run",
                        "worker_id": old.worker_id,
                        "session_id": old.session_id,
                    }
                )
                self.dropped += 1
                return BLOCKED
            reusable = classification in {"pure", "frozen"} and context_reusable
            if (
                old.key == key
                and old.report
                and intact
                and (
                    old.classification == "effecting"
                    or (reusable and old.status == "completed")
                )
            ):
                self.supervisor.progress(
                    self.handle,
                    identity,
                    "recorded" if old.classification == "effecting" else "cached",
                )
                return old.report.result if old.report.status == "completed" else None
        attempts = 1 + spec.get("retries", 2 if spec.get("retry_on") else 0)
        for attempt in range(attempts):
            report = await self.supervisor.worker(
                self.handle,
                prompt,
                role,
                envelope,
                schema=schema,
                label=str(resolve(spec.get("label", role), context)),
                identity=identity,
                key=key,
                model=spec.get("model"),
            )
            if report.status == "completed":
                return report.result
            if report.status == "blocked":
                self.blocked.append(
                    {
                        "node": identity,
                        "reason": report.question or "worker blocked",
                        "worker_id": report.worker_id,
                        "session_id": report.session_id,
                    }
                )
                return BLOCKED
            if (
                report.error not in {"transient", "interrupted"}
                or attempt + 1 >= attempts
            ):
                break
        self.failed += 1
        return None


async def gather_owned(coroutines: list[Any]) -> list[Any]:
    tasks = [asyncio.create_task(c) for c in coroutines]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
