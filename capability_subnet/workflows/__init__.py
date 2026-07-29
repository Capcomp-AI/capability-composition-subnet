"""Workflow registry.

The protocol is written for many workflows even though V1 ships one. Registering
them by identifier rather than importing the V1 module directly is what keeps a
second workflow from becoming a rewrite: the engine, the sandbox and the scorer
all address a workflow through this table.

Workflows are discovered two ways. The ones shipped in this package are listed
below. Anything else is found through the ``capability_subnet.workflows`` entry
point group, so a workflow can live in its own distribution — installed
alongside this one — without editing this file or forking the repository.

That matters for one case in particular. A workflow built on a customer's real
business process cannot have its generator published, because the generator
*is* the process. Entry points let such a workflow ship as a private package
while the protocol, the merge engine and the evaluation loop stay public and
reviewable.

The cost is stated rather than hidden. A workflow nobody outside the operator can
install is a workflow nobody outside the operator can replay, and replay is what
makes a published score checkable. So every workflow declares whether it is
publicly verifiable, the engine records that in every report, and an operator
running a closed workflow is telling the network plainly that its scores rest on
the operator's word.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from capability_subnet.common import constants as C

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkflowModule:
    """Everything the engine needs from a workflow implementation."""

    workflow_id: str
    title: str
    generate_instance: Callable[..., Any]
    score_instance: Callable[..., Any]
    build_contract: Callable[..., dict[str, Any]]
    stages: tuple[str, ...]
    critical_axes: tuple[str, ...]
    stage_thresholds: dict[str, float]
    tool_schemas: list[dict[str, Any]]
    #: Runs one instance against a served package and returns its sample row.
    #:
    #: Owned by the workflow rather than the engine because workflows differ in
    #: how a package is *asked*, not only in what it is asked. The V1 workflow
    #: drives a twelve-turn agent loop over seven tool services; a single-turn
    #: benchmark asks one question and reads one answer. Hardcoding the first
    #: shape into the evaluator meant the second could not be plugged in at all,
    #: which made the entry-point mechanism a promise the code could not keep.
    #:
    #: Signature: ``(instance, client, *, config) -> SandboxOutcome``.
    run_instance: Callable[..., Any]
    #: Whether anyone can obtain this workflow and re-score a closed window from
    #: its published seeds and traces. False for a workflow whose generator
    #: cannot be published — which is a legitimate choice and a materially
    #: weaker guarantee, so it is recorded rather than assumed.
    publicly_verifiable: bool = True


def _load_industrial_maintenance_de_v1() -> WorkflowModule:
    from capability_subnet.workflows import industrial_maintenance_de_v1 as module

    return WorkflowModule(
        workflow_id=module.WORKFLOW_ID,
        title=module.WORKFLOW_TITLE,
        generate_instance=module.generate_instance,
        score_instance=module.score_instance,
        build_contract=module.build_contract,
        stages=tuple(module.STAGES),
        critical_axes=tuple(module.CRITICAL_AXES),
        stage_thresholds=dict(module.STAGE_THRESHOLDS),
        tool_schemas=list(module.TOOL_SCHEMAS),
        run_instance=module.run_instance,
        publicly_verifiable=True,
    )


#: Entry point group third-party workflow distributions register under.
ENTRY_POINT_GROUP = "capability_subnet.workflows"


def _load_lora_merger_logic_v1() -> WorkflowModule:
    from capability_subnet.workflows import lora_merger_logic_v1 as module

    return WorkflowModule(
        workflow_id=module.WORKFLOW_ID,
        title=module.WORKFLOW_TITLE,
        generate_instance=module.generate_instance,
        score_instance=module.score_instance,
        build_contract=module.build_contract,
        stages=tuple(module.STAGES),
        critical_axes=tuple(module.CRITICAL_AXES),
        stage_thresholds=dict(module.STAGE_THRESHOLDS),
        tool_schemas=list(module.TOOL_SCHEMAS),
        run_instance=module.run_instance,
        publicly_verifiable=True,
    )


_BUILTIN_LOADERS: dict[str, Callable[[], WorkflowModule]] = {
    C.DEFAULT_WORKFLOW_ID: _load_industrial_maintenance_de_v1,
    "lora_merger_logic_v1": _load_lora_merger_logic_v1,
}

_CACHE: dict[str, WorkflowModule] = {}


def _entry_point_loaders() -> dict[str, Callable[[], WorkflowModule]]:
    """Workflow loaders published by other installed distributions.

    A built-in name is never overridden. Letting an installed package silently
    replace ``industrial_maintenance_de_v1`` would mean two engines agreeing on
    a workflow id while scoring different things, and the mismatch would surface
    as an unexplained consensus divergence rather than as an install problem.
    """
    from importlib.metadata import entry_points

    loaders: dict[str, Callable[[], WorkflowModule]] = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        if entry.name in _BUILTIN_LOADERS:
            log.warning(
                "ignoring workflow %r from %s: that identifier is built in",
                entry.name,
                getattr(entry, "dist", None),
            )
            continue
        loaders[entry.name] = entry.load
    return loaders


def _loaders() -> dict[str, Callable[[], WorkflowModule]]:
    return {**_entry_point_loaders(), **_BUILTIN_LOADERS}


def available_workflows() -> tuple[str, ...]:
    return tuple(sorted(_loaders()))


def get_workflow(workflow_id: str = C.DEFAULT_WORKFLOW_ID) -> WorkflowModule:
    """Resolve a workflow by identifier.

    Raises:
        KeyError: naming every workflow this build knows about, because the usual
            cause is a recipe or config pointing at a workflow that was renamed.
    """
    if workflow_id in _CACHE:
        return _CACHE[workflow_id]

    loader = _loaders().get(workflow_id)
    if loader is None:
        raise KeyError(
            f"unknown workflow {workflow_id!r}; this build provides {list(available_workflows())}. "
            f"A workflow from another distribution must register under the "
            f"{ENTRY_POINT_GROUP!r} entry point group."
        )

    module = loader()
    if module.workflow_id != workflow_id:
        raise KeyError(
            f"the loader registered as {workflow_id!r} produced a workflow calling itself "
            f"{module.workflow_id!r}. Recipes and reports are keyed by this identifier, so "
            "the two must agree."
        )
    if not module.publicly_verifiable:
        log.warning(
            "workflow %r is not publicly verifiable: its generator is not distributed, so "
            "nobody outside this operator can replay a closed window. Scores on this "
            "workflow rest on the operator's word.",
            workflow_id,
        )

    _CACHE[workflow_id] = module
    return module


__all__ = ["ENTRY_POINT_GROUP", "WorkflowModule", "available_workflows", "get_workflow"]
