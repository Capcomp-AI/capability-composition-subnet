"""Workflow registry.

The protocol is written for many workflows even though V1 ships one. Registering
them by identifier rather than importing the V1 module directly is what keeps a
second workflow from becoming a rewrite: the engine, the sandbox and the scorer
all address a workflow through this table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from capability_subnet.common import constants as C


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
    )


_LOADERS: dict[str, Callable[[], WorkflowModule]] = {
    C.DEFAULT_WORKFLOW_ID: _load_industrial_maintenance_de_v1,
}

_CACHE: dict[str, WorkflowModule] = {}


def available_workflows() -> tuple[str, ...]:
    return tuple(sorted(_LOADERS))


def get_workflow(workflow_id: str = C.DEFAULT_WORKFLOW_ID) -> WorkflowModule:
    """Resolve a workflow by identifier.

    Raises:
        KeyError: naming every workflow this build knows about, because the usual
            cause is a recipe or config pointing at a workflow that was renamed.
    """
    if workflow_id in _CACHE:
        return _CACHE[workflow_id]
    loader = _LOADERS.get(workflow_id)
    if loader is None:
        raise KeyError(
            f"unknown workflow {workflow_id!r}; this build provides "
            f"{list(available_workflows())}"
        )
    _CACHE[workflow_id] = loader()
    return _CACHE[workflow_id]


__all__ = ["WorkflowModule", "available_workflows", "get_workflow"]
