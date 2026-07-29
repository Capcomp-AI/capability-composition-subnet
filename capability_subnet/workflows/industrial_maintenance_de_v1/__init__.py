"""Industrial Maintenance DE — the V1 workflow.

A German industrial-maintenance agent works one fault from a controller log all
the way to a signed-off replacement decision. Eight capabilities are exercised in
one dependent chain, and every stage is judged against truth computed before the
candidate ever saw the instance.
"""

from capability_subnet.workflows.industrial_maintenance_de_v1.contract import (
    WORKFLOW_ID,
    WORKFLOW_TITLE,
    build_contract,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.generator import (
    generate_batch,
    generate_instance,
    sample_seeds,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import (
    CRITICAL_AXES,
    STAGE_THRESHOLDS,
    STAGES,
    WorkflowInstance,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.scoring import (
    InstanceScore,
    score_instance,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.tools_schema import (
    TOOL_NAMES,
    TOOL_SCHEMAS,
)

__all__ = [
    "CRITICAL_AXES",
    "STAGES",
    "STAGE_THRESHOLDS",
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "WORKFLOW_ID",
    "WORKFLOW_TITLE",
    "InstanceScore",
    "WorkflowInstance",
    "build_contract",
    "generate_batch",
    "generate_instance",
    "sample_seeds",
    "score_instance",
]


def run_instance(instance, client, *, config=None):
    """Drive one instance through the twelve-turn agent loop.

    Re-exported so the workflow owns how a package is asked, not just what it
    is asked. The implementation is unchanged; only the ownership moved.
    """
    from capability_subnet.sandbox.orchestrator import run_instance as _run

    return _run(instance, client, config=config)
