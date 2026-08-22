"""The published workflow contract.

This is the document a miner reads before building anything. It states the fixed
parts of the arena — base model, adapter pool, tools, stages, limits, gates,
scoring weights and the champion-challenge rule — in one machine-readable place,
so nothing a candidate is judged on has to be inferred from prose.

Publishing the contract is also what makes the arena falsifiable: if the engine
ever scored something the contract does not describe, the difference would be
visible.
"""

from __future__ import annotations

from typing import Any

from capability_subnet.common import constants as C
from capability_subnet.workflows.industrial_maintenance_de_v1.final_schema import (
    FINAL_PAYLOAD_SCHEMA,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import (
    CRITICAL_AXES,
    STAGE_THRESHOLDS,
    STAGES,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.tools_schema import TOOL_SCHEMAS
from capability_subnet.workflows.shared_contract import (
    base_model_contract,
    hard_gates_contract,
    incentive_contract,
    qualified_scoring_contract,
    ranking_contract,
    recipe_contract,
    resolve_snapshot,
    runs_contract,
    source_pool_contract,
)

#: This workflow's own identifier, stated rather than read from the shipped
#: default. Recipes and reports are keyed by it, so it must not move when an
#: operator changes which workflow runs.
WORKFLOW_ID = "industrial_maintenance_de_v1"
WORKFLOW_TITLE = "Industrial Maintenance DE"

DESCRIPTION = (
    "A German industrial-maintenance agent reads a technical manual and a sensor "
    "log, identifies a fault, queries maintenance history, writes and executes "
    "diagnostic code, selects a replacement part, enforces a safety policy, and "
    "returns a strict JSON report. Later steps consume earlier outputs, so the "
    "task cannot be decomposed into independent benchmark questions."
)

CAPABILITIES = (
    "german_technical_language",
    "structured_log_extraction",
    "fault_reasoning",
    "text_to_sql",
    "python_code_generation",
    "tool_calling",
    "safety_policy_compliance",
    "structured_output",
)


def build_contract(snapshot=None, seed_root_commitment: str = "") -> dict[str, Any]:
    """Assemble the contract document.

    Args:
        snapshot: the frozen pool. Passing ``None`` loads the shipped one, which
            is what the CLI does; the engine passes its own so the contract it
            publishes always describes the pool it is actually evaluating against.
    """
    snapshot = resolve_snapshot(snapshot)

    return {
        "contract_version": 1,
        "workflow_id": WORKFLOW_ID,
        "title": WORKFLOW_TITLE,
        "description": DESCRIPTION,
        "capabilities": list(CAPABILITIES),
        "base_model": base_model_contract(snapshot),
        "source_pool": source_pool_contract(snapshot),
        "recipe": recipe_contract(snapshot),
        "agent_harness": {
            "tools": TOOL_SCHEMAS,
            "max_turns": C.MAX_AGENT_TURNS,
            "max_output_tokens": C.MAX_OUTPUT_TOKENS,
            "temperature": C.SANDBOX_TEMPERATURE,
            "top_p": C.SANDBOX_TOP_P,
            "enable_thinking": C.SANDBOX_ENABLE_THINKING,
            "note": (
                "The loop, the system prompt and the tool schemas are identical for "
                "every candidate. Sampling is greedy and seeded per instance, and the "
                "model's separate reasoning channel is disabled — one thinking block "
                "would consume the whole output budget before the first tool call."
            ),
        },
        "stages": {
            "order": list(STAGES),
            "thresholds": dict(STAGE_THRESHOLDS),
            "critical_axes": list(CRITICAL_AXES),
        },
        "final_payload_schema": FINAL_PAYLOAD_SCHEMA,
        "hard_gates": hard_gates_contract(),
        "scoring": qualified_scoring_contract(),
        "champion_challenge": ranking_contract(),
        "runs": runs_contract(seed_root_commitment),
        "incentive": incentive_contract(),
    }
