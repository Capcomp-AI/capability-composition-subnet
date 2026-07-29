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

from capability_subnet import __spec_version__
from capability_subnet.common import constants as C
from capability_subnet.common.schemas import recipe_json_schema
from capability_subnet.merge_engine.methods import PIPELINES
from capability_subnet.workflows.industrial_maintenance_de_v1.final_schema import (
    FINAL_PAYLOAD_SCHEMA,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import (
    CRITICAL_AXES,
    STAGE_THRESHOLDS,
    STAGES,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.tools_schema import TOOL_SCHEMAS

WORKFLOW_ID = C.DEFAULT_WORKFLOW_ID
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


def build_contract(snapshot=None) -> dict[str, Any]:
    """Assemble the contract document.

    Args:
        snapshot: the frozen pool. Passing ``None`` loads the shipped one, which
            is what the CLI does; the engine passes its own so the contract it
            publishes always describes the pool it is actually evaluating against.
    """
    if snapshot is None:
        from capability_subnet.registry.snapshot import load_snapshot

        snapshot = load_snapshot()

    return {
        "contract_version": 1,
        "spec_version": __spec_version__,
        "workflow_id": WORKFLOW_ID,
        "title": WORKFLOW_TITLE,
        "description": DESCRIPTION,
        "capabilities": list(CAPABILITIES),
        "base_model": {
            "repo": snapshot.manifest.model_repo,
            "revision": snapshot.manifest.revision,
            "revision_pinned": snapshot.manifest.is_pinned,
            "dtype": snapshot.manifest.dtype,
            "num_hidden_layers": snapshot.manifest.num_hidden_layers,
        },
        "source_pool": {
            "source_snapshot_sha256": snapshot.sha256,
            "canonical_rank": snapshot.registry.canonical_rank,
            "canonical_lora_alpha": snapshot.registry.canonical_lora_alpha,
            "target_modules": list(snapshot.registry.canonical_target_modules),
            "adapters": list(snapshot.adapter_ids),
            "distractors": list(snapshot.registry.distractors()),
        },
        "recipe": {
            "schema_version": C.RECIPE_SCHEMA_VERSION,
            "json_schema": recipe_json_schema(),
            "layer_groups": {
                name: {"first_layer": low, "last_layer": high}
                for name, (low, high) in snapshot.manifest.layer_groups.items()
            },
            "merge_methods": {
                name: {
                    "sparsify": pipeline.sparsify,
                    "sign_election": pipeline.sign_election,
                    "aggregate": pipeline.aggregate,
                    "description": pipeline.description,
                }
                for name, pipeline in sorted(PIPELINES.items())
            },
            "bounds": {
                "adapter_weight": [C.ADAPTER_WEIGHT_MIN, C.ADAPTER_WEIGHT_MAX],
                "density": [C.DENSITY_MIN, C.DENSITY_MAX],
                "output_rank": list(C.ALLOWED_OUTPUT_RANKS),
                "svd_clamp_quantile": [C.SVD_CLAMP_QUANTILE_MIN, C.SVD_CLAMP_QUANTILE_MAX],
                "random_seed": [C.RANDOM_SEED_MIN, C.RANDOM_SEED_MAX],
                "selected_adapters": [C.MIN_SELECTED_ADAPTERS, C.MAX_SELECTED_ADAPTERS],
            },
        },
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
        "hard_gates": {
            "artifact_size_bytes": C.MAX_ARTIFACT_BYTES,
            "peak_vram_gb": C.MAX_PEAK_VRAM_GB,
            "p95_workflow_seconds": C.MAX_P95_WORKFLOW_SECONDS,
            "max_turns": C.MAX_AGENT_TURNS,
            "max_output_tokens": C.MAX_OUTPUT_TOKENS,
            "critical_unsafe_actions": C.MAX_CRITICAL_UNSAFE_ACTIONS,
            "base_retention_floor": C.BASE_RETENTION_FLOOR,
        },
        "scoring": {
            "weights": dict(C.QUALIFIED_SCORE_WEIGHTS),
            "formula": (
                "Q = 0.60·end_to_end + 0.15·stage_balance + 0.10·ood + "
                "0.05·retention + 0.05·latency + 0.05·artifact_efficiency"
            ),
            "stage_balance": (
                "Geometric mean of the per-stage means. Rewards packages that are "
                "competent everywhere over packages that are excellent at one stage "
                "and broken at another."
            ),
        },
        "champion_challenge": {
            "axis_margin": C.DEFAULT_AXIS_MARGIN,
            "axis_tolerance": C.DEFAULT_AXIS_TOLERANCE,
            "min_dominant_axes": C.DEFAULT_MIN_DOMINANT_AXES,
            "min_axis_samples": C.DEFAULT_MIN_AXIS_SAMPLES,
            "end_to_end_margin": C.DEFAULT_END_TO_END_MARGIN,
            "bootstrap_resamples": C.BOOTSTRAP_RESAMPLES,
            "bootstrap_confidence": C.BOOTSTRAP_CONFIDENCE,
            "rule": (
                "A challenger takes the throne only if it dominates at least the "
                "required number of capability axes, is not worse on any other axis, "
                "raises end-to-end completion over the strongest reference by the "
                "absolute margin, and its paired bootstrap lower confidence bound "
                "against that reference is above zero. One shot per hotkey: a "
                "decisive loss terminates the challenger permanently."
            ),
        },
        "windows": {
            "window_blocks": C.DEFAULT_WINDOW_BLOCKS,
            "hidden_instances": C.DEFAULT_HIDDEN_INSTANCES,
            "ood_instances": C.DEFAULT_OOD_INSTANCES,
            "public_pack_instances": C.PUBLIC_PACK_INSTANCES,
        },
        "incentive": {
            "default_mode": C.MODE_WINNER_TAKE_ALL,
            "graded_shares": list(C.GRADED_TOP3_SHARES),
            "burn_uid": C.BURN_UID,
            "note": (
                "The reigning champion holds the full workflow weight. If no "
                "candidate and no incumbent qualify, the workflow share is burned "
                "rather than redistributed."
            ),
        },
    }
