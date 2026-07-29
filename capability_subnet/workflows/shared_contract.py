"""The parts of a published contract that do not depend on the workflow.

The base model, the frozen pool, the recipe schema and its bounds are facts about
the *protocol*, not about whichever workflow is judging. A miner needs them
whatever arena is running, and they were previously written out inside one
workflow's contract — so a second workflow published a contract with no base
model and no pool in it, which is not a contract a miner can build against.

Kept here so the two cannot drift. A third workflow gets them by calling this.
"""

from __future__ import annotations

from typing import Any

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import recipe_json_schema
from capability_subnet.merge_engine.methods import PIPELINES


def resolve_snapshot(snapshot=None):
    """The pool to describe, loading the shipped one when none was given.

    The engine passes its own so the contract it publishes always describes the
    pool it is actually evaluating against; the CLI passes nothing.
    """
    if snapshot is None:
        from capability_subnet.registry.snapshot import load_snapshot

        snapshot = load_snapshot()
    return snapshot


def base_model_contract(snapshot) -> dict[str, Any]:
    return {
        "repo": snapshot.manifest.model_repo,
        "revision": snapshot.manifest.revision,
        "revision_pinned": snapshot.manifest.is_pinned,
        "dtype": snapshot.manifest.dtype,
        "num_hidden_layers": snapshot.manifest.num_hidden_layers,
    }


def source_pool_contract(snapshot) -> dict[str, Any]:
    return {
        "source_snapshot_sha256": snapshot.sha256,
        "canonical_rank": snapshot.registry.canonical_rank,
        "canonical_lora_alpha": snapshot.registry.canonical_lora_alpha,
        "target_modules": list(snapshot.registry.canonical_target_modules),
        "adapters": list(snapshot.adapter_ids),
        "distractors": list(snapshot.registry.distractors()),
    }


def hard_gates_contract() -> dict[str, Any]:
    return {
        "artifact_size_bytes": C.MAX_ARTIFACT_BYTES,
        "peak_vram_gb": C.MAX_PEAK_VRAM_GB,
        "p95_workflow_seconds": C.MAX_P95_WORKFLOW_SECONDS,
        "max_turns": C.MAX_AGENT_TURNS,
        "max_output_tokens": C.MAX_OUTPUT_TOKENS,
        "critical_unsafe_actions": C.MAX_CRITICAL_UNSAFE_ACTIONS,
        "base_retention_floor": C.BASE_RETENTION_FLOOR,
    }


def qualified_scoring_contract() -> dict[str, Any]:
    return {
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
    }


def ranking_contract() -> dict[str, Any]:
    """How the board is ordered, in both contracts the engine supports.

    Stated as two modes rather than one rule, because `require_beat_reference`
    decides which applies and it ships **off**. A contract that described only the
    strict rule would describe a configuration the operator is not running.
    """
    return {
        "axis_margin": C.DEFAULT_AXIS_MARGIN,
        "axis_tolerance": C.DEFAULT_AXIS_TOLERANCE,
        "min_dominant_axes": C.DEFAULT_MIN_DOMINANT_AXES,
        "min_axis_samples": C.DEFAULT_MIN_AXIS_SAMPLES,
        "end_to_end_margin": C.DEFAULT_END_TO_END_MARGIN,
        "bootstrap_resamples": C.BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": C.BOOTSTRAP_CONFIDENCE,
        "default_rule": "highest_score_wins",
        "rule": (
            "By default the highest score on the board is paid, whether or not it "
            "cleared the strongest permanent reference. Scores closer together than "
            "the window can resolve are ranked as tied and ties resolve to the "
            "earliest commitment, so a copy of the leader cannot take its slot on "
            "sampling noise — it has to be measurably better. References are "
            "measured and published every window either way."
        ),
        "strict_rule": (
            "With require_beat_reference enabled, a challenger additionally has to "
            "dominate at least the required number of capability axes, be no worse "
            "on any other axis, raise end-to-end completion over the strongest "
            "reference by the absolute margin, and show a paired bootstrap lower "
            "confidence bound above zero against that reference."
        ),
        "always_enforced": (
            "The base-retention floor applies in both modes. A package that "
            "destroyed the base model's general ability is not deployable whatever "
            "it scored."
        ),
    }


def windows_contract() -> dict[str, Any]:
    return {
        "window_blocks": C.DEFAULT_WINDOW_BLOCKS,
        "hidden_instances": C.DEFAULT_HIDDEN_INSTANCES,
        "ood_instances": C.DEFAULT_OOD_INSTANCES,
        "public_pack_instances": C.PUBLIC_PACK_INSTANCES,
    }


def incentive_contract() -> dict[str, Any]:
    return {
        "default_mode": C.MODE_GRADED_CONTRIBUTION,
        "graded_shares": list(C.GRADED_TOP3_SHARES),
        "burn_uid": C.BURN_UID,
        "contribution_weights": {
            "quality": C.CONTRIBUTION_WEIGHT_QUALITY,
            "improvement": C.CONTRIBUTION_WEIGHT_IMPROVEMENT,
            "proximity": C.CONTRIBUTION_WEIGHT_PROXIMITY,
            "cost": C.CONTRIBUTION_WEIGHT_COST,
        },
        "note": (
            "The leader holds most of the workflow weight and everything below it is "
            "graded on quality, improvement, proximity and cost, so a submission "
            "that moved the state of the art without leading is still paid for what "
            "it contributed. Only candidates clearing every hard gate are graded. If "
            "nothing qualifies, the share is burned rather than redistributed."
        ),
    }


def recipe_contract(snapshot) -> dict[str, Any]:
    return {
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
    }
