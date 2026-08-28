"""The parts of a published contract that do not depend on the workflow.

The base model, the frozen pool, the recipe schema and its bounds are facts about
the *protocol*, not about whichever workflow is judging, and a miner needs them
whatever arena is running.

Kept here so two workflows cannot drift. A third gets them by calling this.
"""

from __future__ import annotations

from typing import Any

from capability_subnet.common import constants as C
from capability_subnet.common.merge_methods import PIPELINES
from capability_subnet.common.schemas import recipe_json_schema


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
        # Only what a recipe may actually name. Publishing every id in the
        # registry included adapters that are present but not certified for
        # selection, so the contract advertised a pool a third larger than the
        # one the engine would accept from — and a recipe that used one was
        # refused for naming an adapter the contract had listed.
        "adapters": list(snapshot.registry.selectable_ids),
        "unselectable": sorted(set(snapshot.adapter_ids) - set(snapshot.registry.selectable_ids)),
        "distractors": list(snapshot.registry.distractors()),
    }


def hard_gates_contract(
    *,
    max_output_tokens: int | None = None,
    max_turns: int | None = None,
    stage_floors: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Every hard gate, as a miner can read it.

    The token and turn budgets are overridable because they are the workflow's,
    not the protocol's: a workflow that asks one question and allows a thousand
    tokens published the protocol's eight thousand and one miner in every ten
    would have budgeted for eight times what they actually get.

    ``stage_floors`` is published for the same reason. A floor terminates a
    submission outright, and a threshold nobody can read is not a contract.
    """
    return {
        "artifact_size_bytes": C.MAX_ARTIFACT_BYTES,
        "max_turns": C.MAX_AGENT_TURNS if max_turns is None else max_turns,
        "max_output_tokens": C.MAX_OUTPUT_TOKENS
        if max_output_tokens is None
        else max_output_tokens,
        "critical_unsafe_actions": C.MAX_CRITICAL_UNSAFE_ACTIONS,
        "base_retention_floor": C.BASE_RETENTION_FLOOR,
        "stage_floors": dict(stage_floors or {}),
    }


def qualified_scoring_contract() -> dict[str, Any]:
    w = C.QUALIFIED_SCORE_WEIGHTS
    return {
        "weights": dict(w),
        # Terms in descending weight, and every one at the same precision: a
        # published formula a miner optimises against must not round a weight
        # to a different number than the one applied.
        "formula": "Q = "
        + " + ".join(
            f"{weight:.6f}·{axis}" for axis, weight in sorted(w.items(), key=lambda kv: -kv[1])
        ),
        "stage_balance": (
            "Geometric mean of the per-stage means. Rewards packages that are "
            "competent everywhere over packages that are excellent at one stage "
            "and broken at another."
        ),
    }


def ranking_contract() -> dict[str, Any]:
    """How the board is ordered, in both modes the engine supports.

    `require_beat_reference` decides which applies, and it ships off.
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
            "cleared the strongest permanent reference. Ranking is by measured "
            "score alone: an earlier commitment is never advanced over a "
            "higher-scoring later one, and an exact tie falls back to uid only so "
            "the order is reproducible. What stops a copy taking the throne is the "
            "comparator, not the sort — a challenger must beat the strongest "
            "reference, the incumbent included, by an absolute end-to-end margin, "
            "and a copy cannot beat what it copied by any margin. References are "
            "measured and published every run either way."
        ),
        "strict_rule": (
            "With require_beat_reference enabled, a challenger additionally has to "
            "dominate at least the required number of capability axes, be no worse "
            "on any other axis, raise end-to-end completion over the strongest "
            "reference by the absolute margin, and show a paired bootstrap lower "
            "confidence bound above zero against that reference."
        ),
        "always_enforced": (
            "Artifact size, sample sufficiency, agent limits, safety and the "
            "stage floors apply in both modes. Base retention is measured and "
            "scored in both and gates neither: the probe is scored against the "
            "base model's own score on a set redrawn each run, so a fixed floor "
            "fell between two reachable values and decided qualification on one "
            "drawn question."
        ),
    }


def runs_contract(root_commitment: str = "") -> dict[str, Any]:
    """How a run is sized, and what its instance draw is bound to.

    Args:
        root_commitment: hash of the operator's seed root. Published so a miner
            can see that one root produces every run: it reveals nothing about
            the root, and a value that changes between runs is the operator
            changing the draw where everyone can see it.
    """
    return {
        "run_blocks": C.DEFAULT_RUN_BLOCKS,
        "hidden_instances": C.DEFAULT_HIDDEN_INSTANCES,
        "ood_instances": C.DEFAULT_OOD_INSTANCES,
        "public_pack_instances": C.PUBLIC_PACK_INSTANCES,
        "seed_root_commitment": root_commitment,
        "note": (
            "Hidden instances are drawn per run from a secret root the operator "
            "holds, mixed with the hash of the block the run opened at. The "
            "block hash is public and not the operator's to choose, so the draw "
            "cannot be selected after seeing a candidate; the commitment above "
            "binds the operator to one root across every run. Both are "
            "published in each closed run's disclosure."
        ),
    }


def incentive_contract() -> dict[str, Any]:
    miner_pool = 1.0 - C.BURN_SHARE
    # Rounded because these are published figures a miner reads and compares
    # against its own arithmetic, and binary floating point renders an exact
    # fifth as 0.19999999999999996.
    return {
        "burn_uid": C.BURN_UID,
        "burn_share": round(C.BURN_SHARE, 6),
        "miner_pool_share": round(miner_pool, 6),
        "rank_shares_of_pool": [round(share, 6) for share in C.RANK_SHARES],
        "rank_shares_of_run": [round(share * miner_pool, 6) for share in C.RANK_SHARES],
        "tail_share_of_pool": round(C.TAIL_SHARE, 6),
        "tail_share_of_run": round(C.TAIL_SHARE * miner_pool, 6),
        "paid_ranks": C.PAID_RANKS,
        "champion_dethrone_margin": C.CHAMPION_DETHRONE_MARGIN,
        "contribution_weights": {
            "quality": C.CONTRIBUTION_WEIGHT_QUALITY,
            "improvement": C.CONTRIBUTION_WEIGHT_IMPROVEMENT,
            "cost": C.CONTRIBUTION_WEIGHT_COST,
        },
        "note": (
            (
                "No fixed share of a run burns: a run that produces a new "
                "champion and fills every paid rank pays its whole emission to "
                "miners. "
                if C.BURN_SHARE == 0
                else f"{C.BURN_SHARE:.0%} of every run burns to the subnet owner's UID. "
            )
            + "Every candidate that clears the hard gates is paid; the bar is "
            "the entry gate, an absolute margin of end-to-end completion over "
            "the strongest permanent reference, so it is a statement about the "
            "package rather than about the incumbent. A run where nothing "
            f"clears it burns entirely. Exceeding the champion's grade by "
            f"{C.CHAMPION_DETHRONE_MARGIN} takes the throne, which records who "
            "leads the network and does not change what anyone is paid. The "
            "pool is split by rank — "
            + ", ".join(f"{share:.1%}" for share in C.RANK_SHARES)
            + f" for the first {len(C.RANK_SHARES)}, and {C.TAIL_SHARE:.1%} "
            f"across ranks {len(C.RANK_SHARES) + 1} to {C.PAID_RANKS} in "
            "proportion to grade. Grading is on quality, improvement and cost, "
            "and only candidates clearing every hard gate are graded. An "
            f"unfilled rank in the first {len(C.RANK_SHARES)} burns rather than "
            f"being redistributed, so a field of {len(C.RANK_SHARES)} pays "
            f"{sum(C.RANK_SHARES):.1%} of the pool and burns the rest; the "
            f"tail share is split across whoever occupies ranks "
            f"{len(C.RANK_SHARES) + 1} to {C.PAID_RANKS}, so it is paid in full "
            "once any one of them is filled."
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
