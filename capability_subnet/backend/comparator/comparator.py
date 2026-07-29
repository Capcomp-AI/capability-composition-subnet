"""The dethrone decision.

A challenger takes the throne only by clearing four independent bars:

1. **Per-axis dominance.** It must be clearly better on at least the required
   number of capability axes.
2. **No abandoned capability.** It must be no worse on every remaining axis. This
   is the bar that stops a package from trading away, say, safety compliance for
   a better SQL score — the average would improve, and the package would be
   worse at the job.
3. **An absolute end-to-end margin** over the strongest reference, not merely
   over the incumbent. That keeps the network from ever crowning a package an
   off-the-shelf merge already beats.
4. **Statistical significance** on the paired comparison against that reference.

Everything is measured on instances both packages actually completed. Comparing
averages over different instance sets would let a challenger win by being
evaluated on an easier draw.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from capability_subnet.backend.comparator.bootstrap import (
    outcome_map,
    paired_bootstrap,
    paired_differences,
    stage_outcome_map,
)
from capability_subnet.common import constants as C
from capability_subnet.common.schemas import (
    AxisVerdict,
    ComparatorOutcome,
    InstanceResult,
    PairedComparison,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ComparatorConfig:
    """The thresholds the dethrone rule uses. Published in the contract."""

    axis_margin: float = C.DEFAULT_AXIS_MARGIN
    axis_tolerance: float = C.DEFAULT_AXIS_TOLERANCE
    min_dominant_axes: int = C.DEFAULT_MIN_DOMINANT_AXES
    min_axis_samples: int = C.DEFAULT_MIN_AXIS_SAMPLES
    end_to_end_margin: float = C.DEFAULT_END_TO_END_MARGIN
    #: Margin over the *incumbent*, before decay. Separate from the reference
    #: margin so beating the previous winner does not get permanently harder.
    champion_margin: float = C.DEFAULT_CHAMPION_MARGIN
    champion_margin_decay_blocks: int = C.CHAMPION_MARGIN_DECAY_BLOCKS
    bootstrap_resamples: int = C.BOOTSTRAP_RESAMPLES
    bootstrap_confidence: float = C.BOOTSTRAP_CONFIDENCE
    strict_pareto: bool = False


def decayed_champion_margin(config: ComparatorConfig, *, blocks_held: int) -> float:
    """The margin a challenger must clear over the incumbent right now.

    Full at the moment of crowning, falling linearly to zero over the decay
    window. A defender's advantage is worth having — it stops a copy of the
    champion from trading places with it on noise — but it should be an
    advantage for holding the throne *well*, not a freehold. An incumbent that
    nothing has displaced for a month is either genuinely excellent or simply
    unopposed, and the network cannot tell those apart from the inside; letting
    the bar fall resolves it in favour of the contest continuing.
    """
    if config.champion_margin_decay_blocks <= 0:
        return config.champion_margin
    elapsed = max(0, blocks_held) / config.champion_margin_decay_blocks
    return max(0.0, config.champion_margin * (1.0 - min(1.0, elapsed)))


def compare_axis(
    axis: str,
    challenger: list[InstanceResult],
    champion: list[InstanceResult],
    config: ComparatorConfig,
) -> AxisVerdict:
    """Classify one capability axis as dominant, not worse, or worse.

    Both means are taken over the instances *both* packages scored. An axis with
    too few paired samples counts as worse, not as a tie: absence of evidence
    that a challenger kept a capability is not evidence that it did.
    """
    challenger_scores = stage_outcome_map(challenger, axis)
    champion_scores = stage_outcome_map(champion, axis)

    shared = sorted(set(challenger_scores) & set(champion_scores))
    paired = len(shared)

    if paired < config.min_axis_samples:
        return AxisVerdict(
            axis=axis,
            challenger_mean=0.0,
            champion_mean=0.0,
            paired_samples=paired,
            verdict="worse",
            margin=config.axis_margin,
            tolerance=config.axis_tolerance,
        )

    challenger_mean = sum(challenger_scores[key] for key in shared) / paired
    champion_mean = sum(champion_scores[key] for key in shared) / paired

    if challenger_mean > champion_mean + config.axis_margin:
        verdict = "dominant"
    elif challenger_mean >= champion_mean * (1.0 - config.axis_tolerance):
        verdict = "not_worse"
    else:
        verdict = "worse"

    return AxisVerdict(
        axis=axis,
        challenger_mean=challenger_mean,
        champion_mean=champion_mean,
        paired_samples=paired,
        verdict=verdict,  # type: ignore[arg-type]
        margin=config.axis_margin,
        tolerance=config.axis_tolerance,
    )


def compare(
    challenger: list[InstanceResult],
    champion: list[InstanceResult],
    reference: list[InstanceResult],
    *,
    axes: tuple[str, ...],
    reference_id: str,
    config: ComparatorConfig | None = None,
    bootstrap_seed: int = 0,
    champion_blocks_held: int = 0,
) -> ComparatorOutcome:
    """Decide whether ``challenger`` dethrones the incumbent.

    Args:
        challenger: the challenger's sample rows for this window.
        champion: the incumbent's rows for the same window.
        reference: the strongest reference's rows — the incumbent itself when
            the incumbent is the strongest thing on the board.
        axes: the workflow's capability axes.
        reference_id: which reference ``reference`` belongs to, for the report.
        bootstrap_seed: fixes the resampling so the verdict is reproducible.
        champion_blocks_held: how long the incumbent has held the throne, which
            sets how much of its defender's advantage remains.

    Returns:
        The full outcome: every axis verdict, the paired statistics, and the
        decision with its reason.
    """
    config = config or ComparatorConfig()

    verdicts = [compare_axis(axis, challenger, champion, config) for axis in axes]
    dominant = [verdict for verdict in verdicts if verdict.verdict == "dominant"]
    worse = [verdict for verdict in verdicts if verdict.verdict == "worse"]

    required = len(axes) if config.strict_pareto else config.min_dominant_axes

    champion_margin = decayed_champion_margin(config, blocks_held=champion_blocks_held)
    champion_outcomes = outcome_map(champion)
    challenger_outcomes = outcome_map(challenger)
    reference_outcomes = outcome_map(reference)
    differences = paired_differences(challenger_outcomes, reference_outcomes)

    shared_with_champion = sorted(set(challenger_outcomes) & set(champion_outcomes))
    champion_margin_observed = (
        sum(challenger_outcomes[k] for k in shared_with_champion) / len(shared_with_champion)
        - sum(champion_outcomes[k] for k in shared_with_champion) / len(shared_with_champion)
        if shared_with_champion
        else 0.0
    )

    shared = sorted(set(challenger_outcomes) & set(reference_outcomes))
    challenger_e2e = (
        sum(challenger_outcomes[key] for key in shared) / len(shared) if shared else 0.0
    )
    reference_e2e = sum(reference_outcomes[key] for key in shared) / len(shared) if shared else 0.0
    observed_margin = challenger_e2e - reference_e2e

    bootstrap = paired_bootstrap(
        differences,
        resamples=config.bootstrap_resamples,
        confidence=config.bootstrap_confidence,
        seed=bootstrap_seed,
    )

    paired = PairedComparison(
        reference_id=reference_id,
        paired_instances=len(shared),
        challenger_mean=challenger_e2e,
        reference_mean=reference_e2e,
        difference=bootstrap.difference,
        bootstrap_lcb=bootstrap.lower_confidence_bound,
        bootstrap_resamples=bootstrap.resamples,
        confidence=bootstrap.confidence,
        passed=bootstrap.significant,
    )

    reasons: list[str] = []
    if len(dominant) < required:
        reasons.append(
            f"dominant on {len(dominant)} of the required {required} axes "
            f"({', '.join(v.axis for v in dominant) or 'none'})"
        )
    if worse:
        reasons.append(
            "regressed on " + ", ".join(f"{v.axis} ({v.paired_samples} paired)" for v in worse)
        )
    if observed_margin < config.end_to_end_margin:
        reasons.append(
            f"end-to-end margin {observed_margin:+.3f} over {reference_id} "
            f"below the required {config.end_to_end_margin:+.3f}"
        )
    if shared_with_champion and champion_margin_observed < champion_margin:
        reasons.append(
            f"margin {champion_margin_observed:+.3f} over the incumbent below the "
            f"{champion_margin:+.3f} it still holds ({champion_blocks_held} blocks in)"
        )
    if not bootstrap.significant:
        reasons.append(
            f"paired lower bound {bootstrap.lower_confidence_bound:+.4f} is not above zero"
        )

    dethrones = not reasons

    return ComparatorOutcome(
        per_axis_verdicts=verdicts,
        dominant_count=len(dominant),
        min_dominant_required=required,
        any_worse_axis=bool(worse),
        strict_pareto=config.strict_pareto,
        end_to_end_margin_required=config.end_to_end_margin,
        end_to_end_margin_observed=observed_margin,
        champion_margin_required=champion_margin,
        champion_margin_observed=champion_margin_observed,
        paired=paired,
        dethrones=dethrones,
        reason="dethrones the incumbent" if dethrones else "; ".join(reasons),
    )


def decisive_loss(outcome: ComparatorOutcome) -> bool:
    """Whether the challenger should be terminated rather than retried.

    A loss is decisive when the challenger was genuinely measured and genuinely
    lost. It is *not* decisive when the comparison could not be made — too few
    paired samples on some axis means the engine did not gather the evidence, and
    terminating on that would punish a candidate for an evaluation the engine
    failed to complete.
    """
    starved = [
        verdict
        for verdict in outcome.per_axis_verdicts
        if verdict.verdict == "worse" and verdict.paired_samples == 0
    ]
    if starved:
        log.warning(
            "not treating this as a decisive loss: %d axes had no paired samples",
            len(starved),
        )
        return False
    return not outcome.dethrones


def strongest_reference(
    reference_scores: dict[str, float],
) -> tuple[str, float]:
    """Pick the reference a challenger has to beat.

    Ties resolve to the alphabetically first identifier so the choice is stable
    across runs, which keeps the published report reproducible.
    """
    if not reference_scores:
        return "", 0.0
    best = max(reference_scores.values())
    winner = min(name for name, value in reference_scores.items() if value == best)
    return winner, best
