"""Paired bootstrap.

The question the comparator has to answer is not "did the challenger score
higher" — with a hundred instances, a package that is genuinely no better than
the incumbent will score higher about half the time. The question is whether the
observed difference is larger than the noise in the measurement.

The test is *paired* because both packages are run on the same instances. Pairing
removes instance difficulty from the comparison entirely: what gets resampled is
the vector of per-instance differences, not two independent score distributions.
That is a substantially tighter test, and it is only available because the engine
controls which instances both sides saw.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from capability_subnet.common import constants as C


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Outcome of a one-sided paired bootstrap."""

    difference: float
    lower_confidence_bound: float
    resamples: int
    confidence: float
    paired_count: int

    @property
    def significant(self) -> bool:
        """Whether the challenger is better than the reference with confidence.

        A lower bound above zero means that across the resampled worlds, the
        challenger was ahead in at least the confidence fraction of them.
        """
        return self.lower_confidence_bound > 0.0


def paired_differences(challenger: dict[str, float], reference: dict[str, float]) -> list[float]:
    """Per-instance differences over the instances both sides have.

    Args:
        challenger: instance identifier to outcome, typically 1.0 or 0.0.
        reference: the same, for the package being compared against.

    Returns:
        One difference per shared instance, in sorted instance order so the
        vector does not depend on dictionary iteration order.
    """
    shared = sorted(set(challenger) & set(reference))
    return [challenger[key] - reference[key] for key in shared]


def paired_bootstrap(
    differences: list[float],
    *,
    resamples: int = C.BOOTSTRAP_RESAMPLES,
    confidence: float = C.BOOTSTRAP_CONFIDENCE,
    seed: int = 0,
) -> BootstrapResult:
    """One-sided lower confidence bound on the mean paired difference.

    Args:
        differences: per-instance differences, challenger minus reference.
        resamples: how many bootstrap worlds to draw.
        confidence: one-sided confidence level.
        seed: fixes the resampling so the same inputs always give the same
            verdict. A statistical test whose answer changes between runs is not
            a test anyone can dispute or reproduce.

    Returns:
        The observed mean difference and its lower bound.
    """
    count = len(differences)
    if count == 0:
        return BootstrapResult(0.0, 0.0, resamples, confidence, 0)

    observed = sum(differences) / count

    if count == 1:
        # A single pair carries no information about variance. Reporting a bound
        # of zero makes the significance test fail, which is the correct
        # outcome — not a bound equal to the point estimate, which would let one
        # lucky instance dethrone a champion.
        return BootstrapResult(observed, 0.0, resamples, confidence, 1)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += differences[rng.randrange(count)]
        means.append(total / count)

    means.sort()
    index = int((1.0 - confidence) * resamples)
    index = max(0, min(resamples - 1, index))

    return BootstrapResult(
        difference=observed,
        lower_confidence_bound=means[index],
        resamples=resamples,
        confidence=confidence,
        paired_count=count,
    )


def outcome_map(results: list, *, attribute: str = "end_to_end_success") -> dict[str, float]:
    """Build the instance-to-outcome mapping the bootstrap consumes.

    Only rows that scored are included; a harness failure has no outcome to
    compare, and coercing it to zero would attribute the engine's problem to the
    candidate.
    """
    mapping: dict[str, float] = {}
    for row in results:
        if not row.is_valid_sample:
            continue
        value = getattr(row, attribute)
        mapping[row.instance_id] = float(value)
    return mapping


def stage_outcome_map(results: list, stage: str) -> dict[str, float]:
    """The same mapping, for one capability axis."""
    return {row.instance_id: row.stage_score(stage) for row in results if row.is_valid_sample}
