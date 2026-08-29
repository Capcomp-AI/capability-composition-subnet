"""Grading a measured candidate.

One number per candidate, in [0, 1], and it is what everything downstream
decides on: who holds the throne, who is paid, and in what order. A candidate
takes the throne by exceeding the reigning grade by
``CHAMPION_DETHRONE_MARGIN``. Being paid is a separate question and a lower
bar: everything clearing the hard gates is ranked by this number and paid by
rank, so a run that crowns nobody still pays its field.

Three terms the evaluation already measures:

* **quality** — the qualified score, which blends completion, stage balance,
  out-of-distribution robustness, retention, tokens and size;
* **improvement** — how far the package moved past the strongest permanent
  reference, which is the network's definition of "composition added value";
* **cost** — token efficiency, because two packages that finish the same
  fraction of workflows are not equally valuable if one costs twice as much to
  run.

Every term is measured against fixed points: the run's own instances and the
permanent reference. Nothing is measured against the incumbent, so a grade
means the same thing in every run and a fixed dethrone margin is a fixed bar.
A term relative to the throne would move the scale each time the throne
changed hands.

Nothing here can pay a package that failed a hard gate. Grading applies
*within* the qualified set; it is not a consolation prize for producing
something undeployable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CandidateScores

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContributionInputs:
    """What one candidate's grade is computed from."""

    scores: CandidateScores
    #: End-to-end completion of the strongest permanent reference in the run
    #: this candidate was measured in.
    reference_e2e: float


def improvement_over_reference(candidate_e2e: float, reference_e2e: float) -> float:
    """How far past the strongest non-learned reference the package got, in [0, 1].

    Normalised by the headroom that actually remained rather than by a constant:
    moving completion from 0.90 to 0.95 is a larger achievement than moving it
    from 0.10 to 0.15, and dividing by ``1 - reference`` says so. A package at or
    below the reference scores zero — it did not demonstrate that composition
    added anything.
    """
    headroom = 1.0 - reference_e2e
    if headroom <= 0.0:
        # The reference already completes everything; there is nothing to add.
        return 0.0
    return max(0.0, min(1.0, (candidate_e2e - reference_e2e) / headroom))


def cost_efficiency(scores: CandidateScores) -> float:
    """The blended running cost of the finished package, in [0, 1].

    Token spend alone, now that latency is not scored: the two correlated at
    0.9992 on real instances, so blending them was averaging a quantity with
    itself. What a buyer pays per workflow run is a first-order property of a
    deployable package, not a tiebreak, which is why it is re-weighted here
    against the share it carries in the qualified score.
    """
    return scores.token_efficiency


def contribution_score(inputs: ContributionInputs) -> float:
    """Grade one qualified candidate, in [0, 1].

    Quality dominates, because a package that does not finish workflows is not
    made valuable by being cheap or by being nearly as good as something that
    does.
    """
    scores = inputs.scores
    improvement = improvement_over_reference(scores.end_to_end, inputs.reference_e2e)
    cost = cost_efficiency(scores)

    graded = (
        C.CONTRIBUTION_WEIGHT_QUALITY * scores.qualified_score
        + C.CONTRIBUTION_WEIGHT_IMPROVEMENT * improvement
        + C.CONTRIBUTION_WEIGHT_COST * cost
    )
    return max(0.0, min(1.0, graded))


def explain(inputs: ContributionInputs) -> dict[str, float]:
    """The grade broken into its terms, for the published report.

    A miner that earned a partial share is entitled to know which part of its
    package earned it — that is the difference between a signal it can act on
    and a number it has to guess at.
    """
    scores = inputs.scores
    return {
        "quality": scores.qualified_score,
        "improvement": improvement_over_reference(scores.end_to_end, inputs.reference_e2e),
        "cost": cost_efficiency(scores),
        "contribution": contribution_score(inputs),
    }
