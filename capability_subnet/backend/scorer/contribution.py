"""Grading a candidate that did not take the throne.

The champion-challenger rule answers one question — who holds the throne — and
answers it correctly. As a *payment* rule it is far too blunt, because almost
every submission that ever gets evaluated will fail to dethrone, and paying them
all identically (nothing) discards the only information the network has about
which failures were close and which were hopeless.

That matters here more than in a subnet where miners submit code they can iterate
on. A recipe is one shot. A miner who moves end-to-end completion from 0.41 to
0.58 against a champion at 0.60 has demonstrated something real about which
adapters compose, and telling them apart from a miner who submitted a distractor
soup is exactly what a network trying to *learn* composition should do.

So the throne stays winner-takes-most, and everything below it is graded on four
things the evaluation already measures:

* **quality** — the qualified score, which already blends completion, stage
  balance, out-of-distribution robustness, retention, latency, tokens and size;
* **improvement** — how far the package moved past the strongest non-learned
  reference, which is the network's definition of "composition added value";
* **proximity** — how close it came to the champion, which is what distinguishes
  a near miss from a wasted registration;
* **cost** — token and latency efficiency, because two packages that finish the
  same fraction of workflows are not equally valuable if one costs twice as much
  to run.

Nothing here can pay a package that failed a hard gate. Grading applies *within*
the qualified set; it is not a consolation prize for producing something
undeployable.
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
    #: End-to-end completion of the strongest permanent reference in the window
    #: this candidate was measured in.
    reference_e2e: float
    #: The reigning champion's completion in that window, or ``None`` when the
    #: throne was empty.
    champion_e2e: float | None


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


def proximity_to_champion(candidate_e2e: float, champion_e2e: float | None) -> float:
    """How close the package came to the incumbent, in [0, 1].

    One when there is no incumbent, or when the candidate matched or beat it.
    Otherwise the ratio, which falls off smoothly — a candidate three points
    short scores near one, and a candidate at half the champion's completion
    scores near a half.

    This is the term that distinguishes a near miss from a hopeless one. It is
    deliberately *not* a gate: rewarding proximity alone would pay for copying
    the champion, which is why it is one term of four and why the anti-copy rule
    sits in front of the evaluation entirely.
    """
    if champion_e2e is None or champion_e2e <= 0.0:
        return 1.0
    return max(0.0, min(1.0, candidate_e2e / champion_e2e))


def cost_efficiency(scores: CandidateScores) -> float:
    """The blended running cost of the finished package, in [0, 1].

    Token spend and latency weighted equally. Both are already scored components;
    they are re-blended here so that cost carries real weight in the *grade*
    rather than the 10% it carries in the qualified score, where quality
    correctly dominates. What a buyer pays per workflow run is a first-order
    property of a deployable package, not a tiebreak.
    """
    return 0.5 * scores.token_efficiency + 0.5 * scores.latency


def contribution_score(inputs: ContributionInputs) -> float:
    """Grade one qualified candidate, in [0, 1].

    Quality dominates, because a package that does not finish workflows is not
    made valuable by being cheap or by being nearly as good as something that
    does.
    """
    scores = inputs.scores
    improvement = improvement_over_reference(scores.end_to_end, inputs.reference_e2e)
    proximity = proximity_to_champion(scores.end_to_end, inputs.champion_e2e)
    cost = cost_efficiency(scores)

    graded = (
        C.CONTRIBUTION_WEIGHT_QUALITY * scores.qualified_score
        + C.CONTRIBUTION_WEIGHT_IMPROVEMENT * improvement
        + C.CONTRIBUTION_WEIGHT_PROXIMITY * proximity
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
        "proximity": proximity_to_champion(scores.end_to_end, inputs.champion_e2e),
        "cost": cost_efficiency(scores),
        "contribution": contribution_score(inputs),
    }
