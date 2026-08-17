"""One window, run by a validator, with nobody above it.

This is the loop that replaces the operator. A validator derives the window from
a block hash, works out which instances are its own, measures every candidate it
can see, ranks them on what it measured, and sets weights from that. No signed
vector arrives from anywhere, and there is no allow-list of whose evaluation to
believe, because the validator is not believing anyone's.

Everything that varies is injected. Chain access, serving and reconstruction are
all parameters, so the mechanism can be tested end to end without a GPU or a
node — the ordering and the arithmetic are what matter here, and both are pure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CandidateScores, Recipe, WeightVector
from capability_subnet.scoring.comparator import minimum_detectable_effect
from capability_subnet.scoring.ranking import Submission, rank
from capability_subnet.scoring.sampler import WindowSample, draw_window_open
from capability_subnet.scoring.weight_vector import graded_top3
from capability_subnet.validator.agreement import outlier_validators
from capability_subnet.validator.assignment import Assignment, assign
from capability_subnet.validator.evaluator import CandidateEvaluation, core_results

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A miner's submission, as the chain presents it."""

    uid: int
    hotkey: str
    recipe: Recipe
    #: Block the commitment was made at. Carried because it is the tie-break:
    #: when two submissions are closer than the window can resolve, the earlier
    #: commitment ranks first, which is what stops a copy taking a slot from the
    #: thing it copied.
    first_block: int = 0


@dataclass(slots=True)
class WindowOutcome:
    """What one validator concluded, and the evidence for it."""

    window_id: int
    beacon: str
    sample: WindowSample
    assignment: Assignment
    evaluations: list[CandidateEvaluation] = field(default_factory=list)
    weights: WeightVector | None = None
    #: Peers whose core results are inconsistent with the majority. Reported
    #: rather than acted on: which validators to distrust is a network-level
    #: decision, and one validator concluding it alone is how a split becomes
    #: two factions each certain the other is lying.
    flagged_peers: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> list[CandidateEvaluation]:
        return [e for e in self.evaluations if e.usable]


#: Given a candidate and the instances to measure it on, produce a measurement.
#: Injected so the loop can be exercised without reconstruction or serving.
Measure = Callable[["Candidate", Assignment], CandidateEvaluation]


def run_window(
    candidates: list[Candidate],
    *,
    window_id: int,
    beacon: str,
    hotkey: str,
    block: int,
    measure: Measure,
    hidden_count: int = C.DEFAULT_HIDDEN_INSTANCES,
    ood_count: int = C.DEFAULT_OOD_INSTANCES,
    workflow_id: str = C.DEFAULT_WORKFLOW_ID,
    burn_percentage: float = 0.0,
    burn_uid: int = C.BURN_UID,
    peer_core_results: dict[str, dict[int, bool]] | None = None,
) -> WindowOutcome:
    """Evaluate a window and produce this validator's own weight vector.

    Args:
        peer_core_results: other validators' core results for one shared
            candidate, when this validator has them. Used only to report
            inconsistency — a miner is not paid less because another validator
            looks wrong, since the miner did not do anything.
    """
    sample = draw_window_open(
        window_id, beacon=beacon, hidden_count=hidden_count, ood_count=ood_count
    )
    assignment = assign(sample.hidden_seeds, hotkey=hotkey, beacon=beacon)
    outcome = WindowOutcome(
        window_id=window_id, beacon=beacon, sample=sample, assignment=assignment
    )

    for candidate in candidates:
        try:
            outcome.evaluations.append(measure(candidate, assignment))
        except Exception as exc:  # one bad candidate must not stop the window
            log.warning("uid %s could not be measured: %s", candidate.uid, exc)
            outcome.evaluations.append(
                CandidateEvaluation(
                    candidate_id=candidate.hotkey,
                    recipe_sha256=candidate.recipe.digest(),
                    artifact_sha256="",
                    artifact_bytes=0,
                    scores=CandidateScores(),
                    error=str(exc),
                )
            )

    if peer_core_results:
        outcome.flagged_peers = _flag_peers(outcome, hotkey, peer_core_results)

    outcome.weights = _weights_from(
        outcome,
        candidates=candidates,
        window_id=window_id,
        block=block,
        workflow_id=workflow_id,
        burn_percentage=burn_percentage,
        burn_uid=burn_uid,
    )
    return outcome


def _flag_peers(
    outcome: WindowOutcome,
    hotkey: str,
    peer_core_results: dict[str, dict[int, bool]],
) -> dict[str, str]:
    """Compare this validator against its peers on one shared candidate.

    One candidate is enough. A validator that is not doing the work diverges on
    everything it reports, so adding subjects multiplies the comparison without
    adding evidence.
    """
    mine = {e.candidate_id: core_results(e, outcome.assignment) for e in outcome.usable}
    subject = next(
        (c for c in mine if all(c in peer for peer in peer_core_results.values())),
        None,
    )
    if subject is None:
        return {}

    table = {hotkey: mine[subject]}
    table.update({who: peer[subject] for who, peer in peer_core_results.items()})
    return outlier_validators(table)


def _weights_from(
    outcome: WindowOutcome,
    *,
    candidates: list[Candidate],
    window_id: int,
    block: int,
    workflow_id: str,
    burn_percentage: float,
    burn_uid: int,
) -> WeightVector:
    """Rank what was measured and turn it into weights.

    An unmeasurable candidate earns nothing and burns nothing extra: it is simply
    absent, exactly as a miner who never submitted is absent. Paying it would
    reward a recipe that does not reconstruct; charging the window for it would
    let one broken submission tax everybody else.

    ``resolvable`` is this validator's own figure, not the network's. It measured
    a slice of the window rather than all of it, so the gap it can honestly
    separate is wider than the full draw's — and two candidates inside that gap
    are one measurement repeated, which the ranking treats as a tie broken by
    who committed first.
    """
    by_hotkey = {c.hotkey: c for c in candidates}
    resolvable = minimum_detectable_effect(len(outcome.assignment))

    submissions = [
        Submission(
            uid=by_hotkey[e.candidate_id].uid,
            hotkey=e.candidate_id,
            score=e.scores.qualified_score,
            first_block=by_hotkey[e.candidate_id].first_block,
            resolvable=resolvable,
        )
        for e in outcome.usable
        if e.candidate_id in by_hotkey
    ]

    ordered = [(s.uid, s.hotkey) for s in rank(submissions)]
    return graded_top3(
        ordered,
        window_id=window_id,
        block=block,
        workflow_id=workflow_id,
        burn_percentage=burn_percentage,
        burn_uid=burn_uid,
    )
