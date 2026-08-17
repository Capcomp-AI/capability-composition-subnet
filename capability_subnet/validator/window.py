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
from capability_subnet.scoring.contribution import ContributionInputs, contribution_score
from capability_subnet.scoring.ranking import Submission, rank
from capability_subnet.scoring.sampler import WindowSample, draw_window_open
from capability_subnet.scoring.weight_vector import graded_contribution, graded_top3
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
    incentive_mode: str = C.MODE_GRADED_TOP3,
    reference_e2e: float = 0.0,
    workers: int = 1,
    peer_core_results: dict[str, dict[int, bool]] | None = None,
) -> WindowOutcome:
    """Evaluate a window and produce this validator's own weight vector.

    Args:
        workers: how many candidates to measure at once. One per GPU: a
            candidate reserves almost the whole card while it is served, so
            concurrency above the device count would make the packages contend
            for memory and measure each other rather than themselves.
        incentive_mode: how the measured field is turned into weights. See
            :func:`_weights_from`.
        reference_e2e: end-to-end completion of the strongest permanent
            reference, used by the graded mode's improvement term. This loop
            measures committed candidates only and never a reference, so there
            is no honest value to derive here — it is a parameter rather than a
            default so a caller that *does* measure references can supply one,
            and a caller that does not is visibly claiming zero.
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

    outcome.evaluations.extend(_measure_all(candidates, assignment, measure, workers))

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
        incentive_mode=incentive_mode,
        reference_e2e=reference_e2e,
    )
    return outcome


def _failed(candidate: Candidate, exc: Exception) -> CandidateEvaluation:
    """A candidate this host could not measure.

    Recorded rather than dropped so the window still reports how many candidates
    it saw, and so a reader can tell "measured and scored zero" from "never
    measured" — which are the same number and very different claims.
    """
    return CandidateEvaluation(
        candidate_id=candidate.hotkey,
        recipe_sha256=candidate.recipe.digest(),
        artifact_sha256="",
        artifact_bytes=0,
        scores=CandidateScores(),
        error=str(exc),
    )


def _measure_all(
    candidates: list[Candidate],
    assignment: Assignment,
    measure: Measure,
    workers: int,
) -> list[CandidateEvaluation]:
    """Measure every candidate, in commit order, ``workers`` at a time.

    Results are returned in the order the candidates were given rather than the
    order the measurements finished. Nothing downstream may depend on which
    device happened to be free first: two validators running different numbers
    of GPUs have to agree about a candidate, and a result list that reordered
    itself under concurrency would put the tie-break — and the peer comparison's
    choice of subject — on the hardware instead of on the chain.
    """
    if workers <= 1 or len(candidates) <= 1:
        results = []
        for candidate in candidates:
            try:
                results.append(measure(candidate, assignment))
            except Exception as exc:  # one bad candidate must not stop the window
                log.warning("uid %s could not be measured: %s", candidate.uid, exc)
                results.append(_failed(candidate, exc))
        return results

    from concurrent.futures import ThreadPoolExecutor

    ordered: list[CandidateEvaluation | None] = [None] * len(candidates)

    def run(index: int) -> None:
        candidate = candidates[index]
        try:
            ordered[index] = measure(candidate, assignment)
        except Exception as exc:  # one bad candidate must not stop the window
            log.warning("uid %s could not be measured: %s", candidate.uid, exc)
            ordered[index] = _failed(candidate, exc)

    log.info("measuring %d candidates, %d at a time", len(candidates), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(len(candidates))))

    return [e for e in ordered if e is not None]


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
    incentive_mode: str = C.MODE_GRADED_TOP3,
    reference_e2e: float = 0.0,
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

    ranked = rank(submissions)
    ordered = [(s.uid, s.hotkey) for s in ranked]

    if incentive_mode == C.MODE_GRADED_CONTRIBUTION:
        return _graded_contribution_weights(
            outcome,
            ranked=ranked,
            window_id=window_id,
            block=block,
            workflow_id=workflow_id,
            burn_percentage=burn_percentage,
            burn_uid=burn_uid,
            reference_e2e=reference_e2e,
        )

    return graded_top3(
        ordered,
        window_id=window_id,
        block=block,
        workflow_id=workflow_id,
        burn_percentage=burn_percentage,
        burn_uid=burn_uid,
    )


def _graded_contribution_weights(
    outcome: WindowOutcome,
    *,
    ranked: list[Submission],
    window_id: int,
    block: int,
    workflow_id: str,
    burn_percentage: float,
    burn_uid: int,
    reference_e2e: float,
) -> WeightVector:
    """Grade the measured field and let the graded split decide the payout.

    ``champion`` is deliberately ``None``. The throne is held by whoever last
    *dethroned* the incumbent, and that is a fact about the subnet's history
    rather than about this window — nothing in the ownerless loop carries a
    ChampionRecord between windows, so this validator has not seen anyone take
    it. Synthesising one from the window's own leader would crown a package for
    winning a field of one, skip the leaderless burn entirely, and pay the full
    champion share every window regardless of whether anything was dethroned.

    So the leaderless branch is the honest one, and it is also the arrangement
    the contract describes for it: half the window burns, the best measured
    package leads what remains on the champion's terms, and the rest of the
    graded field splits what is left of that. Because the function pops the
    leader out of the graded list *after* capping it, the miners sharing the
    remainder are exactly ranks two through ten.

    Grades come from the four terms the published contract already defines, over
    scores this window produced. A candidate graded zero does not appear at all,
    and the share nobody earned burns rather than inflating the leader's.
    """
    scores_by_hotkey = {e.candidate_id: e.scores for e in outcome.usable}

    contributors: list[tuple[int, str, float]] = []
    for submission in ranked:
        scores = scores_by_hotkey.get(submission.hotkey)
        if scores is None:
            continue
        grade = contribution_score(
            ContributionInputs(
                scores=scores,
                reference_e2e=reference_e2e,
                # No throne, so proximity has no incumbent to measure against
                # and scores one for everybody. That is the correct reading:
                # nobody was close to a champion, because there was no champion.
                champion_e2e=None,
            )
        )
        if grade > 0.0:
            contributors.append((submission.uid, submission.hotkey, grade))

    return graded_contribution(
        None,
        contributors,
        window_id=window_id,
        block=block,
        workflow_id=workflow_id,
        burn_percentage=burn_percentage,
        burn_uid=burn_uid,
    )
