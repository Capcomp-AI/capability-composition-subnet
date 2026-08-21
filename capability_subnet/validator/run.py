"""One run, run by a validator, with nobody above it.

This is the loop that replaces the operator. A validator derives the run from
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
from capability_subnet.scoring.retention import ProbeOutcome
from capability_subnet.scoring.sampler import RunSample, draw_run_open
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
    #: Block the commitment was made at. Carried for the record and for audit;
    #: it no longer affects ranking, which is by measured score alone. A copy is
    #: kept off the throne by the champion margin, not by commit order.
    first_block: int = 0


@dataclass(slots=True)
class RunOutcome:
    """What one validator concluded, and the evidence for it."""

    run_id: int
    beacon: str
    sample: RunSample
    assignment: Assignment
    evaluations: list[CandidateEvaluation] = field(default_factory=list)
    weights: WeightVector | None = None
    #: The bar this run measured. Recorded so a reader can tell what the
    #: candidates were held to rather than inferring it from the winner.
    reference_e2e: float = 0.0
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
@dataclass(frozen=True)
class BaseMeasurement:
    """The base model on this run's own draw.

    The bar every candidate is held to, and the probe retention is scored
    against. Both have to come from the same run as the candidates — a
    reference measured on a different draw is not a paired comparison, which is
    the property the whole design rests on.
    """

    end_to_end: float
    probe: ProbeOutcome
    reference_id: str = "reference:base_model"


@dataclass(frozen=True)
class RunInputs:
    """Everything a measurement needs that is not the candidate itself.

    Passed as one object rather than a widening argument list because the
    previous signature made it possible — and it happened — to call a
    measurement without the out-of-distribution draw or the base probe, which
    scores those terms zero and one respectively without anything saying so.
    """

    assignment: Assignment
    ood_seeds: tuple[int, ...]
    probe_seed: int
    base: BaseMeasurement


Measure = Callable[["Candidate", RunInputs], CandidateEvaluation]

#: Measures the base model on this run's draw. Required rather than
#: defaulted: a run with no reference cannot say a candidate improved on
#: anything, and defaulting the reference to zero made "improvement" mean
#: "score", silently.
MeasureBase = Callable[[Assignment, "RunSample"], BaseMeasurement]


def evaluate_run(
    candidates: list[Candidate],
    *,
    run_id: int,
    beacon: str,
    hotkey: str,
    block: int,
    measure: Measure,
    measure_base: MeasureBase,
    hidden_count: int = C.DEFAULT_HIDDEN_INSTANCES,
    ood_count: int = C.DEFAULT_OOD_INSTANCES,
    workflow_id: str = C.DEFAULT_WORKFLOW_ID,
    burn_percentage: float = 0.0,
    burn_uid: int = C.BURN_UID,
    incentive_mode: str = C.MODE_GRADED_TOP3,
    workers: int = 1,
    peer_core_results: dict[str, dict[int, bool]] | None = None,
) -> RunOutcome:
    """Evaluate a run and produce this validator's own weight vector.

    Args:
        workers: how many candidates to measure at once. One per GPU: a
            candidate reserves almost the whole card while it is served, so
            concurrency above the device count would make the packages contend
            for memory and measure each other rather than themselves.
        incentive_mode: how the measured field is turned into weights. See
            :func:`_weights_from`.
        measure_base: measures the base model on this run's draw. The base
            model is the only permanent reference, and this loop cannot produce
            a bar without it. It used to be a ``reference_e2e`` parameter
            defaulting to zero, which made the graded mode's improvement term
            equal to the candidate's raw score and let a package that beat
            nothing collect improvement credit.
        peer_core_results: other validators' core results for one shared
            candidate, when this validator has them. Used only to report
            inconsistency — a miner is not paid less because another validator
            looks wrong, since the miner did not do anything.
    """
    sample = draw_run_open(
        run_id, beacon=beacon, hidden_count=hidden_count, ood_count=ood_count
    )
    assignment = assign(sample.hidden_seeds, hotkey=hotkey, beacon=beacon)
    outcome = RunOutcome(
        run_id=run_id, beacon=beacon, sample=sample, assignment=assignment
    )

    # The reference first, and alone. Every candidate's retention is scored
    # against this probe and every candidate's margin against this score, so
    # measuring it on the same draw is what makes the comparison paired.
    base = measure_base(assignment, sample)
    log.info(
        "run %d reference %s: end_to_end %.4f, probe %d/%d",
        run_id,
        base.reference_id,
        base.end_to_end,
        base.probe.correct,
        base.probe.total,
    )
    inputs = RunInputs(
        assignment=assignment,
        ood_seeds=sample.ood_seeds,
        probe_seed=sample.probe_seed,
        base=base,
    )
    outcome.reference_e2e = base.end_to_end

    outcome.evaluations.extend(_measure_all(candidates, inputs, measure, workers))

    for evaluation in outcome.evaluations:
        if evaluation.error:
            log.warning("%s could not be measured: %s", evaluation.candidate_id[:12], evaluation.error)
        elif not evaluation.usable:
            log.info(
                "%s did not clear its gates: %s",
                evaluation.candidate_id[:12],
                "; ".join(evaluation.gate_failures) or "no gates ran",
            )

    if peer_core_results:
        outcome.flagged_peers = _flag_peers(outcome, hotkey, peer_core_results)

    outcome.weights = _weights_from(
        outcome,
        candidates=candidates,
        run_id=run_id,
        block=block,
        workflow_id=workflow_id,
        burn_percentage=burn_percentage,
        burn_uid=burn_uid,
        incentive_mode=incentive_mode,
        reference_e2e=base.end_to_end,
    )
    return outcome


def _failed(candidate: Candidate, exc: Exception) -> CandidateEvaluation:
    """A candidate this host could not measure.

    Recorded rather than dropped so the run still reports how many candidates
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
    inputs: RunInputs,
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
                results.append(measure(candidate, inputs))
            except Exception as exc:  # one bad candidate must not stop the run
                log.warning("uid %s could not be measured: %s", candidate.uid, exc)
                results.append(_failed(candidate, exc))
        return results

    from concurrent.futures import ThreadPoolExecutor

    ordered: list[CandidateEvaluation | None] = [None] * len(candidates)

    def run(index: int) -> None:
        candidate = candidates[index]
        try:
            ordered[index] = measure(candidate, inputs)
        except Exception as exc:  # one bad candidate must not stop the run
            log.warning("uid %s could not be measured: %s", candidate.uid, exc)
            ordered[index] = _failed(candidate, exc)

    log.info("measuring %d candidates, %d at a time", len(candidates), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(len(candidates))))

    return [e for e in ordered if e is not None]


def _flag_peers(
    outcome: RunOutcome,
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
    outcome: RunOutcome,
    *,
    candidates: list[Candidate],
    run_id: int,
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
    reward a recipe that does not reconstruct; charging the run for it would
    let one broken submission tax everybody else.

    ``resolvable`` is this validator's own figure, not the network's, and is
    carried on each submission for audit and for :func:`tied`. Ranking itself no
    longer consults it: submissions are ordered by measured score alone, with no
    advantage to whoever committed first.
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
            run_id=run_id,
            block=block,
            workflow_id=workflow_id,
            burn_percentage=burn_percentage,
            burn_uid=burn_uid,
            reference_e2e=reference_e2e,
        )

    return graded_top3(
        ordered,
        run_id=run_id,
        block=block,
        workflow_id=workflow_id,
        burn_percentage=burn_percentage,
        burn_uid=burn_uid,
    )


def _graded_contribution_weights(
    outcome: RunOutcome,
    *,
    ranked: list[Submission],
    run_id: int,
    block: int,
    workflow_id: str,
    burn_percentage: float,
    burn_uid: int,
    reference_e2e: float,
) -> WeightVector:
    """Grade the measured field and let the graded split decide the payout.

    ``champion`` is deliberately ``None``. The throne is held by whoever last
    *dethroned* the incumbent, and that is a fact about the subnet's history
    rather than about this run — nothing in the ownerless loop carries a
    ChampionRecord between runs, so this validator has not seen anyone take
    it. Synthesising one from the run's own leader would crown a package for
    winning a field of one, skip the leaderless burn entirely, and pay the full
    champion share every run regardless of whether anything was dethroned.

    So the leaderless branch is the honest one, and it is also the arrangement
    the contract describes for it: half the run burns, the best measured
    package leads what remains on the champion's terms, and the rest of the
    graded field splits what is left of that. Because the function pops the
    leader out of the graded list *after* capping it, the miners sharing the
    remainder are exactly ranks two through ten.

    Grades come from the four terms the published contract already defines, over
    scores this run produced. A candidate graded zero does not appear at all,
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
        run_id=run_id,
        block=block,
        workflow_id=workflow_id,
        burn_percentage=burn_percentage,
        burn_uid=burn_uid,
    )
