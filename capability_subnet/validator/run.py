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
from capability_subnet.scoring.contribution import ContributionInputs, contribution_score
from capability_subnet.scoring.retention import ProbeOutcome
from capability_subnet.scoring.sampler import RunSample, draw_run_open
from capability_subnet.scoring.weight_vector import champion_ladder
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
    burn_share: float = C.BURN_SHARE,
    burn_uid: int = C.BURN_UID,
    champion_grade: float | None = None,
    workers: int = 1,
    peer_core_results: dict[str, dict[int, bool]] | None = None,
) -> RunOutcome:
    """Evaluate a run and produce this validator's own weight vector.

    Args:
        workers: how many candidates to measure at once. One per GPU: a
            candidate reserves almost the whole card while it is served, so
            concurrency above the device count would make the packages contend
            for memory and measure each other rather than themselves.
        champion_grade: the reigning champion's grade, or ``None`` when the
            throne is empty and this run fills it. A challenger is paid only by
            exceeding it by ``CHAMPION_DETHRONE_MARGIN``.
        measure_base: measures the base model on this run's draw. The base
            model is the only permanent reference, and this loop cannot produce
            a bar without it — every candidate's retention is scored against
            its probe and every candidate's improvement against its score.
        peer_core_results: other validators' core results for one shared
            candidate, when this validator has them. Used only to report
            inconsistency — a miner is not paid less because another validator
            looks wrong, since the miner did not do anything.
    """
    sample = draw_run_open(run_id, beacon=beacon, hidden_count=hidden_count, ood_count=ood_count)
    assignment = assign(sample.hidden_seeds, hotkey=hotkey, beacon=beacon)
    outcome = RunOutcome(run_id=run_id, beacon=beacon, sample=sample, assignment=assignment)

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
            log.warning(
                "%s could not be measured: %s", evaluation.candidate_id[:12], evaluation.error
            )
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
        burn_share=burn_share,
        burn_uid=burn_uid,
        reference_e2e=base.end_to_end,
        champion_grade=champion_grade,
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
    candidates: list[Candidate],
    run_id: int,
    block: int,
    workflow_id: str,
    burn_share: float,
    burn_uid: int,
    reference_e2e: float,
    champion_grade: float | None,
) -> WeightVector:
    """Grade what was measured and turn it into weights.

    An unmeasurable candidate earns nothing and burns nothing extra: it is
    simply absent, exactly as a miner who never submitted is absent. Paying it
    would reward a recipe that does not reconstruct; charging the run for it
    would let one broken submission tax everybody else.

    Args:
        champion_grade: the reigning champion's grade, or ``None`` when the
            throne is empty and this run fills it.
    """
    by_hotkey = {c.hotkey: c for c in candidates}

    graded: list[tuple[int, str, float]] = []
    for evaluation in outcome.usable:
        candidate = by_hotkey.get(evaluation.candidate_id)
        if candidate is None:
            continue
        grade = contribution_score(
            ContributionInputs(scores=evaluation.scores, reference_e2e=reference_e2e)
        )
        graded.append((candidate.uid, candidate.hotkey, grade))

    graded.sort(key=lambda row: -row[2])

    return champion_ladder(
        graded,
        run_id=run_id,
        block=block,
        champion_grade=champion_grade,
        burn_uid=burn_uid,
        burn_share=burn_share,
        workflow_id=workflow_id,
    )
