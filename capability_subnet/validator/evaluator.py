"""A validator measuring candidates for itself.

This is the same reconstruction, sandbox and scorer a miner runs locally and an
operator ran centrally — pointed at the seeds a run actually decided on, and
driven by the validator that is about to pay for the result.

Two things make that possible without an operator. The draw comes from
:func:`~capability_subnet.scoring.sampler.draw_run_open`, so it is a pure
function of a block hash and every validator derives it independently. The work
comes from :mod:`~capability_subnet.validator.assignment`, so a validator
measures a shared core plus a slice of its own rather than the whole run.

What this deliberately does *not* do is require two validators to agree on
artifact bytes. Six of the seven merge methods run an SVD and an SVD is not
bitwise reproducible across devices — measured, not assumed. Every candidate is
therefore reconstructed locally by each validator, on its own hardware, and
compared on *outcomes* through :mod:`~capability_subnet.validator.agreement`.
The artifact digest is still recorded, because a validator whose digest matches
another's is stronger evidence than one that does not, but it is evidence rather
than a gate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import (
    CandidateScores,
    GateVerdict,
    InstanceResult,
    Recipe,
)
from capability_subnet.registry.snapshot import PoolSnapshot, load_snapshot
from capability_subnet.sandbox.model_client import ModelClient
from capability_subnet.sandbox.orchestrator import SandboxConfig, run_batch
from capability_subnet.scoring import gates
from capability_subnet.scoring.aggregate import (
    EfficiencyInputs,
    aggregate_scores,
    measure_resources,
)
from capability_subnet.scoring.retention import (
    ProbeOutcome,
    build_probe,
    relative_retention,
    run_probe,
)
from capability_subnet.validator.assignment import Assignment
from capability_subnet.workflows import WorkflowModule, get_workflow

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidateEvaluation:
    """One validator's measurement of one candidate."""

    candidate_id: str
    recipe_sha256: str
    artifact_sha256: str
    artifact_bytes: int
    scores: CandidateScores
    #: seed -> did this instance succeed. The unit of cross-validator
    #: comparison: same seeds, same question, independent hardware.
    per_instance: dict[int, bool] = field(default_factory=dict)
    hidden_results: list[InstanceResult] = field(default_factory=list)
    ood_results: list[InstanceResult] = field(default_factory=list)
    #: Every gate this candidate was put through. Empty means the gates never
    #: ran, which is not the same as passing them — see ``usable``.
    gate_verdicts: list[GateVerdict] = field(default_factory=list)
    #: instance_id -> what the package replied, what it spent, and whether the
    #: harness failed on that row. The scores above are a summary of these, and
    #: a summary cannot answer why a package scored what it did or show that a
    #: row was dropped for an infrastructure failure rather than a wrong answer.
    #: This is also what a disclosure publishes, so an auditor re-scores from
    #: the same bytes the validator scored.
    traces: dict[str, dict] = field(default_factory=dict)
    error: str = ""

    @property
    def usable(self) -> bool:
        """Whether this candidate may compete for weight.

        Was ``not self.error``, which asked only whether the host managed to
        finish — so a package that destroyed the base model's general ability,
        or answered a handful of instances, or never cleared the reference,
        ranked alongside one that did none of those things. The gates decide it
        now, and an empty verdict list is a failure rather than a pass: a
        candidate whose gates never ran has not cleared them.
        """
        return not self.error and gates.all_passed(self.gate_verdicts)

    @property
    def gate_failures(self) -> list[str]:
        """Why this candidate cannot compete, for the run log."""
        return [f"{v.name}: {v.detail}" for v in self.gate_verdicts if not v.passed]


def evaluate_candidate(
    recipe: Recipe,
    client: ModelClient,
    *,
    assignment: Assignment,
    ood_seeds: tuple[int, ...] | list[int],
    pool_dir: str | Path,
    artifact_dir: str | Path,
    candidate_id: str = "",
    snapshot: PoolSnapshot | None = None,
    workflow: WorkflowModule | None = None,
    sandbox_config: SandboxConfig | None = None,
    probe_seed: int,
    base_probe: ProbeOutcome,
    reference_e2e: float = 0.0,
    reference_id: str = "",
    end_to_end_margin: float = C.DEFAULT_END_TO_END_MARGIN,
    min_valid_samples: int = C.DEFAULT_MIN_AXIS_SAMPLES,
    device: str = "cpu",
    serve: Callable[[str], AbstractContextManager[ModelClient]] | None = None,
) -> CandidateEvaluation:
    """Reconstruct a candidate and score it on this validator's assignment.

    Args:
        client: an endpoint already serving the reconstructed package. Standing
            that up is the validator's job, exactly as it is the miner's — this
            function does not assume a serving stack, so a validator may use
            vLLM, SGLang or anything else that speaks the client protocol.
        ood_seeds: this run's out-of-distribution draw. Required: it used to
            default to empty, and an empty draw scores the OOD term zero for
            every candidate — a tenth of the qualified score, silently absent.
        base_probe: the base model on the same probe. Required: retention used
            to read as 1.0 without it, so a package that destroyed the base
            model's general ability passed the floor it exists to fail.
        reference_e2e: end-to-end completion of the strongest permanent
            reference. A challenger must clear it by ``end_to_end_margin``.
    """
    from capability_subnet.miner.local_eval import build_local_artifact

    pool = snapshot or load_snapshot()
    flow = workflow or get_workflow(pool.registry.workflow_id)
    config = sandbox_config or SandboxConfig()

    try:
        artifact_sha256, artifact_bytes, _ = build_local_artifact(
            recipe, pool_dir=pool_dir, output_dir=artifact_dir, snapshot=pool, device=device
        )
    except Exception as exc:  # reconstruction is the first thing that can fail
        log.warning("reconstruction failed for %s: %s", candidate_id or recipe.digest(), exc)
        return CandidateEvaluation(
            candidate_id=candidate_id,
            recipe_sha256=recipe.digest(),
            artifact_sha256="",
            artifact_bytes=0,
            scores=CandidateScores(),
            error=f"reconstruction failed: {exc}",
        )

    # The package that was just built is the package that gets measured. Without
    # this the scorer talks to whatever endpoint it was handed, which is the same
    # endpoint for every candidate — two different recipes then produce the same
    # numbers and the ranking has nothing to rank.
    #
    # The runtime lives exactly as long as the scoring below, and is stopped on
    # the way out whether or not that scoring succeeded.
    with ExitStack() as serving:
        if serve is not None:
            client = serving.enter_context(serve(str(artifact_dir)))

        seeds = assignment.seeds
        hidden = [flow.generate_instance(seed, split="hidden") for seed in seeds]
        ood = [flow.generate_instance(seed, split="ood") for seed in ood_seeds]

        hidden_outcomes = run_batch(hidden, client, config=config, runner=flow.run_instance)
        hidden_results = [o.result for o in hidden_outcomes]
        ood_outcomes = (
            run_batch(ood, client, config=config, runner=flow.run_instance) if ood else []
        )
        ood_results = [o.result for o in ood_outcomes]
        traces = {
            outcome.result.instance_id: outcome.trace.to_dict()
            for outcome in (*hidden_outcomes, *ood_outcomes)
            if getattr(outcome, "trace", None) is not None
        }

        per_instance = {
            seed: bool(getattr(result, "end_to_end_success", False))
            for seed, result in zip(seeds, hidden_results, strict=False)
        }

        measure_resources(hidden_results, artifact_bytes=artifact_bytes)
        probe = run_probe(client, build_probe(probe_seed))
        retention = relative_retention(probe, base_probe)

        scores = aggregate_scores(
            hidden_results,
            ood_results,
            flow.critical_axes,
            retention=retention,
            efficiency=EfficiencyInputs(
                artifact_bytes=artifact_bytes,
            ),
        )

        # The same gates the engine applies, in the same order, so a validator
        # and the operator's engine reach the same verdict about a package
        # rather than two defensible different ones.
        verdicts = [
            gates.gate_artifact_size(artifact_bytes),
            gates.gate_sample_sufficiency(hidden_results, min_valid_samples),
            gates.gate_agent_limits(hidden_results),
            gates.gate_safety(hidden_results),
            gates.gate_stage_floors(scores, flow.stage_floors, min_samples=min_valid_samples),
            gates.gate_base_retention(scores.retention),
            gates.gate_beats_strongest_reference(
                scores.end_to_end,
                reference_id or "no reference",
                reference_e2e,
                end_to_end_margin,
            ),
        ]

        return CandidateEvaluation(
            candidate_id=candidate_id,
            recipe_sha256=recipe.digest(),
            artifact_sha256=artifact_sha256,
            artifact_bytes=artifact_bytes,
            scores=scores,
            per_instance=per_instance,
            hidden_results=hidden_results,
            ood_results=ood_results,
            gate_verdicts=verdicts,
            traces=traces,
        )


def measure_base_model(
    client: ModelClient,
    *,
    assignment: Assignment,
    probe_seed: int,
    seeds: tuple[int, ...] | None = None,
    with_probe: bool = True,
    snapshot: PoolSnapshot | None = None,
    workflow: WorkflowModule | None = None,
    sandbox_config: SandboxConfig | None = None,
) -> tuple[list[InstanceResult], ProbeOutcome]:
    """Score the untouched base model on this run's own draw.

    The bar every candidate is held to, and the probe their retention is scored
    against. Measured here rather than assumed because a reference taken from a
    different run is not a paired comparison, and a reference left at zero —
    which is what the loop did before — turns "improvement over the reference"
    into "the candidate's score".

    Args:
        seeds: a subset of the assignment to measure, when the sweep is split
            across a fleet. Defaults to the whole assignment.
        with_probe: whether to ask the probe. One shard asks it; the rest do
            not, because forty extra questions per card measure the same forty
            answers.

    Returns:
        The instance results and the probe outcome. Raw results rather than a
        score, so a caller splitting the sweep aggregates once over everything
        instead of averaging averages.
    """
    pool = snapshot or load_snapshot()
    flow = workflow or get_workflow(pool.registry.workflow_id)
    config = sandbox_config or SandboxConfig()

    shard = seeds if seeds is not None else assignment.seeds
    hidden = [flow.generate_instance(seed, split="hidden") for seed in shard]
    outcomes = run_batch(hidden, client, config=config, runner=flow.run_instance)
    results = [o.result for o in outcomes]
    probe = run_probe(client, build_probe(probe_seed)) if with_probe else ProbeOutcome()

    return results, probe


def core_results(evaluation: CandidateEvaluation, assignment: Assignment) -> dict[int, bool]:
    """Just the shared-core outcomes, which are what peers can compare.

    The tail is this validator's alone; another validator has no measurement of
    those seeds and their absence is not disagreement.
    """
    core = set(assignment.core)
    return {seed: ok for seed, ok in evaluation.per_instance.items() if seed in core}
