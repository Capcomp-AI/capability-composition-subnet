"""A validator measuring candidates for itself.

This is the same reconstruction, sandbox and scorer a miner runs locally and an
operator ran centrally — pointed at the seeds a window actually decided on, and
driven by the validator that is about to pay for the result.

Two things make that possible without an operator. The draw comes from
:func:`~capability_subnet.scoring.sampler.draw_window_open`, so it is a pure
function of a block hash and every validator derives it independently. The work
comes from :mod:`~capability_subnet.validator.assignment`, so a validator
measures a shared core plus a slice of its own rather than the whole window.

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
from capability_subnet.common.schemas import CandidateScores, InstanceResult, Recipe
from capability_subnet.registry.snapshot import PoolSnapshot, load_snapshot
from capability_subnet.sandbox.model_client import ModelClient
from capability_subnet.sandbox.orchestrator import SandboxConfig, run_batch
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
    error: str = ""

    @property
    def usable(self) -> bool:
        return not self.error


def evaluate_candidate(
    recipe: Recipe,
    client: ModelClient,
    *,
    assignment: Assignment,
    ood_seeds: tuple[int, ...] | list[int] = (),
    pool_dir: str | Path,
    artifact_dir: str | Path,
    candidate_id: str = "",
    snapshot: PoolSnapshot | None = None,
    workflow: WorkflowModule | None = None,
    sandbox_config: SandboxConfig | None = None,
    probe_seed: int | None = None,
    base_probe: ProbeOutcome | None = None,
    device: str = "cpu",
    serve: Callable[[str], AbstractContextManager[ModelClient]] | None = None,
) -> CandidateEvaluation:
    """Reconstruct a candidate and score it on this validator's assignment.

    Args:
        client: an endpoint already serving the reconstructed package. Standing
            that up is the validator's job, exactly as it is the miner's — this
            function does not assume a serving stack, so a validator may use
            vLLM, SGLang or anything else that speaks the client protocol.
        base_probe: the base model on the same probe. Without it retention reads
            as 1.0, which flatters every candidate equally and therefore changes
            the ranking less than it looks, but it is still wrong.
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
        ood_results = (
            [o.result for o in run_batch(ood, client, config=config, runner=flow.run_instance)]
            if ood
            else []
        )

        per_instance = {
            seed: bool(getattr(result, "end_to_end_success", False))
            for seed, result in zip(seeds, hidden_results, strict=False)
        }

        # A default SandboxConfig leaves this unset, and the scorer wants a number.
        # Which number matters: the efficiency term is 1 - peak/limit, so passing
        # 0.0 for "nobody measured this" awards the *maximum* VRAM credit — 0.667
        # against the 0.170 a real 23.8 GiB measurement earns. That is a bonus for
        # a measurement that was never taken, and it lands on every candidate this
        # validator scores, so its numbers stop being comparable with a validator
        # that did measure.
        #
        # Unmeasured therefore reads as the worst admissible case, which is what
        # the engine already does: "an unmeasured package scores no efficiency
        # credit here". The value is reported unchanged so a reader can see that
        # nothing was measured rather than infer it from a suspiciously round score.
        measured_vram = config.peak_vram_gb
        peak_vram = C.MAX_PEAK_VRAM_GB if measured_vram is None else float(measured_vram)
        if measured_vram is None:
            log.warning(
                "peak VRAM was not measured for %s; scoring it as the %.0f GB limit so it "
                "earns no efficiency credit it did not demonstrate",
                candidate_id or recipe.digest()[:19],
                C.MAX_PEAK_VRAM_GB,
            )

        resources = measure_resources(
            hidden_results, artifact_bytes=artifact_bytes, peak_vram_gb=peak_vram
        )
        probe = run_probe(client, build_probe(probe_seed if probe_seed is not None else 0))
        retention = relative_retention(probe, base_probe) if base_probe is not None else 1.0

        scores = aggregate_scores(
            hidden_results,
            ood_results,
            flow.critical_axes,
            retention=retention,
            efficiency=EfficiencyInputs(
                artifact_bytes=artifact_bytes,
                peak_vram_gb=peak_vram,
                reference_seconds=resources.mean_workflow_seconds,
            ),
        )

        return CandidateEvaluation(
            candidate_id=candidate_id,
            recipe_sha256=recipe.digest(),
            artifact_sha256=artifact_sha256,
            artifact_bytes=artifact_bytes,
            scores=scores,
            per_instance=per_instance,
            hidden_results=hidden_results,
            ood_results=ood_results,
        )


def core_results(evaluation: CandidateEvaluation, assignment: Assignment) -> dict[int, bool]:
    """Just the shared-core outcomes, which are what peers can compare.

    The tail is this validator's alone; another validator has no measurement of
    those seeds and their absence is not disagreement.
    """
    core = set(assignment.core)
    return {seed: ok for seed, ok in evaluation.per_instance.items() if seed in core}
