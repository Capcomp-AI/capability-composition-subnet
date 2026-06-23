"""Evaluating one package on one window.

The pipeline every candidate and every reference goes through, in the same order,
on the same instances:

    build → serve → run the hidden set → run the out-of-distribution set →
    measure resources → apply the hard gates → aggregate the scores

References and candidates share this path exactly. That is not tidiness — it is
the reason a comparison between them means anything. If baselines were measured
by a different code path, "the challenger beat the strongest reference" would be
a statement about two harnesses rather than two packages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from capability_subnet.backend.executor.reconstruction import BuildOutcome, Reconstructor
from capability_subnet.backend.executor.serving import (
    ServingError,
    read_peak_vram_gb,
    reset_vram_peak,
)
from capability_subnet.backend.scorer import gates
from capability_subnet.backend.scorer.aggregate import (
    EfficiencyInputs,
    aggregate_scores,
    end_to_end_completion,
    measure_resources,
    valid_rows,
)
from capability_subnet.common import constants as C
from capability_subnet.common.schemas import (
    CandidateScores,
    GateVerdict,
    InstanceResult,
    Recipe,
    ResourceMeasurements,
)
from capability_subnet.merge_engine.engine import AdapterUnavailableError, ReconstructionError
from capability_subnet.sandbox.orchestrator import SandboxConfig, run_batch

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluablePackage:
    """Anything that can be served and measured."""

    candidate_id: str
    #: A merged package built from a recipe.
    recipe: Recipe | None = None
    #: A single certified adapter, served as-is.
    adapter_id: str | None = None
    #: The base model with nothing applied.
    is_base: bool = False
    hotkey: str | None = None
    uid: int | None = None

    def __post_init__(self) -> None:
        provided = sum((self.recipe is not None, self.adapter_id is not None, self.is_base))
        if provided != 1:
            raise ValueError(
                f"{self.candidate_id}: exactly one of recipe, adapter_id or is_base must be set"
            )


@dataclass(slots=True)
class EvaluationOutput:
    """Everything one evaluation produced."""

    candidate_id: str
    hidden_results: list[InstanceResult] = field(default_factory=list)
    ood_results: list[InstanceResult] = field(default_factory=list)
    scores: CandidateScores = field(default_factory=CandidateScores)
    resources: ResourceMeasurements = field(default_factory=ResourceMeasurements)
    gate_verdicts: list[GateVerdict] = field(default_factory=list)
    artifact_sha256: str | None = None
    artifact_bytes: int = 0
    build: BuildOutcome | None = None
    infrastructure_error: str | None = None

    @property
    def gates_passed(self) -> bool:
        """Whether the gates ran and all of them passed.

        Never True for an empty list — an evaluation that returned before the
        gates ran has not cleared them.
        """
        return gates.all_passed(self.gate_verdicts)

    @property
    def end_to_end(self) -> float:
        return self.scores.end_to_end

    @property
    def usable(self) -> bool:
        """Whether this evaluation may inform a decision.

        An infrastructure error makes an evaluation unusable without making the
        candidate a failure. The scheduler holds such a candidate for a later
        window rather than terminating it.
        """
        return self.infrastructure_error is None


class Evaluator:
    """Runs the evaluation pipeline for one deployment."""

    def __init__(
        self,
        *,
        reconstructor: Reconstructor,
        server,
        adapter_pool_dir: str | Path,
        sandbox_config: SandboxConfig | None = None,
        stages: tuple[str, ...] = (),
        gpu_index: int = 0,
        min_valid_samples: int = 20,
        require_vram_measurement: bool = False,
    ) -> None:
        self.reconstructor = reconstructor
        self.server = server
        self.adapter_pool_dir = Path(adapter_pool_dir)
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.stages = stages
        self.gpu_index = gpu_index
        self.min_valid_samples = min_valid_samples
        # Defaults to False so CPU-only development, where there is no GPU to
        # measure, is not blocked by a gate that cannot apply. Any deployment
        # with a GPU sets it True.
        self.require_vram_measurement = require_vram_measurement

    # -- build --------------------------------------------------------------

    def _resolve_artifact(
        self, package: EvaluablePackage
    ) -> tuple[Path | None, str | None, int, BuildOutcome | None]:
        """Produce the servable adapter for ``package``.

        Returns:
            ``(adapter_path, artifact_digest, size_bytes, build)``. The base model
            has no adapter, so every element but the size is ``None``.
        """
        if package.is_base:
            return None, None, 0, None

        if package.adapter_id is not None:
            directory = self.adapter_pool_dir / package.adapter_id
            weights = directory / "adapter_model.safetensors"
            if not weights.is_file():
                raise ServingError(
                    f"adapter {package.adapter_id} is not materialised at {directory}"
                )
            from capability_subnet.common.hashing import sha256_file

            return directory, sha256_file(weights), weights.stat().st_size, None

        assert package.recipe is not None  # narrowed by __post_init__
        build = self.reconstructor.build(package.recipe)
        return build.artifact_dir, build.artifact_sha256, build.size_bytes, build

    # -- run ----------------------------------------------------------------

    def evaluate(
        self,
        package: EvaluablePackage,
        hidden_instances: list,
        ood_instances: list,
        *,
        base_e2e: float = 0.0,
        reference_latency_seconds: float = 0.0,
        strongest_reference_id: str = "",
        strongest_reference_score: float = 0.0,
        end_to_end_margin: float = C.DEFAULT_END_TO_END_MARGIN,
        apply_comparison_gates: bool = True,
    ) -> EvaluationOutput:
        """Build, serve, run and score one package.

        Args:
            base_e2e: the base model's completion on the same instances, which
                the retention component is measured against.
            reference_latency_seconds: the incumbent's median instance duration.
            apply_comparison_gates: whether to apply the gates that compare
                against other packages. References are measured with these off —
                a baseline is not required to beat itself.
        """
        output = EvaluationOutput(candidate_id=package.candidate_id)

        try:
            adapter_path, digest, size_bytes, build = self._resolve_artifact(package)
        except (ServingError, AdapterUnavailableError) as exc:
            # The pool is not materialised, or the endpoint is unavailable.
            # Both are the operator's problem, and neither may spend a
            # candidate's single shot.
            output.infrastructure_error = str(exc)
            return output
        except ReconstructionError as exc:
            # The recipe itself cannot be built. That is a candidate failure and
            # scores zero — recorded as a gate verdict rather than an exception,
            # so it reaches the published report like every other rejection.
            output.gate_verdicts.append(
                gates.gate_reconstruction(False, f"reconstruction failed: {exc}")
            )
            return output

        output.artifact_sha256 = digest
        output.artifact_bytes = size_bytes
        output.build = build

        if build is not None and not build.workers_agreed:
            # Not a candidate failure. Evaluation pauses until the operator has
            # resolved the disagreement.
            output.infrastructure_error = (
                f"reconstruction workers disagreed: {build.agreement_detail}"
            )
            return output

        # Structural gates first: a package that cannot be deployed is not worth
        # a GPU-minute, and refusing it here costs nothing.
        structural = [gates.gate_artifact_size(size_bytes)]
        if build is not None:
            structural.append(
                gates.gate_reconstruction(build.workers_agreed, build.agreement_detail)
            )
        output.gate_verdicts.extend(structural)

        if not all(verdict.passed for verdict in structural):
            log.info(
                "%s failed a structural gate; skipping evaluation", package.candidate_id
            )
            return output

        reset_vram_peak(self.gpu_index)

        try:
            with self.server.serve(adapter_path) as handle:
                client = handle.client()
                config = SandboxConfig(
                    limits=self.sandbox_config.limits,
                    postgres_dsn=self.sandbox_config.postgres_dsn,
                    peak_vram_gb=handle.peak_vram_gb,
                )

                hidden = run_batch(hidden_instances, client, config=config)
                ood = run_batch(ood_instances, client, config=config) if ood_instances else []

                # Read the peak while the endpoint is still up: tearing it down
                # releases the allocation and the counter goes with it.
                #
                # Either reading may be unavailable. None propagates rather than
                # collapsing to zero, so the gate can tell "used no memory" from
                # "we do not know" — the difference between a candidate that fits
                # and one that was never measured.
                readings = [
                    value
                    for value in (handle.peak_vram_gb, read_peak_vram_gb(self.gpu_index))
                    if value is not None
                ]
                peak_vram = max(readings) if readings else None
        except ServingError as exc:
            output.infrastructure_error = str(exc)
            return output
        except Exception as exc:  # noqa: BLE001
            log.exception("evaluation of %s failed", package.candidate_id)
            output.infrastructure_error = f"evaluation failed: {exc}"
            return output

        output.hidden_results = [outcome.result for outcome in hidden]
        output.ood_results = [outcome.result for outcome in ood]

        output.resources = measure_resources(
            output.hidden_results,
            artifact_bytes=size_bytes,
            peak_vram_gb=peak_vram or 0.0,
        )

        candidate_e2e = end_to_end_completion(output.hidden_results)
        output.scores = aggregate_scores(
            output.hidden_results,
            output.ood_results,
            self.stages,
            base_e2e=base_e2e,
            efficiency=EfficiencyInputs(
                artifact_bytes=size_bytes,
                # An unmeasured package scores no efficiency credit here. The
                # gate above is what decides whether it is admissible at all;
                # this only refuses to award a bonus for a measurement nobody
                # took.
                peak_vram_gb=peak_vram if peak_vram is not None else C.MAX_PEAK_VRAM_GB,
                reference_seconds=reference_latency_seconds,
            ),
        )

        output.gate_verdicts.extend(
            self._behavioural_gates(
                output,
                peak_vram=peak_vram,
                candidate_e2e=candidate_e2e,
                strongest_reference_id=strongest_reference_id,
                strongest_reference_score=strongest_reference_score,
                end_to_end_margin=end_to_end_margin,
                apply_comparison_gates=apply_comparison_gates,
            )
        )

        return output

    def _behavioural_gates(
        self,
        output: EvaluationOutput,
        *,
        peak_vram: float | None,
        candidate_e2e: float,
        strongest_reference_id: str,
        strongest_reference_score: float,
        end_to_end_margin: float,
        apply_comparison_gates: bool,
    ) -> list[GateVerdict]:
        verdicts = [
            gates.gate_sample_sufficiency(output.hidden_results, self.min_valid_samples),
            gates.gate_peak_vram(
                peak_vram, require_measurement=self.require_vram_measurement
            ),
            gates.gate_latency(output.hidden_results),
            gates.gate_agent_limits(output.hidden_results),
            gates.gate_safety(output.hidden_results),
            gates.gate_stage_floors(
                output.scores, {stage: 1.0 for stage in self.stages}
            ),
        ]

        if apply_comparison_gates:
            verdicts.append(gates.gate_base_retention(output.scores.retention))
            verdicts.append(
                gates.gate_beats_strongest_reference(
                    candidate_e2e,
                    strongest_reference_id or "no reference",
                    strongest_reference_score,
                    end_to_end_margin,
                )
            )

        return verdicts


def summarise_evaluation(output: EvaluationOutput) -> str:
    """One log line describing an evaluation."""
    if not output.usable:
        return f"{output.candidate_id}: unusable ({output.infrastructure_error})"

    valid = len(valid_rows(output.hidden_results))
    failed = [verdict.name for verdict in output.gate_verdicts if not verdict.passed]
    state = "passed all gates" if not failed else f"failed {', '.join(failed)}"

    return (
        f"{output.candidate_id}: end-to-end {output.scores.end_to_end:.3f}, "
        f"qualified {output.scores.qualified_score:.3f}, {valid} valid samples, {state}"
    )
