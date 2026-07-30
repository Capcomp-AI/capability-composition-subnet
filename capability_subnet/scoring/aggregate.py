"""Turning sample rows into scores.

Six components, weighted into one qualified score. Quality carries 85% of the
weight and efficiency 15%, which encodes the judgement that a cheap package that
does not finish the job is worth less than an expensive one that does.

Every component is computed only from rows that scored — rows carrying a harness
error are dropped first, everywhere, so infrastructure trouble can lower the
sample count but never the score.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CandidateScores, InstanceResult, ResourceMeasurements


@dataclass(frozen=True, slots=True)
class EfficiencyInputs:
    """Measurements the efficiency components are derived from."""

    artifact_bytes: int = 0
    peak_vram_gb: float = 0.0
    #: Reference latency the candidate is compared against, in seconds. The
    #: incumbent champion's median, so latency is scored relative to what the
    #: network already achieves rather than against an arbitrary constant.
    reference_seconds: float = 0.0


def valid_rows(results: list[InstanceResult]) -> list[InstanceResult]:
    """Rows that may take part in scoring."""
    return [row for row in results if row.is_valid_sample]


def end_to_end_completion(results: list[InstanceResult]) -> float:
    """Fraction of instances completed correctly from start to finish."""
    rows = valid_rows(results)
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.end_to_end_success) / len(rows)


def per_stage_means(results: list[InstanceResult], stages: tuple[str, ...]) -> dict[str, float]:
    """Mean score on each capability axis."""
    rows = valid_rows(results)
    if not rows:
        return {stage: 0.0 for stage in stages}
    return {stage: sum(row.stage_score(stage) for row in rows) / len(rows) for stage in stages}


def stage_balance(results: list[InstanceResult], stages: tuple[str, ...]) -> float:
    """Geometric mean of the per-stage means.

    A geometric mean is the right shape here because it punishes imbalance: a
    package scoring 1.0 on six stages and 0.1 on the seventh lands far below one
    scoring 0.8 everywhere, even though their arithmetic means are close. That is
    the intended incentive — the workflow needs every stage, so a package that
    abandoned one has not solved it.

    A small floor keeps a single zero from collapsing the term entirely and
    destroying the ranking information in the remaining stages.
    """
    means = per_stage_means(results, stages)
    if not means:
        return 0.0
    log_sum = sum(math.log(max(C.STAGE_BALANCE_EPSILON, value)) for value in means.values())
    return math.exp(log_sum / len(means))


def latency_efficiency(candidate_seconds: float, reference_seconds: float) -> float:
    """How the candidate's workflow latency compares with the reference."""
    if candidate_seconds <= 0.0:
        return 0.0
    if reference_seconds <= 0.0:
        return 1.0
    return min(1.0, reference_seconds / candidate_seconds)


def token_efficiency(results: list[InstanceResult]) -> float:
    """Output tokens spent per *completed* instance, against a fixed budget.

    Per completed instance rather than per attempted one, because dividing by
    attempts rewards failing early: a package that gives up after one turn spends
    almost nothing and would score best on a naive mean. Charging its tokens
    against the instances it actually finished makes cheap failure expensive,
    which is the correct direction — an unfinished workflow has no value to
    divide the cost into.

    A package that completes nothing scores zero here, not one.
    """
    rows = valid_rows(results)
    completed = [row for row in rows if row.end_to_end_success]
    if not completed:
        return 0.0

    spent = sum(row.output_tokens for row in rows)
    per_completion = spent / len(completed)
    if per_completion <= 0.0:
        return 1.0
    return min(1.0, C.REFERENCE_OUTPUT_TOKENS / per_completion)


def artifact_efficiency(artifact_bytes: int, peak_vram_gb: float) -> float:
    """Deployment cost, as an equal blend of artifact size and peak memory.

    Both terms are expressed as headroom below the hard gate, so a package that
    sits at the limit scores zero on this component while still passing the gate.
    """
    size_term = 1.0 - min(1.0, artifact_bytes / C.MAX_ARTIFACT_BYTES)
    vram_term = 1.0 - min(1.0, peak_vram_gb / C.MAX_PEAK_VRAM_GB)
    return 0.5 * size_term + 0.5 * vram_term


def measure_resources(
    results: list[InstanceResult], *, artifact_bytes: int, peak_vram_gb: float
) -> ResourceMeasurements:
    """Aggregate the per-instance measurements."""
    rows = valid_rows(results)
    durations = sorted(row.wall_seconds for row in rows)

    return ResourceMeasurements(
        adapter_mb=artifact_bytes / (1024 * 1024),
        peak_vram_gb=peak_vram_gb,
        p95_workflow_seconds=percentile(durations, 0.95),
        mean_workflow_seconds=statistics.fmean(durations) if durations else 0.0,
        input_tokens=sum(row.input_tokens for row in rows),
        output_tokens=sum(row.output_tokens for row in rows),
        model_turns=sum(row.turns_used for row in rows),
    )


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Nearest-rank rather than interpolated: the latency gate is a promise about
    an observed run, and interpolating invents a duration that never happened.
    """
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def aggregate_scores(
    hidden_results: list[InstanceResult],
    ood_results: list[InstanceResult],
    stages: tuple[str, ...],
    *,
    retention: float,
    efficiency: EfficiencyInputs,
) -> CandidateScores:
    """Compute every component and the qualified score.

    Args:
        hidden_results: rows from the canonical hidden instances.
        ood_results: rows from the out-of-distribution instances.
        stages: the workflow's capability axes.
        retention: probe score relative to the base model's, from
            :func:`retention.relative_retention`. Passed in rather than derived
            here because it is measured on a different set of questions than the
            workflow, which is the whole point of it.
        efficiency: measured deployment cost.
    """
    rows = valid_rows(hidden_results)

    completion = end_to_end_completion(hidden_results)
    balance = stage_balance(hidden_results, stages)
    ood = end_to_end_completion(ood_results)

    durations = sorted(row.wall_seconds for row in rows)
    latency = latency_efficiency(percentile(durations, 0.5), efficiency.reference_seconds)
    artifact = artifact_efficiency(efficiency.artifact_bytes, efficiency.peak_vram_gb)
    tokens = token_efficiency(hidden_results)

    qualified = (
        C.WEIGHT_END_TO_END * completion
        + C.WEIGHT_STAGE_BALANCE * balance
        + C.WEIGHT_OOD * ood
        + C.WEIGHT_RETENTION * retention
        + C.WEIGHT_LATENCY * latency
        + C.WEIGHT_TOKEN_EFFICIENCY * tokens
        + C.WEIGHT_ARTIFACT_EFFICIENCY * artifact
    )

    return CandidateScores(
        end_to_end=completion,
        stage_balance=balance,
        ood=ood,
        retention=retention,
        latency=latency,
        token_efficiency=tokens,
        artifact_efficiency=artifact,
        qualified_score=min(1.0, qualified),
        per_stage_means=per_stage_means(hidden_results, stages),
        valid_samples=len(rows),
        total_samples=len(hidden_results),
    )
