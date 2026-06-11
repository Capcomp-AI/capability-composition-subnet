"""Deterministic reconstruction.

This module turns a recipe into an adapter. It is the one component where two
independent workers must agree exactly: a disagreement here is a protocol
failure, not a scoring difference, and evaluation of the affected candidate stops
until it is resolved.

The procedure is fixed:

1. resolve the recipe against the frozen pool,
2. load adapters in sorted identifier order — never in the order the miner listed
   them, so the artifact cannot depend on how the recipe was written,
3. widen every tensor to float32,
4. run the chosen merge pipeline at each projection independently,
5. factorise back to the declared output rank,
6. narrow to bfloat16 and serialise with sorted keys,
7. hash the resulting bytes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import Recipe
from capability_subnet.merge_engine import methods
from capability_subnet.merge_engine.canonical_writer import (
    WrittenArtifact,
    artifact_digest_of_tensors,
    write_artifact,
)
from capability_subnet.merge_engine.delta import (
    TensorSite,
    enumerate_sites,
    low_rank_delta_norm,
    scaled_factors,
)
from capability_subnet.merge_engine.determinism import WORK_DTYPE, deterministic_context
from capability_subnet.merge_engine.factorize import factorize, pad_to_rank
from capability_subnet.merge_engine.loader import AdapterSource
from capability_subnet.registry.snapshot import PoolSnapshot

log = logging.getLogger(__name__)


class ReconstructionError(Exception):
    """Raised when a recipe cannot be reconstructed.

    This is a candidate-level failure — the recipe is rejected and scores zero.
    It is distinct from a worker disagreement, which is an engine-level failure.
    """


class AdapterUnavailableError(ReconstructionError):
    """Raised when the certified weights are not materialised on this host.

    Deliberately a distinct type. Everything else that stops a reconstruction is
    the miner's fault and scores zero; this one is the *operator's*, and scoring
    zero for it would spend a hotkey's single shot on a pool that had not
    finished downloading.
    """


@dataclass(slots=True)
class ReconstructionStats:
    """Diagnostics collected during reconstruction.

    These feed the compatibility history: which adapters actually contributed at
    which depth, and how much of the merged update survived compression.
    """

    sites: int = 0
    mean_retained_energy: float = 1.0
    min_retained_energy: float = 1.0
    mean_effective_rank: float = 0.0
    clamped_components: int = 0
    used_factor_space: bool = False
    seconds: float = 0.0
    #: Frobenius norm of each adapter's contribution, summed per layer group.
    contribution_by_group: dict[str, dict[str, float]] = field(default_factory=dict)

    def record_contribution(self, group: str, adapter_id: str, norm: float) -> None:
        self.contribution_by_group.setdefault(group, {})
        self.contribution_by_group[group][adapter_id] = (
            self.contribution_by_group[group].get(adapter_id, 0.0) + norm
        )


@dataclass(slots=True)
class ReconstructionResult:
    """A reconstructed candidate."""

    recipe_sha256: str
    artifact_sha256: str
    output_rank: int
    adapter_name: str
    stats: ReconstructionStats
    artifact: WrittenArtifact | None = None
    tensors: dict[str, torch.Tensor] | None = None

    @property
    def size_bytes(self) -> int:
        return self.artifact.size_bytes if self.artifact else 0

    @property
    def size_mb(self) -> float:
        return self.artifact.size_mb if self.artifact else 0.0


def validate_against_pool(recipe: Recipe, snapshot: PoolSnapshot) -> None:
    """Check that the recipe was built against exactly this frozen pool.

    Raises:
        ReconstructionError: with every problem found, not just the first — a
            miner fixing a recipe should not have to resubmit once per mistake.
    """
    problems = snapshot.validate_recipe_scope(recipe.base_revision, recipe.source_snapshot_sha256)

    unknown = snapshot.registry.unknown_ids(recipe.selected_adapters)
    if unknown:
        problems.append(
            f"recipe selects adapters that are not in the certified pool: {unknown}"
        )

    if recipe.workflow_id != snapshot.registry.workflow_id:
        problems.append(
            f"recipe targets workflow {recipe.workflow_id!r}, this pool serves "
            f"{snapshot.registry.workflow_id!r}"
        )

    if problems:
        raise ReconstructionError("; ".join(problems))


def _needs_reduction(recipe: Recipe, source_rank: int) -> bool:
    """Whether the cheap factor-space path can be used.

    It cannot when the miner asked for a rank below what the pool provides (the
    update genuinely has to be compressed) or asked for singular-value clamping
    (which is defined on the decomposition and has no factor-space equivalent).
    """
    return (
        recipe.compression.output_rank < source_rank
        or recipe.compression.svd_clamp_quantile < 1.0
    )


def _merge_site_factor_space(
    recipe: Recipe,
    snapshot: PoolSnapshot,
    source: AdapterSource,
    site: TensorSite,
    adapters: list[str],
    group: str,
    stats: ReconstructionStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum the factors directly, never materialising the full update."""
    accumulated_a: torch.Tensor | None = None
    accumulated_b: torch.Tensor | None = None

    for adapter_id in adapters:
        entry = snapshot.registry.get(adapter_id)
        coefficient = recipe.effective_weight(adapter_id, group)
        lora_a, lora_b = scaled_factors(
            source, adapter_id, site, scaling=entry.scaling, coefficient=coefficient
        )
        stats.record_contribution(
            group, adapter_id, low_rank_delta_norm(lora_a, lora_b)
        )
        accumulated_a = lora_a if accumulated_a is None else accumulated_a + lora_a
        accumulated_b = lora_b if accumulated_b is None else accumulated_b + lora_b

    if accumulated_a is None or accumulated_b is None:
        raise ReconstructionError("no adapters selected")

    return pad_to_rank(accumulated_a, accumulated_b, recipe.compression.output_rank)


def _merge_site_delta_space(
    recipe: Recipe,
    snapshot: PoolSnapshot,
    source: AdapterSource,
    site: TensorSite,
    adapters: list[str],
    group: str,
    pipeline: methods.MergePipeline,
    stats: ReconstructionStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise each update, run the pipeline, factorise the result."""
    deltas: list[torch.Tensor] = []
    coefficients: list[float] = []

    for adapter_id in adapters:
        entry = snapshot.registry.get(adapter_id)
        lora_a = source.get_tensor(adapter_id, site.key_a).to(WORK_DTYPE)
        lora_b = source.get_tensor(adapter_id, site.key_b).to(WORK_DTYPE)
        coefficient = recipe.effective_weight(adapter_id, group)

        stats.record_contribution(
            group, adapter_id, low_rank_delta_norm(lora_a, lora_b, entry.scaling * coefficient)
        )

        deltas.append((lora_b @ lora_a) * entry.scaling)
        coefficients.append(coefficient)

    merged = methods.merge_deltas(
        deltas,
        coefficients,
        pipeline,
        density=recipe.merge.density,
        sign_method=recipe.merge.majority_sign_method,
        seed=recipe.merge.random_seed,
        site_name=site.name,
        adapter_ids=adapters,
    )
    del deltas

    factorization = factorize(
        merged,
        recipe.compression.output_rank,
        clamp_quantile=recipe.compression.svd_clamp_quantile,
    )
    del merged

    stats.mean_retained_energy += factorization.retained_energy
    stats.min_retained_energy = min(stats.min_retained_energy, factorization.retained_energy)
    stats.mean_effective_rank += factorization.effective_rank
    stats.clamped_components += factorization.clamped_components

    return factorization.lora_a, factorization.lora_b


def reconstruct(
    recipe: Recipe,
    snapshot: PoolSnapshot,
    source: AdapterSource,
    *,
    output_dir: str | Path | None = None,
    keep_tensors: bool = False,
    threads: int = 1,
) -> ReconstructionResult:
    """Build the merged adapter a recipe describes.

    Args:
        recipe: the validated miner submission.
        snapshot: the frozen pool the recipe must match.
        source: where the certified adapter tensors come from.
        output_dir: where to write the artifact. ``None`` computes the digest
            without touching the filesystem, which is what the cache does when
            checking whether this artifact already exists.
        keep_tensors: retain the merged tensors on the result. Only useful for
            tests and for in-process serving; at 8B scale they are large.
        threads: torch thread count. Left at one for reproducible accumulation.

    Returns:
        The reconstruction, including the artifact digest.

    Raises:
        ReconstructionError: if the recipe does not match the pool or an adapter
            tensor is missing or malformed.
    """
    validate_against_pool(recipe, snapshot)

    manifest = snapshot.manifest
    pipeline = methods.pipeline_for(recipe.merge.combination_type)
    adapters = recipe.sorted_adapters()
    groups = manifest.layer_groups
    source_rank = snapshot.registry.canonical_rank

    missing = [adapter_id for adapter_id in adapters if not source.has_adapter(adapter_id)]
    if missing:
        raise AdapterUnavailableError(
            f"adapter weights are not available on this host: {missing}"
        )

    use_factor_space = pipeline.factor_space and not _needs_reduction(recipe, source_rank)
    stats = ReconstructionStats(used_factor_space=use_factor_space)
    if not use_factor_space:
        # These two are accumulators on the delta-space path and are averaged
        # over sites at the end. On the factor-space path nothing is discarded,
        # so their defaults already describe the outcome.
        stats.mean_retained_energy = 0.0
        stats.mean_effective_rank = 0.0

    started = time.monotonic()
    tensors: dict[str, torch.Tensor] = {}

    with deterministic_context(seed=recipe.merge.random_seed, threads=threads):
        for site in enumerate_sites(manifest):
            group = site.layer_group(groups)
            try:
                if use_factor_space:
                    lora_a, lora_b = _merge_site_factor_space(
                        recipe, snapshot, source, site, adapters, group, stats
                    )
                else:
                    lora_a, lora_b = _merge_site_delta_space(
                        recipe, snapshot, source, site, adapters, group, pipeline, stats
                    )
            except ReconstructionError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ReconstructionError(f"failed at {site.name}: {exc}") from exc

            tensors[site.key_a] = lora_a
            tensors[site.key_b] = lora_b
            stats.sites += 1

    if stats.sites == 0:
        raise ReconstructionError("the base manifest describes no adapted projections")

    if not use_factor_space:
        stats.mean_retained_energy /= stats.sites
        stats.mean_effective_rank /= stats.sites
    else:
        stats.mean_effective_rank = float(min(source_rank, recipe.compression.output_rank))

    stats.seconds = time.monotonic() - started

    if output_dir is None:
        artifact_sha256 = artifact_digest_of_tensors(tensors)
        written = None
    else:
        written = write_artifact(
            tensors,
            output_dir,
            base_model_repo=manifest.model_repo,
            base_revision=manifest.revision,
            output_rank=recipe.compression.output_rank,
            adapter_name=recipe.output.adapter_name,
            target_modules=C.CANONICAL_TARGET_MODULES,
        )
        artifact_sha256 = written.artifact_sha256

    log.info(
        "reconstructed %s in %.1fs: %d sites, rank %d, artifact %s",
        recipe.merge.combination_type,
        stats.seconds,
        stats.sites,
        recipe.compression.output_rank,
        artifact_sha256[:19],
    )

    return ReconstructionResult(
        recipe_sha256=recipe.digest(),
        artifact_sha256=artifact_sha256,
        output_rank=recipe.compression.output_rank,
        adapter_name=recipe.output.adapter_name,
        stats=stats,
        artifact=written,
        tensors=tensors if keep_tensors else None,
    )


def artifact_hashes_agree(results: list[ReconstructionResult]) -> tuple[bool, str]:
    """Cross-check several workers' reconstructions of the same recipe.

    A mismatch means two workers running what should be identical software
    produced different artifacts. No score is assigned to the affected candidate
    until it is resolved — scoring one of two disagreeing artifacts would mean
    paying for a result nobody can reproduce.
    """
    if not results:
        return False, "no reconstructions to compare"

    digests = {result.artifact_sha256 for result in results}
    if len(digests) == 1:
        return True, f"{len(results)} workers agree on {next(iter(digests))[:19]}"
    return False, (
        f"{len(results)} workers produced {len(digests)} distinct artifacts: "
        + ", ".join(sorted(digest[:19] for digest in digests))
    )
