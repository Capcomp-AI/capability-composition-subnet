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
from capability_subnet.merge_engine.factorize import factorize, factorize_product, pad_to_rank
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
    #: Which of the three site paths ran: factor-space, exact low-rank, or the
    #: materialised delta path. Published so a report says how it was built.
    merge_path: str = "delta_space"
    merge_device: str = "cpu"
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
        problems.append(f"recipe selects adapters that are not in the certified pool: {unknown}")

    # Present but not yet measured. Reported separately because it is a
    # different problem with a different fix: the adapter exists and is safe to
    # load, nobody has characterised it, and it will become selectable when
    # somebody does.
    unmeasured = snapshot.registry.unselectable_ids(recipe.selected_adapters)
    if unmeasured:
        problems.append(
            f"recipe selects adapters that are in the pool but not yet certified for "
            f"selection: {unmeasured}"
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
        recipe.compression.output_rank < source_rank or recipe.compression.svd_clamp_quantile < 1.0
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
        stats.record_contribution(group, adapter_id, low_rank_delta_norm(lora_a, lora_b))
        accumulated_a = lora_a if accumulated_a is None else accumulated_a + lora_a
        accumulated_b = lora_b if accumulated_b is None else accumulated_b + lora_b

    if accumulated_a is None or accumulated_b is None:
        raise ReconstructionError("no adapters selected")

    return pad_to_rank(accumulated_a, accumulated_b, recipe.compression.output_rank)


def _merge_site_low_rank(
    recipe: Recipe,
    snapshot: PoolSnapshot,
    source: AdapterSource,
    site: TensorSite,
    adapters: list[str],
    group: str,
    stats: ReconstructionStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact path for pipelines that never sparsify.

    With no sparsification and no sign election, the merged update is just a
    weighted sum of the adapters' updates:

        Σ cᵢ · sᵢ · Bᵢ Aᵢ  =  [c₁s₁B₁ … cₙsₙBₙ] · [A₁ … Aₙ]ᵀ

    — a single product of an ``out × nk`` and an ``nk × in`` matrix, whose rank
    is at most ``nk`` (twelve adapters at rank 64 is 768, against a 12288-wide
    projection). Concatenating the factors and decomposing *that* is exact and
    costs a thin QR instead of a dense decomposition of a materialised update.

    The materialised path remains for everything that trims or elects signs,
    because both operate entrywise and genuinely destroy the low-rank structure.
    """
    scaled_a: list[torch.Tensor] = []
    scaled_b: list[torch.Tensor] = []

    for adapter_id in adapters:
        entry = snapshot.registry.get(adapter_id)
        coefficient = recipe.effective_weight(adapter_id, group)
        lora_a = source.get_tensor(adapter_id, site.key_a).to(WORK_DTYPE).to(device)
        lora_b = source.get_tensor(adapter_id, site.key_b).to(WORK_DTYPE).to(device)

        stats.record_contribution(
            group, adapter_id, low_rank_delta_norm(lora_a, lora_b, entry.scaling * coefficient)
        )
        # The whole coefficient goes on one side so the product is unchanged and
        # neither factor's dynamic range is skewed by a large weight.
        scaled_b.append(lora_b * (entry.scaling * coefficient))
        scaled_a.append(lora_a)

    concatenated_b = torch.cat(scaled_b, dim=1)
    concatenated_a = torch.cat(scaled_a, dim=0)
    del scaled_a, scaled_b

    factorization = factorize_product(
        concatenated_b,
        concatenated_a,
        recipe.compression.output_rank,
        clamp_quantile=recipe.compression.svd_clamp_quantile,
    )

    stats.mean_retained_energy += factorization.retained_energy
    stats.min_retained_energy = min(stats.min_retained_energy, factorization.retained_energy)
    stats.mean_effective_rank += factorization.effective_rank
    stats.clamped_components += factorization.clamped_components

    return factorization.lora_a.cpu(), factorization.lora_b.cpu()


def _merge_site_delta_space(
    recipe: Recipe,
    snapshot: PoolSnapshot,
    source: AdapterSource,
    site: TensorSite,
    adapters: list[str],
    group: str,
    pipeline: methods.MergePipeline,
    stats: ReconstructionStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise each update, run the pipeline, factorise the result."""
    coefficients: list[float] = []

    for adapter_id in adapters:
        entry = snapshot.registry.get(adapter_id)
        lora_a = source.get_tensor(adapter_id, site.key_a).to(WORK_DTYPE).to(device)
        lora_b = source.get_tensor(adapter_id, site.key_b).to(WORK_DTYPE).to(device)
        coefficient = recipe.effective_weight(adapter_id, group)

        stats.record_contribution(
            group, adapter_id, low_rank_delta_norm(lora_a, lora_b, entry.scaling * coefficient)
        )
        coefficients.append(coefficient)
        del lora_a, lora_b

    def delta_for(index: int) -> torch.Tensor:
        """Materialise one adapter's update on demand.

        Read from the source each time rather than held in a list: the pipeline
        makes at most two passes, and a re-read from an open safetensors handle
        costs far less than keeping every selected adapter's full update
        resident. Twelve of them at the widest projection is 2.3 GB before the
        aggregation allocates anything.
        """
        adapter_id = adapters[index]
        entry = snapshot.registry.get(adapter_id)
        lora_a = source.get_tensor(adapter_id, site.key_a).to(WORK_DTYPE).to(device)
        lora_b = source.get_tensor(adapter_id, site.key_b).to(WORK_DTYPE).to(device)
        return (lora_b @ lora_a) * entry.scaling

    merged = methods.merge_deltas(
        delta_for,
        coefficients,
        pipeline,
        density=recipe.merge.density,
        sign_method=recipe.merge.majority_sign_method,
        seed=recipe.merge.random_seed,
        site_name=site.name,
        adapter_ids=adapters,
    )

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

    return factorization.lora_a.cpu(), factorization.lora_b.cpu()


def reconstruct(
    recipe: Recipe,
    snapshot: PoolSnapshot,
    source: AdapterSource,
    *,
    output_dir: str | Path | None = None,
    keep_tensors: bool = False,
    threads: int = 1,
    device: str = "cpu",
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
        device: where the merge arithmetic runs. ``cuda`` is roughly thirty times
            faster on the trimming methods, which materialise a full update per
            projection and must decompose it densely.

            This is consensus-relevant and deliberately explicit. A decomposition
            is only unique up to the sign convention this engine pins, but the
            *values* still depend on the LAPACK or cuSOLVER implementation
            underneath, so two workers agree byte-for-byte only when they run the
            same device class. The engine records the device in every published
            report for exactly that reason: an artifact digest is reproducible by
            anyone who matches the hardware it was built on, and a dispute that
            cannot be reproduced on matching hardware is a real finding rather
            than an ambiguity.

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
        raise AdapterUnavailableError(f"adapter weights are not available on this host: {missing}")

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

    torch_device = torch.device(device)
    # Falling back rather than failing: a miner checking a recipe on a laptop
    # should get an answer, not an error about hardware they were never asked to
    # have. The engine's own preflight is where a misconfigured device is caught.
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        log.warning("cuda requested but unavailable; merging on the CPU")
        torch_device = torch.device("cpu")

    # No sparsification and no sign election means the merged update stays a sum
    # of low-rank products, which can be decomposed exactly from the factors.
    use_low_rank = (
        not use_factor_space and pipeline.sparsify == "none" and pipeline.sign_election == "none"
    )

    with deterministic_context(seed=recipe.merge.random_seed, threads=threads):
        for site in enumerate_sites(manifest):
            group = site.layer_group(groups)
            try:
                if use_factor_space:
                    lora_a, lora_b = _merge_site_factor_space(
                        recipe, snapshot, source, site, adapters, group, stats
                    )
                elif use_low_rank:
                    lora_a, lora_b = _merge_site_low_rank(
                        recipe, snapshot, source, site, adapters, group, stats, torch_device
                    )
                else:
                    lora_a, lora_b = _merge_site_delta_space(
                        recipe,
                        snapshot,
                        source,
                        site,
                        adapters,
                        group,
                        pipeline,
                        stats,
                        torch_device,
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

    stats.merge_device = torch_device.type
    stats.merge_path = (
        "factor_space" if use_factor_space else ("low_rank" if use_low_rank else "delta_space")
    )
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
