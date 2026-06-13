"""Merge methods.

Every allowed method is a preset over one explicit three-stage pipeline:

1. **Sparsify** — optionally discard part of each adapter's update before it can
   interfere with the others. Either the smallest-magnitude entries (magnitude
   trimming) or a random subset with the survivors rescaled (drop-and-rescale).
2. **Elect signs** — optionally decide, per entry, which direction the merged
   update should move in, so adapters that disagree cannot cancel each other into
   noise.
3. **Aggregate** — sum the surviving contributions, or average only over the
   adapters that agree with the elected sign.

Naming the stages rather than hiding them inside seven separate functions makes
the presets comparable, and makes it obvious which pairs coincide: with no
sparsification and no sign election there is exactly one sensible combination,
so ``svd`` and ``cat_svd`` produce the same update and differ only in the
factorisation path they document.

All arithmetic happens in delta space, in float32, one projection at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from capability_subnet.common import constants as C
from capability_subnet.merge_engine.determinism import WORK_DTYPE, generator_for

SparsifyMode = Literal["none", "magnitude", "random_rescale"]
SignMode = Literal["none", "total", "frequency"]
AggregateMode = Literal["sum", "disjoint_mean"]


@dataclass(frozen=True, slots=True)
class MergePipeline:
    """The resolved stage configuration for one merge method."""

    sparsify: SparsifyMode
    sign_election: SignMode
    aggregate: AggregateMode
    factor_space: bool
    description: str


#: Each allowed method name resolved to its pipeline.
PIPELINES: dict[str, MergePipeline] = {
    C.MERGE_LINEAR: MergePipeline(
        sparsify="none",
        sign_election="none",
        aggregate="sum",
        factor_space=True,
        description=(
            "Weighted sum performed on the factors themselves. The cheapest path: "
            "no full update is ever materialised. Because the factors are summed "
            "rather than their products, the result carries cross terms between "
            "adapters, which is what distinguishes it from the delta-space sum."
        ),
    ),
    C.MERGE_SVD: MergePipeline(
        sparsify="none",
        sign_election="none",
        aggregate="sum",
        factor_space=False,
        description=(
            "Exact weighted sum of effective updates, then a rank reduction back "
            "to the declared output rank."
        ),
    ),
    C.MERGE_CAT_SVD: MergePipeline(
        sparsify="none",
        sign_election="none",
        aggregate="sum",
        factor_space=False,
        description=(
            "Concatenates the adapters' factorisations and reduces the result. "
            "Concatenating factors is algebraically the sum of the updates, so "
            "this produces the same update as the plain delta-space sum; the "
            "mandatory reduction back to the output rank is what makes it usable."
        ),
    ),
    C.MERGE_TIES_SVD: MergePipeline(
        sparsify="magnitude",
        sign_election="total",
        aggregate="disjoint_mean",
        factor_space=False,
        description=(
            "Trims each update to its largest-magnitude entries, elects one sign "
            "per entry, and averages only the adapters that agree with it."
        ),
    ),
    C.MERGE_DARE_TIES_SVD: MergePipeline(
        sparsify="random_rescale",
        sign_election="total",
        aggregate="disjoint_mean",
        factor_space=False,
        description=(
            "Drops a random subset of each update and rescales the survivors so "
            "the expected update is preserved, then elects signs and averages "
            "the agreeing adapters."
        ),
    ),
    C.MERGE_DARE_LINEAR_SVD: MergePipeline(
        sparsify="random_rescale",
        sign_election="none",
        aggregate="sum",
        factor_space=False,
        description="Random drop-and-rescale followed by a plain weighted sum.",
    ),
    C.MERGE_MAGNITUDE_PRUNE_SVD: MergePipeline(
        sparsify="magnitude",
        sign_election="none",
        aggregate="sum",
        factor_space=False,
        description="Magnitude trimming followed by a plain weighted sum.",
    ),
}


def pipeline_for(method: str) -> MergePipeline:
    try:
        return PIPELINES[method]
    except KeyError:
        raise ValueError(
            f"unknown merge method {method!r}; allowed: {sorted(PIPELINES)}"
        ) from None


# ---------------------------------------------------------------------------
# Stage 1 — sparsify
# ---------------------------------------------------------------------------


def magnitude_trim(delta: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the largest-magnitude entries, zero the rest.

    The cut is made at a *threshold* rather than by selecting exactly ``k``
    indices. Index selection has to break ties somehow, and different backends
    break them differently; a threshold comparison keeps every tied entry and is
    identical on every machine. When ties straddle the cut this retains slightly
    more than the requested fraction, which is the right trade: reproducibility
    beats hitting the density target exactly.
    """
    if density >= 1.0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)

    magnitudes = delta.abs()
    total = magnitudes.numel()
    keep = max(1, int(round(density * total)))
    if keep >= total:
        return delta

    flat = magnitudes.reshape(-1)
    # kthvalue returns the k-th *smallest*; the k-th largest is at total - keep + 1.
    threshold = torch.kthvalue(flat, total - keep + 1).values
    return torch.where(magnitudes >= threshold, delta, torch.zeros_like(delta))


def random_drop_rescale(
    delta: torch.Tensor,
    density: float,
    *,
    seed_parts: tuple[object, ...],
) -> torch.Tensor:
    """Drop entries at random and rescale the survivors.

    Keeping a fraction ``d`` and dividing the survivors by ``d`` leaves the
    expected update unchanged while removing most of the entries that could
    interfere with another adapter.

    The generator is keyed on the recipe seed together with the identity of this
    specific tensor, so the mask does not depend on how many tensors were
    processed first.
    """
    if density >= 1.0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)

    generator = generator_for(*seed_parts)
    keep_mask = torch.rand(delta.shape, generator=generator, dtype=WORK_DTYPE) < density
    return torch.where(keep_mask, delta / density, torch.zeros_like(delta))


def sparsify(
    delta: torch.Tensor,
    mode: SparsifyMode,
    density: float | None,
    *,
    seed_parts: tuple[object, ...],
) -> torch.Tensor:
    if mode == "none":
        return delta
    if density is None:
        raise ValueError(f"sparsify mode {mode!r} requires a density")
    if mode == "magnitude":
        return magnitude_trim(delta, density)
    if mode == "random_rescale":
        return random_drop_rescale(delta, density, seed_parts=seed_parts)
    raise ValueError(f"unknown sparsify mode {mode!r}")


# ---------------------------------------------------------------------------
# Stage 2 — sign election
# ---------------------------------------------------------------------------


def elect_signs(stacked: torch.Tensor, mode: SignMode) -> torch.Tensor:
    """Choose one direction per entry across the stacked adapter updates.

    Args:
        stacked: shape ``(num_adapters, out_features, in_features)``.
        mode: ``total`` weighs each adapter by how much it wants to move;
            ``frequency`` gives every adapter one vote regardless of magnitude.

    Returns:
        A tensor of ``+1``/``-1`` with the shape of a single update. Entries where
        the vote is exactly tied resolve to ``+1``, which is arbitrary but fixed.
    """
    if mode == "none":
        raise ValueError("elect_signs called with mode='none'")
    if mode == "total":
        tally = stacked.sum(dim=0)
    elif mode == "frequency":
        tally = torch.sign(stacked).sum(dim=0)
    else:
        raise ValueError(f"unknown sign election mode {mode!r}")

    return torch.where(
        tally < 0,
        torch.tensor(-1.0, dtype=stacked.dtype),
        torch.tensor(1.0, dtype=stacked.dtype),
    )


# ---------------------------------------------------------------------------
# Stage 3 — aggregate
# ---------------------------------------------------------------------------


def aggregate_sum(stacked: torch.Tensor) -> torch.Tensor:
    """Plain sum over adapters. Coefficients are already folded in."""
    return stacked.sum(dim=0)


def aggregate_disjoint_mean(stacked: torch.Tensor, elected: torch.Tensor) -> torch.Tensor:
    """Average only over the adapters that agree with the elected sign.

    Averaging rather than summing is what keeps the merged magnitude comparable
    to a single adapter's: two adapters that both want to move an entry the same
    way should reinforce each other's confidence, not double the step size.

    Entries where no adapter agrees average to zero rather than dividing by zero.
    """
    agrees = torch.sign(stacked) == elected.unsqueeze(0)
    contributions = torch.where(agrees, stacked, torch.zeros_like(stacked))
    agreeing_count = agrees.to(stacked.dtype).sum(dim=0)
    return contributions.sum(dim=0) / agreeing_count.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Composed merge
# ---------------------------------------------------------------------------


def merge_deltas(
    deltas: list[torch.Tensor],
    coefficients: list[float],
    pipeline: MergePipeline,
    *,
    density: float | None,
    sign_method: str | None,
    seed: int,
    site_name: str,
    adapter_ids: list[str],
) -> torch.Tensor:
    """Run the full pipeline for one projection.

    Args:
        deltas: per-adapter effective updates, all the same shape.
        coefficients: per-adapter coefficient, resolved for this layer group.
        pipeline: the resolved stage configuration.
        density: fraction retained by the sparsify stage, or ``None``.
        sign_method: sign election strategy, or ``None``.
        seed: the recipe's random seed.
        site_name: identifies the projection; part of the derived per-tensor seed.
        adapter_ids: identifies each adapter; part of the derived per-tensor seed.

    Returns:
        The merged update for this projection, float32.
    """
    if len(deltas) != len(coefficients) or len(deltas) != len(adapter_ids):
        raise ValueError("deltas, coefficients and adapter_ids must be the same length")
    if not deltas:
        raise ValueError("nothing to merge")

    # Sparsify each adapter independently, before the coefficient is applied.
    # Applying the coefficient first would let a large coefficient protect
    # entries that magnitude trimming should have discarded, which would make the
    # density parameter mean different things at different coefficients.
    processed: list[torch.Tensor] = []
    for delta, coefficient, adapter_id in zip(deltas, coefficients, adapter_ids, strict=True):
        trimmed = sparsify(
            delta,
            pipeline.sparsify,
            density,
            seed_parts=(seed, adapter_id, site_name),
        )
        processed.append(trimmed * coefficient)

    stacked = torch.stack(processed, dim=0)
    del processed

    if pipeline.aggregate == "disjoint_mean":
        if pipeline.sign_election == "none":
            raise ValueError("disjoint mean aggregation requires sign election")
        elected = elect_signs(stacked, sign_method or pipeline.sign_election)  # type: ignore[arg-type]
        return aggregate_disjoint_mean(stacked, elected)

    return aggregate_sum(stacked)


def describe(method: str) -> str:
    """One-line description used in the published workflow contract."""
    return pipeline_for(method).description
