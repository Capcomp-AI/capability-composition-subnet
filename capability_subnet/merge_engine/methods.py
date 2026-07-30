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

from collections.abc import Callable

import torch

from capability_subnet.common import constants as C
from capability_subnet.common.merge_methods import (
    PIPELINES,
    MergePipeline,
    SignMode,
    SparsifyMode,
)
from capability_subnet.merge_engine.determinism import WORK_DTYPE, generator_for

#: Merge names that describe the same package, folded to one spelling.
METHOD_ALIASES: dict[str, str] = {C.MERGE_CAT_SVD: C.MERGE_SVD}


def canonical_method(method: str) -> str:
    """The canonical spelling of ``method``.

    Applied before a recipe is hashed so that two recipes describing the same
    merge share a digest. Without it the anti-copy check sees them as distinct
    submissions that happen to build identical bytes, and terminates whichever
    arrived second for copying.
    """
    return METHOD_ALIASES.get(method, method)


def pipeline_for(method: str) -> MergePipeline:
    try:
        return PIPELINES[method]
    except KeyError:
        raise ValueError(f"unknown merge method {method!r}; allowed: {sorted(PIPELINES)}") from None


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

    The mask is always drawn on the CPU and then moved to wherever the delta
    lives. That is not an oversight to optimise away: CUDA generators produce
    different streams across driver and architecture versions, so drawing the
    mask on the GPU would make a recipe's artifact depend on which card the
    worker was assigned. Drawing it on the CPU and transferring it keeps the mask
    identical on every machine while leaving the arithmetic on the accelerator.
    """
    if density >= 1.0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)

    generator = generator_for(*seed_parts)
    keep_mask = torch.rand(delta.shape, generator=generator, dtype=WORK_DTYPE) < density
    return torch.where(keep_mask.to(delta.device), delta / density, torch.zeros_like(delta))


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
    deltas: Callable[[int], torch.Tensor],
    coefficients: list[float],
    pipeline: MergePipeline,
    *,
    density: float | None,
    sign_method: str | None,
    seed: int,
    site_name: str,
    adapter_ids: list[str],
) -> torch.Tensor:
    """Run the full pipeline for one projection, one adapter at a time.

    Args:
        deltas: called with an adapter's index, returns that adapter's effective
            update. A callable rather than a list because holding every update
            at once is what this function exists to avoid: at the pinned base
            model's widest projection each one is 192 MB in float32, and a recipe
            may select twelve. Materialising them together needs 2.3 GB for one
            projection of one layer, and the intermediates the aggregation builds
            on top of that took a 24 GB card out of memory — so a *legal* recipe
            could not be built at all, and the failure looked like the miner's.
        coefficients: per-adapter coefficient, resolved for this layer group.
        pipeline: the resolved stage configuration.
        density: fraction retained by the sparsify stage, or ``None``.
        sign_method: sign election strategy, or ``None``.
        seed: the recipe's random seed.
        site_name: identifies the projection; part of the derived per-tensor seed.
        adapter_ids: identifies each adapter; part of the derived per-tensor seed.

    Returns:
        The merged update for this projection, float32.

    Peak memory is a small constant number of updates rather than one per
    selected adapter, so reconstruction cost no longer depends on how many
    adapters a miner chose.
    """
    if len(coefficients) != len(adapter_ids):
        raise ValueError("coefficients and adapter_ids must be the same length")
    if not adapter_ids:
        raise ValueError("nothing to merge")

    def contribution(index: int) -> torch.Tensor:
        """One adapter's sparsified, weighted update.

        Sparsify happens before the coefficient is applied. Applying the
        coefficient first would let a large coefficient protect entries that
        magnitude trimming should have discarded, which would make the density
        parameter mean different things at different coefficients.

        Recomputed rather than cached between passes: every step is deterministic
        given the seed and the tensor's identity, so the second pass sees exactly
        what the first did, and recomputing a trim is far cheaper than holding
        every update in memory to avoid it.
        """
        trimmed = sparsify(
            deltas(index),
            pipeline.sparsify,
            density,
            seed_parts=(seed, adapter_ids[index], site_name),
        )
        return trimmed * coefficients[index]

    count = len(adapter_ids)

    if pipeline.aggregate != "disjoint_mean":
        total: torch.Tensor | None = None
        for index in range(count):
            piece = contribution(index)
            total = piece if total is None else total.add_(piece)
            del piece
        assert total is not None  # count > 0 checked above
        return total

    if pipeline.sign_election == "none":
        raise ValueError("disjoint mean aggregation requires sign election")

    mode = sign_method or pipeline.sign_election

    # First pass: the sign tally. `total` weighs each adapter by how much it
    # wants to move, `frequency` gives each one vote.
    tally: torch.Tensor | None = None
    for index in range(count):
        piece = contribution(index)
        term = piece if mode == "total" else torch.sign(piece)
        tally = term.clone() if tally is None else tally.add_(term)
        del piece, term
    assert tally is not None

    elected = torch.where(
        tally < 0,
        torch.tensor(-1.0, dtype=tally.dtype, device=tally.device),
        torch.tensor(1.0, dtype=tally.dtype, device=tally.device),
    )
    del tally

    # Second pass: average only over the adapters that agree with the elected
    # sign. Averaging rather than summing keeps the merged magnitude comparable
    # to a single adapter's — two adapters that both want to move an entry the
    # same way should reinforce each other's confidence, not double the step.
    accumulated: torch.Tensor | None = None
    agreeing: torch.Tensor | None = None
    for index in range(count):
        piece = contribution(index)
        agrees = torch.sign(piece) == elected
        contributionwise = torch.where(agrees, piece, torch.zeros_like(piece))
        counted = agrees.to(piece.dtype)
        accumulated = (
            contributionwise if accumulated is None else accumulated.add_(contributionwise)
        )
        agreeing = counted if agreeing is None else agreeing.add_(counted)
        del piece, agrees, contributionwise, counted
    assert accumulated is not None and agreeing is not None

    # Entries where nothing agreed average to zero rather than dividing by zero.
    return accumulated.div_(agreeing.clamp_(min=1.0))


def describe(method: str) -> str:
    """One-line description used in the published workflow contract."""
    return pipeline_for(method).description
