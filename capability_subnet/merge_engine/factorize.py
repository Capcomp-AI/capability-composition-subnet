"""Factorising a merged update back into a LoRA pair.

Merging happens in delta space, but what gets deployed is a low-rank adapter.
The bridge is a truncated decomposition: keep the ``output_rank`` strongest
directions of the merged update and discard the rest.

That truncation is the compression knob a miner actually tunes. A smaller rank
means a smaller artifact and less memory at serving time, paid for with
reconstruction error. Whether that trade is worth it is exactly the question the
workflow score answers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from capability_subnet.common import constants as C
from capability_subnet.merge_engine.determinism import (
    WORK_DTYPE,
    canonicalize_svd_signs,
    generator_for,
)


@dataclass(frozen=True, slots=True)
class Factorization:
    """A rank-``r`` factorisation with the diagnostics needed to report on it."""

    lora_a: torch.Tensor  # (rank, in_features)
    lora_b: torch.Tensor  # (out_features, rank)
    effective_rank: int
    retained_energy: float
    clamped_components: int

    @property
    def rank(self) -> int:
        return self.lora_a.shape[0]

    def reconstruct(self) -> torch.Tensor:
        """The update this pair applies. Used by tests and error reporting."""
        return self.lora_b @ self.lora_a


def _clamp_singular_values(
    singular_values: torch.Tensor, quantile: float
) -> tuple[torch.Tensor, int]:
    """Cap the largest singular values at a quantile of the retained spectrum.

    A merged update is often dominated by two or three directions inherited from
    whichever adapter had the largest coefficient. Clamping limits how much any
    single direction can dominate, which in practice trades a little peak
    capability for better balance across stages.
    """
    if quantile >= 1.0 or singular_values.numel() == 0:
        return singular_values, 0

    threshold = torch.quantile(singular_values, quantile)
    clamped = torch.minimum(singular_values, threshold)
    changed = int((clamped < singular_values).sum().item())
    return clamped, changed


def _randomized_svd(
    work: torch.Tensor, rank: int, *, seed_parts: tuple[object, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The top ``rank`` singular triplets, by randomised range finding.

    A merged update is a few thousand rows by a few thousand columns and the
    artifact keeps sixty-four components of it. Computing the whole spectrum to
    discard 98% of it is where the merge time went: measured on one card, a full
    decomposition of the seven projections in a layer takes 15.1s and the
    truncated one takes 0.15s — 99x, and the merge is 99.7% decomposition.

    Halko-Martinsson-Tropp: sketch the range with a random projection, refine it
    with a few power iterations, then decompose the small matrix that results.
    The error is in the tail that gets discarded anyway; measured against the
    exact top-64 subspace it is ~1e-4 relative, which is an order of magnitude
    below what bfloat16 can represent — the artifact is written in bfloat16, so
    the approximation is smaller than the format's own rounding.

    Deterministic, which is the part that matters here rather than the speed.
    The sketch is drawn from a CPU generator keyed on the recipe seed *and the
    identity of the tensor being factorised*, exactly as every other stochastic
    step in this engine is, so it cannot depend on iteration order, sharding or
    which device the merge ran on. ``torch.svd_lowrank`` draws from the global
    RNG instead and is not used for that reason.
    """
    rows, cols = work.shape
    sketch_width = min(min(rows, cols), rank + C.SVD_OVERSAMPLE)

    generator = generator_for(*seed_parts, rows, cols, sketch_width)
    omega = torch.randn(
        cols, sketch_width, generator=generator, dtype=WORK_DTYPE, device="cpu"
    ).to(work.device)

    sample = work @ omega
    basis, _ = torch.linalg.qr(sample)
    for _ in range(C.SVD_POWER_ITERATIONS):
        # Re-orthonormalising between iterations; without it the sketch collapses
        # onto the leading direction in float32 and the smaller components of the
        # retained rank come back as noise.
        basis, _ = torch.linalg.qr(work.T @ basis)
        basis, _ = torch.linalg.qr(work @ basis)

    projected = basis.T @ work
    u_small, s, vh = torch.linalg.svd(projected, full_matrices=False)
    return basis @ u_small, s, vh


def factorize(
    delta: torch.Tensor,
    output_rank: int,
    *,
    clamp_quantile: float = 1.0,
    seed_parts: tuple[object, ...] = (),
) -> Factorization:
    """Decompose ``delta`` into a LoRA pair of exactly ``output_rank``.

    The emitted adapter is written with ``alpha == rank`` so its own scaling
    factor is 1. That means ``lora_B @ lora_A`` *is* the update, with no hidden
    multiplier — which is what lets the artifact be compared, hashed and served
    without carrying the recipe alongside it.

    When the update's natural rank is smaller than ``output_rank`` (a narrow
    projection, or a merge that cancelled almost everything), the remaining
    components are written as exact zeros rather than noise, so the artifact keeps
    the shape the deployment target expects.
    """
    if delta.ndim != 2:
        raise ValueError(f"expected a 2-D update, got shape {tuple(delta.shape)}")
    if output_rank <= 0:
        raise ValueError(f"output_rank must be positive, got {output_rank}")

    work = delta.to(WORK_DTYPE)
    out_features, in_features = work.shape

    # Measured over the *whole* spectrum, before truncation. Taking it after
    # would compare the kept components against themselves and report perfect
    # retention for every rank, which is exactly the number a miner tuning
    # `output_rank` must not be given.
    #
    # From the Frobenius norm rather than by summing the singular values, which
    # is the same quantity — the sum of squared singular values *is* the squared
    # Frobenius norm — and does not require computing a spectrum the truncated
    # path never forms.
    total_energy = float((work * work).sum().item())

    if output_rank + C.SVD_OVERSAMPLE < min(out_features, in_features):
        u, s, vh = _randomized_svd(work, output_rank, seed_parts=seed_parts)
    else:
        # The rank asked for is most of the matrix; sketching it would cost as
        # much as decomposing it and would be less accurate.
        u, s, vh = torch.linalg.svd(work, full_matrices=False)

    available = int(s.shape[0])
    keep = min(output_rank, available)

    u = u[:, :keep]
    s = s[:keep]
    vh = vh[:keep, :]

    u, vh = canonicalize_svd_signs(u, vh)

    s, clamped_components = _clamp_singular_values(s, clamp_quantile)
    retained_energy = float((s * s).sum().item())

    # Split the singular values evenly between the two factors so neither ends up
    # with a wildly different dynamic range — important because the artifact is
    # stored in bfloat16, which has only eight mantissa bits.
    root = torch.sqrt(s.clamp(min=0.0))
    lora_b = u * root.unsqueeze(0)
    lora_a = vh * root.unsqueeze(1)

    effective_rank = int((s > 0).sum().item())

    if keep < output_rank:
        pad = output_rank - keep
        lora_b = torch.cat([lora_b, torch.zeros((out_features, pad), dtype=WORK_DTYPE)], dim=1)
        lora_a = torch.cat([lora_a, torch.zeros((pad, in_features), dtype=WORK_DTYPE)], dim=0)

    energy_ratio = 1.0 if total_energy <= 0.0 else retained_energy / total_energy

    return Factorization(
        lora_a=lora_a,
        lora_b=lora_b,
        effective_rank=effective_rank,
        retained_energy=energy_ratio,
        clamped_components=clamped_components,
    )


def factorize_product(
    lora_b: torch.Tensor,
    lora_a: torch.Tensor,
    output_rank: int,
    *,
    clamp_quantile: float = 1.0,
    scale: float = 1.0,
) -> Factorization:
    """Decompose ``scale · lora_b @ lora_a`` without ever forming the product.

    A dense decomposition of the materialised update is the obvious
    implementation and the wrong one whenever the update is known to be low
    rank. ``lora_b @ lora_a`` has rank at most ``k`` — the shared inner
    dimension — so all but ``k`` of its singular values are exactly zero, and a
    full decomposition spends its entire cost computing them. On the real base
    model that is the difference between two minutes and two milliseconds per
    projection, at ``12288 × 4096``.

    The identity used here is standard: thin-QR each factor, decompose the tiny
    ``k × k`` core, and rotate the result back.

        B = Q_b R_b,  Aᵀ = Q_a R_a          (thin QR, k columns each)
        B A = Q_b (R_b R_aᵀ) Q_aᵀ           (an exact rewrite, nothing dropped)
        R_b R_aᵀ = U_c S V_cᵀ               (a k × k decomposition)
        ⇒ U = Q_b U_c,  Vᵀ = V_cᵀ Q_aᵀ

    ``Q_b`` and ``Q_a`` have orthonormal columns, so the singular values of the
    core *are* the singular values of the product — this is exact, not an
    approximation, and it agrees with the dense path to floating-point rounding.

    Args:
        lora_b: ``(out_features, k)``.
        lora_a: ``(k, in_features)``.
        output_rank: rank of the emitted pair.
        clamp_quantile: as in :func:`factorize`.
        scale: folded into the singular values, so an upstream ``alpha / r``
            never has to materialise a scaled copy of either factor.
    """
    if lora_b.ndim != 2 or lora_a.ndim != 2:
        raise ValueError(
            f"expected two 2-D factors, got {tuple(lora_b.shape)} and {tuple(lora_a.shape)}"
        )
    if lora_b.shape[1] != lora_a.shape[0]:
        raise ValueError(
            f"inner dimensions disagree: lora_b is {tuple(lora_b.shape)}, "
            f"lora_a is {tuple(lora_a.shape)}"
        )
    if output_rank <= 0:
        raise ValueError(f"output_rank must be positive, got {output_rank}")

    b_work = lora_b.to(WORK_DTYPE)
    a_work = lora_a.to(WORK_DTYPE)
    out_features = b_work.shape[0]
    in_features = a_work.shape[1]

    q_b, r_b = torch.linalg.qr(b_work, mode="reduced")
    q_a, r_a = torch.linalg.qr(a_work.transpose(0, 1), mode="reduced")

    core = (r_b @ r_a.transpose(0, 1)) * scale
    u_core, s, vh_core = torch.linalg.svd(core, full_matrices=False)

    # As in `factorize`: the whole spectrum, before truncation. Here it is the
    # number that tells an operator whether normalising a rank-128 public
    # adapter down to the canonical rank actually threw anything away.
    total_energy = float((s * s).sum().item())

    keep = min(output_rank, int(s.shape[0]))
    u = q_b @ u_core[:, :keep]
    vh = vh_core[:keep, :] @ q_a.transpose(0, 1)
    s = s[:keep]

    u, vh = canonicalize_svd_signs(u, vh)

    s, clamped_components = _clamp_singular_values(s, clamp_quantile)
    retained_energy = float((s * s).sum().item())

    root = torch.sqrt(s.clamp(min=0.0))
    lora_b_out = u * root.unsqueeze(0)
    lora_a_out = vh * root.unsqueeze(1)

    effective_rank = int((s > 0).sum().item())

    if keep < output_rank:
        pad = output_rank - keep
        lora_b_out = torch.cat(
            [lora_b_out, torch.zeros((out_features, pad), dtype=WORK_DTYPE)], dim=1
        )
        lora_a_out = torch.cat(
            [lora_a_out, torch.zeros((pad, in_features), dtype=WORK_DTYPE)], dim=0
        )

    energy_ratio = 1.0 if total_energy <= 0.0 else retained_energy / total_energy

    return Factorization(
        lora_a=lora_a_out,
        lora_b=lora_b_out,
        effective_rank=effective_rank,
        retained_energy=energy_ratio,
        clamped_components=clamped_components,
    )


def pad_to_rank(
    lora_a: torch.Tensor, lora_b: torch.Tensor, output_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad a factor pair up to ``output_rank``.

    Used by the factor-space path, which produces a pair at the source rank and
    only needs padding when a miner asked for a larger output rank than the pool
    provides. Padding with zeros is exact: it adds nothing to the update.
    """
    current = lora_a.shape[0]
    if current == output_rank:
        return lora_a, lora_b
    if current > output_rank:
        raise ValueError(
            f"cannot pad rank {current} down to {output_rank}; use a decomposition instead"
        )
    pad = output_rank - current
    padded_a = torch.cat([lora_a, torch.zeros((pad, lora_a.shape[1]), dtype=lora_a.dtype)], dim=0)
    padded_b = torch.cat([lora_b, torch.zeros((lora_b.shape[0], pad), dtype=lora_b.dtype)], dim=1)
    return padded_a, padded_b
