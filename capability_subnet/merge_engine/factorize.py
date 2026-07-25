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

from capability_subnet.merge_engine.determinism import WORK_DTYPE, canonicalize_svd_signs


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


def factorize(
    delta: torch.Tensor,
    output_rank: int,
    *,
    clamp_quantile: float = 1.0,
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

    u, s, vh = torch.linalg.svd(work, full_matrices=False)

    available = int(s.shape[0])
    keep = min(output_rank, available)

    u = u[:, :keep]
    s = s[:keep]
    vh = vh[:keep, :]

    u, vh = canonicalize_svd_signs(u, vh)

    total_energy = float((s * s).sum().item())
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
