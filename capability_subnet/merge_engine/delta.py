"""Effective updates.

A LoRA adapter stores two factors. What it *does* to the base model is the
effective update

    ΔW = (α / r) · B · A

The ``α / r`` factor is exactly why raw factors must never be compared directly:
two adapters can hold numerically similar ``B`` and ``A`` matrices and still
apply updates that differ by an order of magnitude. Every merge in this engine
therefore works in delta space, where a coefficient of 1.2 means the same thing
regardless of how the adapter was trained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from capability_subnet.common import constants as C
from capability_subnet.merge_engine.determinism import WORK_DTYPE
from capability_subnet.merge_engine.loader import AdapterSource

#: Matches the layer index in a canonical tensor key.
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass(frozen=True, slots=True)
class TensorSite:
    """One adapted projection in one decoder layer.

    A site is the unit of work: every merge method combines adapters
    independently at each site, so the whole reconstruction is a loop over sites
    with bounded memory.
    """

    layer: int
    module: str
    key_a: str
    key_b: str

    @property
    def name(self) -> str:
        return f"layers.{self.layer}.{self.module}"

    def layer_group(self, groups: dict[str, tuple[int, int]] | None = None) -> str:
        return C.layer_group_of(self.layer, groups)


def parse_layer_index(key: str) -> int:
    """Extract the decoder layer index from a tensor key."""
    match = _LAYER_RE.search(key)
    if match is None:
        raise ValueError(f"tensor key {key!r} does not encode a layer index")
    return int(match.group(1))


def enumerate_sites(manifest, modules: tuple[str, ...] = C.CANONICAL_TARGET_MODULES) -> list[TensorSite]:
    """Every site a fully populated adapter covers, in deterministic order.

    Ordering is ``(layer, module)`` with modules in the canonical tuple order,
    not alphabetical: it matches the forward pass and makes profiling output
    readable. It is fixed, which is all correctness requires.
    """
    sites: list[TensorSite] = []
    for layer in range(manifest.num_hidden_layers):
        for module in modules:
            sites.append(
                TensorSite(
                    layer=layer,
                    module=module,
                    key_a=manifest.tensor_key(layer, module, "lora_A"),
                    key_b=manifest.tensor_key(layer, module, "lora_B"),
                )
            )
    return sites


def effective_delta(
    source: AdapterSource,
    adapter_id: str,
    site: TensorSite,
    *,
    scaling: float,
) -> torch.Tensor:
    """Materialise ``ΔW = scaling · B · A`` for one adapter at one site.

    Args:
        scaling: the adapter's ``α / r``. Passed in rather than read here so the
            caller can apply the registry's certified value instead of trusting
            whatever the adapter file claims.

    Returns:
        A float32 tensor of shape ``(out_features, in_features)``.
    """
    lora_a = source.get_tensor(adapter_id, site.key_a).to(WORK_DTYPE)
    lora_b = source.get_tensor(adapter_id, site.key_b).to(WORK_DTYPE)

    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError(
            f"{adapter_id} at {site.name}: expected 2-D factors, "
            f"got {tuple(lora_a.shape)} and {tuple(lora_b.shape)}"
        )
    if lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError(
            f"{adapter_id} at {site.name}: rank mismatch between factors "
            f"({lora_a.shape[0]} vs {lora_b.shape[1]})"
        )

    return (lora_b @ lora_a) * scaling


def scaled_factors(
    source: AdapterSource,
    adapter_id: str,
    site: TensorSite,
    *,
    scaling: float,
    coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return factors with the coefficient and ``α / r`` folded into ``A``.

    Used by the factor-space path, which never materialises the full update.
    Folding everything into ``A`` rather than splitting across both factors
    keeps the operation exactly representable and avoids a square root.
    """
    lora_a = source.get_tensor(adapter_id, site.key_a).to(WORK_DTYPE)
    lora_b = source.get_tensor(adapter_id, site.key_b).to(WORK_DTYPE)
    return lora_a * (scaling * coefficient), lora_b


def delta_norm(delta: torch.Tensor) -> float:
    """Frobenius norm, used for reporting and per-adapter contribution analysis."""
    return float(torch.linalg.matrix_norm(delta, ord="fro").item())


def low_rank_delta_norm(lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float = 1.0) -> float:
    """Frobenius norm of ``scaling · B · A`` without materialising the product.

    Using the identity ``‖BA‖²_F = tr((BᵀB)(AAᵀ))`` turns a
    ``12288 × 4096`` intermediate into two ``64 × 64`` ones. That matters because
    per-adapter contribution norms are recorded for every site of every
    evaluation and feed the compatibility history; computing them the obvious way
    would roughly double the cost of the cheap merge path.
    """
    a = lora_a.to(WORK_DTYPE)
    b = lora_b.to(WORK_DTYPE)
    gram_b = b.transpose(0, 1) @ b
    gram_a = a @ a.transpose(0, 1)
    squared = float(torch.sum(gram_b * gram_a.transpose(0, 1)).item())
    return abs(scaling) * (max(squared, 0.0) ** 0.5)
