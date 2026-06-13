"""Canonical artifact serialisation.

Two workers that reconstruct the same recipe must produce the same *bytes*, not
merely equivalent tensors. Anything that can vary between two correct writers is
pinned here: key order, metadata content, dtype, memory layout, and the config
file written beside the weights.

The artifact digest is taken over the weights file alone. The config is derived
entirely from the recipe and the pinned base, so including it would add nothing
but a second thing to keep in sync.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from capability_subnet.common import constants as C
from capability_subnet.common.hashing import sha256_bytes, sha256_file
from capability_subnet.merge_engine.determinism import OUTPUT_DTYPE

WEIGHTS_FILENAME = "adapter_model.safetensors"
CONFIG_FILENAME = "adapter_config.json"


@dataclass(frozen=True, slots=True)
class WrittenArtifact:
    """Where an artifact landed and what it hashes to."""

    directory: Path
    weights_path: Path
    config_path: Path
    artifact_sha256: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def build_adapter_config(
    *,
    base_model_repo: str,
    base_revision: str,
    output_rank: int,
    adapter_name: str,
    target_modules: tuple[str, ...] = C.CANONICAL_TARGET_MODULES,
) -> dict:
    """The config written beside the merged weights.

    ``lora_alpha`` is set equal to the rank so the adapter's own scaling factor
    is exactly 1. The merged update is already fully baked into the factors; a
    scaling factor other than 1 would silently rescale it at load time and make
    the artifact mean something different from what was measured.
    """
    return {
        "peft_type": C.CANONICAL_PEFT_TYPE,
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_repo,
        "base_model_revision": base_revision,
        "r": output_rank,
        "lora_alpha": output_rank,
        "lora_dropout": C.CANONICAL_LORA_DROPOUT,
        "bias": C.CANONICAL_BIAS,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": sorted(target_modules),
        "modules_to_save": None,
        "adapter_name": adapter_name,
    }


def canonical_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Narrow to the output dtype in a layout-independent way.

    ``contiguous`` matters: a transposed or sliced view serialises the same
    numbers through a different memory layout, and safetensors records the
    layout. Two writers that disagree on it produce different bytes for
    identical tensors.
    """
    return tensor.detach().to(OUTPUT_DTYPE).contiguous().cpu()


def write_artifact(
    tensors: dict[str, torch.Tensor],
    directory: str | Path,
    *,
    base_model_repo: str,
    base_revision: str,
    output_rank: int,
    adapter_name: str,
    target_modules: tuple[str, ...] = C.CANONICAL_TARGET_MODULES,
) -> WrittenArtifact:
    """Serialise a reconstructed adapter.

    Args:
        tensors: canonical tensor keys mapped to float32 tensors.

    Returns:
        The artifact location and the digest of the weights file.
    """
    from safetensors.torch import save_file

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    ordered = {key: canonical_tensor(tensors[key]) for key in sorted(tensors)}

    weights_path = target / WEIGHTS_FILENAME
    # No metadata: safetensors serialises the metadata dictionary into the
    # header, so any host-specific entry (a timestamp, a hostname, a library
    # version) would change the digest without changing a single weight.
    save_file(ordered, str(weights_path), metadata=None)

    config = build_adapter_config(
        base_model_repo=base_model_repo,
        base_revision=base_revision,
        output_rank=output_rank,
        adapter_name=adapter_name,
        target_modules=target_modules,
    )
    config_path = target / CONFIG_FILENAME
    config_path.write_text(
        json.dumps(config, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return WrittenArtifact(
        directory=target,
        weights_path=weights_path,
        config_path=config_path,
        artifact_sha256=sha256_file(weights_path),
        size_bytes=weights_path.stat().st_size,
    )


def artifact_digest_of_tensors(tensors: dict[str, torch.Tensor]) -> str:
    """Digest a tensor set without touching the filesystem.

    Serialises to an in-memory buffer using the same canonical ordering and
    dtype as :func:`write_artifact`, so the two agree by construction. Used by
    the cache to answer "have we already built this?" before choosing where to
    put it.
    """
    from safetensors.torch import save

    ordered = {key: canonical_tensor(tensors[key]) for key in sorted(tensors)}
    return sha256_bytes(save(ordered, metadata=None))
