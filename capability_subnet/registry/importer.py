"""Normalising a public LoRA into the certified pool.

Public adapters are trained at whatever rank, alpha and dropout their author
chose. The protocol needs one canonical shape — rank 64, alpha 64, the seven
projections, ``bias: none`` — because a single rank is what keeps linear
merging, TIES/DARE behaviour, output-size comparison and resource measurement
directly comparable across candidates.

Normalisation is a re-factorisation of the *same* update, not a retraining:

    delta = (alpha / r) · B · A          the effective update, upstream scaling folded in
    delta ≈ B' · A'                      re-factorised at rank 64 with alpha 64, so scaling is 1

For an upstream rank below 64 this is exact — the update's natural rank is
already lower than the target, so the truncation discards nothing and the
remaining components are written as exact zeros. Above 64 it is lossy, which is
why :func:`certification.check_conversion_recertified` refuses to admit a
converted adapter that has not been measured again afterwards.

Nothing here trusts the upstream repository. Only two files are ever fetched —
the config and the weights — so a repo carrying scripts, pickles or arbitrary
extra tensors contributes none of them, and the certification gates then run
against the file this module wrote rather than against anything upstream
published.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capability_subnet.common import constants as C
from capability_subnet.registry.base_model import BaseManifest

log = logging.getLogger(__name__)

#: The only files fetched from an upstream repository.
FETCHED_FILES: tuple[str, ...] = ("adapter_config.json", "adapter_model.safetensors")


class ImportError_(Exception):
    """Raised when a public adapter cannot be normalised into the pool."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """What the registry says an upstream adapter should be.

    Recorded in the registry rather than discovered at import time on purpose:
    an upstream repo that silently changes rank between two imports must be a
    loud failure, not a different artifact under the same adapter id.
    """

    adapter_id: str
    repo: str
    revision: str
    rank: int
    lora_alpha: int
    base_repo: str = C.BASE_MODEL_REPO


def _download(repo: str, revision: str, filename: str, target: Path) -> Path:
    """Fetch one file at a pinned revision."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        local_dir=str(target),
    )
    return Path(path)


def verify_source_config(config: dict[str, Any], spec: SourceSpec) -> None:
    """Refuse an upstream config that is not what the registry recorded.

    Every check here is a property the re-factorisation depends on. A DoRA
    adapter carries a magnitude vector the delta formula ignores; ``rslora``
    changes the scaling from ``alpha/r`` to ``alpha/sqrt(r)``; ``modules_to_save``
    means full-rank weights travel alongside the low-rank pair. Converting any
    of them with the plain formula would produce a *different model* under the
    adapter's name, so they are rejected rather than approximated.
    """
    problems: list[str] = []

    base = str(config.get("base_model_name_or_path") or "")
    if base != spec.base_repo:
        problems.append(
            f"base_model_name_or_path is {base!r}, expected {spec.base_repo!r}. "
            "An adapter trained on a different base (a quantised mirror, a "
            "continued-pretraining checkpoint) is not a delta on this pool's base model."
        )
    if config.get("peft_type") != C.CANONICAL_PEFT_TYPE:
        problems.append(f"peft_type is {config.get('peft_type')!r}, expected LORA")
    if int(config.get("r", -1)) != spec.rank:
        problems.append(f"r is {config.get('r')}, registry recorded {spec.rank}")
    if int(config.get("lora_alpha", -1)) != spec.lora_alpha:
        problems.append(
            f"lora_alpha is {config.get('lora_alpha')}, registry recorded {spec.lora_alpha}"
        )
    if config.get("bias") not in (None, "none"):
        problems.append(f"bias is {config.get('bias')!r}; only 'none' can be normalised")
    if config.get("use_dora"):
        problems.append(
            "use_dora is set; a DoRA magnitude vector cannot be folded into a LoRA pair"
        )
    if config.get("use_rslora"):
        problems.append("use_rslora is set; its scaling is alpha/sqrt(r), not alpha/r")
    if config.get("modules_to_save"):
        problems.append("modules_to_save is set; full-rank side weights cannot enter the pool")
    if config.get("auto_map"):
        problems.append("auto_map is set; the config would load remote code")

    declared = set(config.get("target_modules") or ())
    if declared != set(C.CANONICAL_TARGET_MODULES):
        problems.append(
            f"target_modules are {sorted(declared)}, expected {sorted(C.CANONICAL_TARGET_MODULES)}"
        )

    if problems:
        raise ImportError_(
            f"{spec.adapter_id}: upstream config is unusable — " + "; ".join(problems)
        )


def _source_key(template: str, layer: int, block: str, module: str, matrix: str) -> str:
    return f"{template.format(layer=layer, block=block, module=module)}.{matrix}.weight"


def normalise_tensors(
    source: dict,
    *,
    spec: SourceSpec,
    manifest: BaseManifest,
) -> tuple[dict, dict[str, float]]:
    """Re-factorise every projection to the canonical rank.

    Runs under :func:`deterministic_context`, without which the artifact this
    writes depends on the machine that wrote it. The decomposition is a
    float32 reduction, and reduction order in a multi-threaded sum is not fixed,
    so torch's default thread count — which follows the host's core count — chose
    the low bits of every tensor. Importing the same upstream adapter at the same
    pinned revision produced a different ``artifact_sha256`` on a 32-core host
    than on an 8-core one, reproducibly on each.

    That is the one thing this pipeline may not do. The digest written back into
    the registry is what every later reconstruction is checked against, and the
    engine cross-checks reconstruction digests between workers precisely to catch
    disagreement. Importing outside the controls the merge engine already defines
    meant the pool's own provenance was the least reproducible step in it.

    Returns:
        ``(tensors, stats)`` — the canonical tensor map keyed by the manifest's
        own template, and per-adapter conversion diagnostics. ``retained_energy``
        below 1.0 means the upstream rank exceeded the canonical rank and the
        conversion genuinely discarded something.
    """
    import torch

    from capability_subnet.merge_engine.determinism import (
        OUTPUT_DTYPE,
        WORK_DTYPE,
        deterministic_context,
    )
    from capability_subnet.merge_engine.factorize import factorize_product

    scaling = spec.lora_alpha / spec.rank
    tensors: dict[str, torch.Tensor] = {}
    energies: list[float] = []

    attention = set(manifest.raw["attention_modules"])
    template = manifest.layer_module_template

    # Entered once around the whole decomposition rather than per projection:
    # the context also seeds the global RNGs and restores thread settings on
    # exit, and doing that 252 times per adapter would be 252 resets for one
    # adapter's worth of arithmetic.
    with deterministic_context(0, threads=1):
        for layer in range(manifest.num_hidden_layers):
            for module in C.CANONICAL_TARGET_MODULES:
                block = "self_attn" if module in attention else "mlp"
                key_a = _source_key(template, layer, block, module, "lora_A")
                key_b = _source_key(template, layer, block, module, "lora_B")

                if key_a not in source or key_b not in source:
                    raise ImportError_(
                        f"{spec.adapter_id}: upstream weights are missing {key_a!r} or {key_b!r}. "
                        "The adapter does not cover every canonical projection."
                    )

                lora_a = source[key_a].to(WORK_DTYPE)
                lora_b = source[key_b].to(WORK_DTYPE)

                shape = manifest.shape_of(module)
                if lora_a.shape != (spec.rank, shape.in_features):
                    raise ImportError_(
                        f"{spec.adapter_id}: {key_a} has shape {tuple(lora_a.shape)}, "
                        f"expected {(spec.rank, shape.in_features)}"
                    )
                if lora_b.shape != (shape.out_features, spec.rank):
                    raise ImportError_(
                        f"{spec.adapter_id}: {key_b} has shape {tuple(lora_b.shape)}, "
                        f"expected {(shape.out_features, spec.rank)}"
                    )

                # The effective update is `scaling · lora_b @ lora_a`, but it is
                # never materialised: it has rank at most `spec.rank`, so the
                # factored path decomposes it exactly from the factors themselves.
                # Forming the product first would mean a dense decomposition of a
                # 12288 × 4096 matrix, 252 times per adapter.
                #
                # The emitted pair carries alpha == rank, so its own scaling is 1
                # and `lora_B @ lora_A` *is* this delta — no hidden multiplier
                # travels with the artifact.
                factored = factorize_product(
                    lora_b, lora_a, C.CANONICAL_RANK, clamp_quantile=1.0, scale=scaling
                )
                energies.append(factored.retained_energy)

                tensors[manifest.tensor_key(layer, module, "lora_A")] = factored.lora_a.to(
                    OUTPUT_DTYPE
                ).contiguous()
                tensors[manifest.tensor_key(layer, module, "lora_B")] = factored.lora_b.to(
                    OUTPUT_DTYPE
                ).contiguous()

    mean_energy = sum(energies) / len(energies) if energies else 0.0
    return tensors, {"mean_retained_energy": mean_energy, "sites": float(len(energies))}


def canonical_config(manifest: BaseManifest) -> dict[str, Any]:
    """The adapter config every pool member carries.

    ``base_model_revision`` is written explicitly because
    :func:`certification.check_base_revision` requires it: an adapter file that
    does not name the revision it applies to is one repin away from being
    silently wrong.
    """
    return {
        "peft_type": C.CANONICAL_PEFT_TYPE,
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": manifest.model_repo,
        "base_model_revision": manifest.revision,
        "r": C.CANONICAL_RANK,
        "lora_alpha": C.CANONICAL_LORA_ALPHA,
        "lora_dropout": C.CANONICAL_LORA_DROPOUT,
        "bias": C.CANONICAL_BIAS,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
        "target_modules": sorted(C.CANONICAL_TARGET_MODULES),
    }


def import_adapter(
    spec: SourceSpec,
    *,
    manifest: BaseManifest,
    out_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Fetch, verify, normalise and write one pool member.

    Returns:
        A record for the registry: the artifact digest, size, and what the
        conversion cost.
    """
    from safetensors.torch import load_file, save_file

    staging = cache_dir / spec.adapter_id
    staging.mkdir(parents=True, exist_ok=True)

    log.info("fetching %s at %s", spec.repo, spec.revision[:12])
    config_path = _download(spec.repo, spec.revision, FETCHED_FILES[0], staging)
    weights_path = _download(spec.repo, spec.revision, FETCHED_FILES[1], staging)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_source_config(config, spec)

    source = load_file(str(weights_path))
    tensors, stats = normalise_tensors(source, spec=spec, manifest=manifest)

    target = out_dir / spec.adapter_id
    target.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target / "adapter_model.safetensors"))
    (target / "adapter_config.json").write_text(
        json.dumps(canonical_config(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    from capability_subnet.common.hashing import sha256_file

    artifact = target / "adapter_model.safetensors"
    record = {
        "adapter_id": spec.adapter_id,
        "artifact_sha256": sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
        "converted_from_rank": spec.rank,
        "mean_retained_energy": stats["mean_retained_energy"],
        "lossless": spec.rank <= C.CANONICAL_RANK,
    }
    log.info(
        "%s: normalised rank %d→%d, retained energy %.6f, %s",
        spec.adapter_id,
        spec.rank,
        C.CANONICAL_RANK,
        record["mean_retained_energy"],
        "lossless" if record["lossless"] else "LOSSY — recertification required",
    )
    return record
