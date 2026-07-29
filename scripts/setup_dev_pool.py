#!/usr/bin/env python3
"""Materialise a synthetic adapter pool for local development.

The certified pool is 12 adapters of roughly 333 MB each against an 8B base
model. Downloading that to try a recipe locally is a poor trade, so this script
writes structurally identical adapters filled with small random values against a
miniature base manifest.

The result is useless for measuring quality — the weights mean nothing — and
exactly right for exercising everything else: reconstruction, determinism,
artifact hashing, size gates and the whole engine loop.

Nothing here can reach a live network. These adapters carry no certification
record, and the engine's preflight refuses to start while any pool member is
uncertified — so a synthetic pool fails closed rather than quietly scoring
miners against random numbers. For the real pool, use
``scripts/import_public_adapters.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from capability_subnet.common import constants as C
from capability_subnet.registry.snapshot import load_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="pool", help="Directory to write adapters into.")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.02,
        help="Standard deviation of the generated factors.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    snapshot = load_snapshot()
    manifest = snapshot.manifest
    root = Path(args.out)

    generator = torch.Generator().manual_seed(args.seed)

    for adapter_id in snapshot.adapter_ids:
        directory = root / adapter_id
        directory.mkdir(parents=True, exist_ok=True)

        tensors: dict[str, torch.Tensor] = {}
        for layer in range(manifest.num_hidden_layers):
            for module in C.CANONICAL_TARGET_MODULES:
                shape = manifest.shape_of(module)
                tensors[manifest.tensor_key(layer, module, "lora_A")] = (
                    torch.randn(C.CANONICAL_RANK, shape.in_features, generator=generator)
                    * args.scale
                ).to(torch.bfloat16)
                tensors[manifest.tensor_key(layer, module, "lora_B")] = (
                    torch.randn(shape.out_features, C.CANONICAL_RANK, generator=generator)
                    * args.scale
                ).to(torch.bfloat16)

        save_file(tensors, str(directory / "adapter_model.safetensors"))

        (directory / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": C.CANONICAL_PEFT_TYPE,
                    "task_type": "CAUSAL_LM",
                    "base_model_name_or_path": manifest.model_repo,
                    "base_model_revision": manifest.revision,
                    "r": C.CANONICAL_RANK,
                    "lora_alpha": C.CANONICAL_LORA_ALPHA,
                    "lora_dropout": C.CANONICAL_LORA_DROPOUT,
                    "bias": C.CANONICAL_BIAS,
                    "target_modules": sorted(C.CANONICAL_TARGET_MODULES),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        size = (directory / "adapter_model.safetensors").stat().st_size
        print(f"  {adapter_id:<34} {size / (1024 * 1024):7.1f} MB")

    print(f"\nWrote {len(snapshot.adapter_ids)} synthetic adapters to {root}")
    print("These weights are random. They exercise the pipeline; they measure nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
