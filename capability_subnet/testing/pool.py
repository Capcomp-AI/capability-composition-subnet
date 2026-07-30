"""Building the miniature pool, without a test framework.

The same structure as the real pool at a fraction of the volume: four decoder
layers so every layer group is populated, 16-wide projections so a merge is
instant. Enough to exercise shapes, determinism and control flow.

Plain functions, importable anywhere. Seeding a store, reproducing a merge by
hand or running the engine outside a test run should all get the identical
objects, and none of them should have to install pytest to do it. The pytest
fixtures that wrap these live in the package ``__init__``.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CompressionSpec, MergeSpec, OutputSpec, Recipe
from capability_subnet.registry.adapters import AdapterEntry, AdapterRegistry, CertificationRecord
from capability_subnet.registry.base_model import BaseManifest, ModuleShape
from capability_subnet.registry.snapshot import PoolSnapshot

#: The maintenance workflow, named explicitly rather than taken from the default.
MAINTENANCE_WORKFLOW_ID = "industrial_maintenance_de_v1"

TINY_LAYERS = 4
TINY_WIDTH = 16
TINY_RANK = 8

TINY_ADAPTERS = (
    "alpha-capability-v1",
    "beta-capability-v1",
    "gamma-capability-v1",
    "delta-distractor-v1",
)


def make_manifest() -> BaseManifest:
    """A base manifest small enough to merge in milliseconds.

    Built directly rather than loaded, because the loader refuses a manifest whose
    depth disagrees with the compiled-in protocol constant — a check that protects
    the real pool and would block this one.

    A plain function, not only a fixture: anything that needs the miniature pool
    outside a test run — seeding a store, reproducing a merge by hand — should get
    the identical objects rather than rebuilding them.
    """
    shapes = {module: ModuleShape(TINY_WIDTH, TINY_WIDTH) for module in C.CANONICAL_TARGET_MODULES}
    raw = {
        "model_repo": "test/tiny-base",
        "revision": "tiny-rev-1",
        "num_hidden_layers": TINY_LAYERS,
        "hidden_size": TINY_WIDTH,
        "attention_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp_modules": ["gate_proj", "up_proj", "down_proj"],
    }
    return BaseManifest(
        model_repo="test/tiny-base",
        revision="tiny-rev-1",
        license="MIT",
        tokenizer_repo="test/tiny-base",
        architecture="TinyForCausalLM",
        dtype="bfloat16",
        num_hidden_layers=TINY_LAYERS,
        hidden_size=TINY_WIDTH,
        module_shapes=shapes,
        layer_module_template="base_model.model.model.layers.{layer}.{block}.{module}",
        raw=raw,
    )


def make_registry() -> AdapterRegistry:
    """Four certified adapters, one of them a controlled distractor."""
    entries = tuple(
        AdapterEntry(
            adapter_id=adapter_id,
            capability=adapter_id.rsplit("-", 1)[0],
            description=f"Test adapter {adapter_id}",
            license="Apache-2.0",
            license_allows_derivatives=True,
            provenance="synthetic",
            training_data_ref="test",
            training_data_date_range="2025-01/2025-12",
            known_overlaps=(),
            artifact_uri=f"hf:test/{adapter_id}",
            artifact_sha256="sha256:" + f"{index:064d}",
            rank=TINY_RANK,
            lora_alpha=TINY_RANK,
            converted_from_rank=None,
            certified=True,
            is_distractor=adapter_id.endswith("distractor-v1"),
            certification=CertificationRecord(
                capability_score=0.8, base_retention=0.99, certified_at="2026-01-01"
            ),
        )
        for index, adapter_id in enumerate(TINY_ADAPTERS)
    )

    return AdapterRegistry(
        registry_version=1,
        workflow_id=C.DEFAULT_WORKFLOW_ID,
        base_revision="tiny-rev-1",
        canonical_rank=TINY_RANK,
        canonical_lora_alpha=TINY_RANK,
        canonical_target_modules=C.CANONICAL_TARGET_MODULES,
        adapters=entries,
    )


def make_snapshot() -> PoolSnapshot:
    return PoolSnapshot(registry=make_registry(), manifest=make_manifest())


def make_tensors(tiny_manifest: BaseManifest) -> dict[str, dict[str, torch.Tensor]]:
    """Deterministic weights for every test adapter.

    Seeded: two callers that merge the same adapters must be merging identical
    inputs, or a determinism assertion would be testing this rather than the
    engine.
    """
    generator = torch.Generator().manual_seed(20260725)
    adapters: dict[str, dict[str, torch.Tensor]] = {}

    for adapter_id in TINY_ADAPTERS:
        tensors: dict[str, torch.Tensor] = {}
        for layer in range(tiny_manifest.num_hidden_layers):
            for module in C.CANONICAL_TARGET_MODULES:
                shape = tiny_manifest.shape_of(module)
                tensors[tiny_manifest.tensor_key(layer, module, "lora_A")] = (
                    torch.randn(TINY_RANK, shape.in_features, generator=generator) * 0.05
                )
                tensors[tiny_manifest.tensor_key(layer, module, "lora_B")] = (
                    torch.randn(shape.out_features, TINY_RANK, generator=generator) * 0.05
                )
        adapters[adapter_id] = tensors

    return adapters


def write_pool(root: Path, tiny_tensors=None, tiny_manifest=None) -> Path:
    """Materialise the miniature adapters on disk as safetensors."""
    import json

    from safetensors.torch import save_file

    tiny_manifest = tiny_manifest or make_manifest()
    tiny_tensors = tiny_tensors if tiny_tensors is not None else make_tensors(tiny_manifest)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for adapter_id, tensors in tiny_tensors.items():
        directory = root / adapter_id
        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {key: value.to(torch.bfloat16).contiguous() for key, value in tensors.items()},
            str(directory / "adapter_model.safetensors"),
        )
        (directory / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": C.CANONICAL_PEFT_TYPE,
                    "base_model_name_or_path": tiny_manifest.model_repo,
                    "base_model_revision": tiny_manifest.revision,
                    "r": TINY_RANK,
                    "lora_alpha": TINY_RANK,
                    "lora_dropout": 0.0,
                    "bias": "none",
                    "target_modules": sorted(C.CANONICAL_TARGET_MODULES),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return root


def build_recipe(
    snapshot: PoolSnapshot,
    *,
    adapters: list[str] | None = None,
    combination_type: str = C.MERGE_TIES_SVD,
    density: float | None = 0.5,
    sign_method: str | None = "total",
    seed: int = 0,
    global_weights: dict[str, float] | None = None,
    layer_group_overrides: dict[str, dict[str, float]] | None = None,
    output_rank: int | None = None,
    clamp: float = 1.0,
) -> Recipe:
    """Build a valid recipe against a snapshot, for tests that need one."""
    return Recipe(
        workflow_id=snapshot.registry.workflow_id,
        base_revision=snapshot.manifest.revision,
        source_snapshot_sha256=snapshot.sha256,
        selected_adapters=adapters or list(TINY_ADAPTERS[:3]),
        merge=MergeSpec(
            combination_type=combination_type,
            density=density,
            majority_sign_method=sign_method,  # type: ignore[arg-type]
            random_seed=seed,
        ),
        global_weights=dict(global_weights or {}),
        layer_group_overrides={
            group: dict(values) for group, values in (layer_group_overrides or {}).items()
        },
        compression=CompressionSpec(
            output_rank=output_rank or snapshot.registry.canonical_rank,
            svd_clamp_quantile=clamp,
        ),
        output=OutputSpec(adapter_name="candidate"),
    )


def make_results(
    stage_scores: dict[str, float],
    *,
    count: int = 30,
    success_rate: float = 1.0,
    prefix: str = "hidden",
    seed: int = 0,
    unsafe: int = 0,
    seconds: float = 5.0,
):
    """Build synthetic sample rows with known statistics.

    Tests for the aggregator and the comparator need rows with *exactly* known
    means; deriving them from a real run would make those tests depend on model
    behaviour, which is the one thing they are not about.
    """
    from capability_subnet.common.schemas import InstanceResult, StageResult

    rng = random.Random(seed)
    rows = []
    successes = int(round(count * success_rate))

    for index in range(count):
        rows.append(
            InstanceResult(
                instance_id=f"{prefix}-{index:06d}",
                instance_seed=index,
                split="hidden" if prefix == "hidden" else "ood",
                stages={
                    stage: StageResult(stage=stage, score=score, passed=score >= 1.0)
                    for stage, score in stage_scores.items()
                },
                final_state_correct=index < successes,
                end_to_end_success=index < successes,
                critical_unsafe_actions=unsafe if index == 0 else 0,
                turns_used=rng.randint(6, 10),
                output_tokens=rng.randint(400, 900),
                input_tokens=rng.randint(2000, 4000),
                wall_seconds=seconds,
            )
        )
    return rows
