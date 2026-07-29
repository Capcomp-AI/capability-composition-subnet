"""Building and checking recipes.

A miner's whole submission is one JSON document. Everything in this module exists
to make that document easy to get right and impossible to get subtly wrong: build
it from the frozen pool, validate it against the same code the engine runs, write
it in the exact canonical form whose digest goes on-chain.

The digest is taken over the *parsed* document, not the file bytes, so
pretty-printing is free — but the engine re-derives it at admission, and a recipe
whose canonical form does not match its commitment is rejected. Producing the
file through :func:`write_recipe` removes any chance of that mismatch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from capability_subnet.common import constants as C
from capability_subnet.common.hashing import canonical_json_str
from capability_subnet.common.schemas import CompressionSpec, MergeSpec, OutputSpec, Recipe
from capability_subnet.registry.snapshot import PoolSnapshot, load_snapshot

log = logging.getLogger(__name__)


class RecipeError(Exception):
    """Raised when a recipe cannot be built, loaded or validated."""


def new_recipe(
    selected_adapters: list[str],
    *,
    combination_type: str = C.MERGE_TIES_SVD,
    density: float | None = 0.45,
    majority_sign_method: str | None = "total",
    random_seed: int = 0,
    global_weights: dict[str, float] | None = None,
    layer_group_overrides: dict[str, dict[str, float]] | None = None,
    output_rank: int = C.CANONICAL_RANK,
    svd_clamp_quantile: float = 1.0,
    adapter_name: str = "candidate",
    snapshot: PoolSnapshot | None = None,
) -> Recipe:
    """Build a recipe against the frozen pool.

    The base revision and snapshot digest are filled in from the pool rather than
    taken as arguments. Those two fields are the only ones a miner can get wrong
    without noticing until admission rejects the submission, and there is no
    legitimate reason to set them by hand.
    """
    pool = snapshot or load_snapshot()

    unknown = pool.registry.unknown_ids(selected_adapters)
    if unknown:
        raise RecipeError(
            f"these adapters are not in the certified pool: {unknown}. "
            f"Available: {', '.join(pool.adapter_ids)}"
        )

    try:
        return Recipe(
            workflow_id=pool.registry.workflow_id,
            base_revision=pool.manifest.revision,
            source_snapshot_sha256=pool.sha256,
            selected_adapters=list(selected_adapters),
            merge=MergeSpec(
                combination_type=combination_type,
                density=density,
                majority_sign_method=majority_sign_method,  # type: ignore[arg-type]
                random_seed=random_seed,
            ),
            global_weights=dict(global_weights or {}),
            layer_group_overrides={
                group: dict(values) for group, values in (layer_group_overrides or {}).items()
            },
            compression=CompressionSpec(
                output_rank=output_rank, svd_clamp_quantile=svd_clamp_quantile
            ),
            output=OutputSpec(adapter_name=adapter_name),
        )
    except Exception as exc:  # noqa: BLE001 - pydantic raises its own type
        raise RecipeError(str(exc)) from exc


def load_recipe(path: str | Path) -> Recipe:
    """Read and validate a recipe file."""
    target = Path(path)
    if not target.is_file():
        raise RecipeError(f"no recipe at {target}")

    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecipeError(f"{target} is not valid JSON: {exc}") from exc

    try:
        return Recipe.model_validate(document)
    except Exception as exc:  # noqa: BLE001
        raise RecipeError(f"{target} is not a valid recipe: {exc}") from exc


def write_recipe(recipe: Recipe, path: str | Path) -> str:
    """Write a recipe in canonical form and return its digest."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = recipe.model_dump(mode="json", exclude_none=True)
    target.write_text(canonical_json_str(document), encoding="utf-8")

    digest = recipe.digest()
    log.info("wrote %s (%s)", target, digest[:19])
    return digest


def check_recipe(recipe: Recipe, snapshot: PoolSnapshot | None = None) -> list[str]:
    """Every problem the engine would find at admission.

    Running this before committing is the difference between finding out in a
    second and finding out after spending the hotkey's one shot.
    """
    pool = snapshot or load_snapshot()
    problems = pool.validate_recipe_scope(recipe.base_revision, recipe.source_snapshot_sha256)

    unknown = pool.registry.unknown_ids(recipe.selected_adapters)
    if unknown:
        problems.append(f"adapters not in the certified pool: {unknown}")

    if recipe.workflow_id != pool.registry.workflow_id:
        problems.append(
            f"recipe targets workflow {recipe.workflow_id!r}, this pool serves "
            f"{pool.registry.workflow_id!r}"
        )

    return problems


def advise(recipe: Recipe, snapshot: PoolSnapshot | None = None) -> list[str]:
    """Non-blocking observations about a recipe.

    None of these prevent admission. They flag choices that are legal but usually
    unintended — the kind of thing that turns into a wasted shot rather than a
    rejected submission.
    """
    pool = snapshot or load_snapshot()
    notes: list[str] = []

    distractors = set(pool.registry.distractors()) & set(recipe.selected_adapters)
    if distractors:
        notes.append(
            f"selects controlled distractor adapter(s) {sorted(distractors)}. They are "
            "selectable on purpose, but they were certified as harmful to this workflow."
        )

    anchor = snapshot.registry.retention_anchor()
    if anchor is not None and anchor not in recipe.selected_adapters:
        notes.append(
            "does not include the general-reasoning retention adapter. The base-retention "
            "gate rejects packages that traded general ability for workflow score."
        )

    estimated = estimate_artifact_bytes(recipe, pool)
    if estimated > C.MAX_ARTIFACT_BYTES:
        notes.append(
            f"output rank {recipe.compression.output_rank} yields roughly "
            f"{estimated / (1024 * 1024):.0f} MB, above the "
            f"{C.MAX_ARTIFACT_BYTES / (1024 * 1024):.0f} MB artifact-size gate. "
            "Choose a lower rank."
        )

    extreme = {
        adapter: weight for adapter, weight in recipe.global_weights.items() if abs(weight) > 1.5
    }
    if extreme:
        notes.append(
            f"uses coefficients above 1.5 in magnitude for {sorted(extreme)}. Large "
            "coefficients tend to dominate the merged update and lower base retention."
        )

    if recipe.merge.combination_type in C.STOCHASTIC_METHODS and recipe.merge.random_seed == 0:
        notes.append(
            "uses a stochastic merge with the default seed of 0. The seed is part of the "
            "recipe and changes the artifact, so it is worth searching over."
        )

    return notes


def estimate_artifact_bytes(recipe: Recipe, snapshot: PoolSnapshot | None = None) -> int:
    """Predicted artifact size, without building anything.

    Exact rather than approximate: the emitted adapter covers every adapted
    projection at the declared rank in bfloat16, so the size follows from the
    base model's shapes and the rank alone.
    """
    pool = snapshot or load_snapshot()
    manifest = pool.manifest
    rank = recipe.compression.output_rank

    parameters = 0
    for module in C.CANONICAL_TARGET_MODULES:
        shape = manifest.shape_of(module)
        parameters += rank * shape.in_features  # lora_A
        parameters += shape.out_features * rank  # lora_B

    return parameters * manifest.num_hidden_layers * 2  # two bytes per bfloat16 value


def describe(recipe: Recipe, snapshot: PoolSnapshot | None = None) -> str:
    """A human-readable summary."""
    pool = snapshot or load_snapshot()
    groups = pool.manifest.layer_groups

    lines = [
        f"workflow       : {recipe.workflow_id}",
        f"base revision  : {recipe.base_revision}",
        f"source snapshot: {recipe.source_snapshot_sha256}",
        f"recipe digest  : {recipe.digest()}",
        f"method         : {recipe.merge.combination_type}",
    ]
    if recipe.merge.density is not None:
        lines.append(f"density        : {recipe.merge.density}")
    if recipe.merge.majority_sign_method:
        lines.append(f"sign election  : {recipe.merge.majority_sign_method}")
    lines.append(f"random seed    : {recipe.merge.random_seed}")
    lines.append(
        f"compression    : rank {recipe.compression.output_rank}, "
        f"clamp {recipe.compression.svd_clamp_quantile}"
    )
    lines.append(f"estimated size : {estimate_artifact_bytes(recipe, pool) / (1024 * 1024):.0f} MB")
    lines.append("adapters       :")

    for adapter in recipe.sorted_adapters():
        coefficients = [
            f"{group}={recipe.effective_weight(adapter, group):.2f}" for group in sorted(groups)
        ]
        lines.append(f"  - {adapter:<34} {' '.join(coefficients)}")

    return "\n".join(lines)
