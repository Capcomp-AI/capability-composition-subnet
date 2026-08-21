"""The pinned base model manifest.

One immutable base model is the foundation of the whole protocol: every
certified adapter targets it, every reconstruction produces a delta against it,
and every score is measured on it. Repinning the base creates a new arena, not a
new run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from capability_subnet.common import constants as C
from capability_subnet.common.hashing import sha256_json

#: Placeholder written into the shipped manifest. A live network must replace it
#: with an immutable upstream commit before genesis.
UNPINNED_SENTINEL = "PIN_BEFORE_GENESIS"


class BaseManifestError(Exception):
    """Raised when the base manifest is missing, malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class ModuleShape:
    """Input/output dimensions of one adapted projection."""

    in_features: int
    out_features: int


@dataclass(frozen=True, slots=True)
class BaseManifest:
    """Everything the merge engine needs to know about the pinned base model."""

    model_repo: str
    revision: str
    license: str
    tokenizer_repo: str
    architecture: str
    dtype: str
    num_hidden_layers: int
    hidden_size: int
    module_shapes: dict[str, ModuleShape]
    layer_module_template: str
    raw: dict[str, Any]

    @property
    def is_pinned(self) -> bool:
        """Whether the manifest points at an immutable revision."""
        return self.revision != UNPINNED_SENTINEL and bool(self.revision.strip())

    @property
    def layer_groups(self) -> dict[str, tuple[int, int]]:
        """The four named layer groups for this model's depth."""
        return C.build_layer_groups(self.num_hidden_layers)

    def digest(self) -> str:
        """Content address of the manifest, used in evaluation reports."""
        return sha256_json(self.raw)

    def expected_lora_a_shape(self, module: str, rank: int) -> tuple[int, int]:
        """Shape of ``lora_A`` for ``module`` at ``rank``: ``(rank, in_features)``."""
        return (rank, self.shape_of(module).in_features)

    def expected_lora_b_shape(self, module: str, rank: int) -> tuple[int, int]:
        """Shape of ``lora_B`` for ``module`` at ``rank``: ``(out_features, rank)``."""
        return (self.shape_of(module).out_features, rank)

    def shape_of(self, module: str) -> ModuleShape:
        try:
            return self.module_shapes[module]
        except KeyError:
            raise BaseManifestError(
                f"module {module!r} is not part of the pinned base model's adapted set "
                f"({sorted(self.module_shapes)})"
            ) from None

    def tensor_key(self, layer: int, module: str, matrix: str) -> str:
        """Canonical safetensors key for one adapter matrix.

        ``matrix`` is ``lora_A`` or ``lora_B``. Keys are built from the manifest
        template so a base model with a different module path does not require
        code changes.
        """
        if matrix not in ("lora_A", "lora_B"):
            raise ValueError(f"matrix must be lora_A or lora_B, got {matrix!r}")
        block = "self_attn" if module in self.raw["attention_modules"] else "mlp"
        prefix = self.layer_module_template.format(layer=layer, block=block, module=module)
        return f"{prefix}.{matrix}.weight"

    def all_tensor_keys(self, modules: tuple[str, ...] = C.CANONICAL_TARGET_MODULES) -> list[str]:
        """Every tensor key a fully populated adapter must contain, sorted."""
        keys: list[str] = []
        for layer in range(self.num_hidden_layers):
            for module in modules:
                keys.append(self.tensor_key(layer, module, "lora_A"))
                keys.append(self.tensor_key(layer, module, "lora_B"))
        return sorted(keys)


def _default_manifest_path() -> Path:
    return Path(str(resources.files("capability_subnet.registry") / "data")) / (
        C.BASE_MANIFEST_FILENAME
    )


@lru_cache(maxsize=4)
def load_base_manifest(path: str | Path | None = None) -> BaseManifest:
    """Load and validate the base manifest.

    Raises:
        BaseManifestError: if the file is missing, cannot be parsed, or declares
            a depth that disagrees with the compiled-in protocol constant. That
            last check matters: layer-group boundaries are derived from the
            depth, so a silent mismatch would change how every recipe
            reconstructs.
    """
    manifest_path = Path(path) if path is not None else _default_manifest_path()
    if not manifest_path.is_file():
        raise BaseManifestError(f"base manifest not found at {manifest_path}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaseManifestError(
            f"base manifest at {manifest_path} is not valid JSON: {exc}"
        ) from exc

    required = (
        "model_repo",
        "revision",
        "num_hidden_layers",
        "hidden_size",
        "module_shapes",
        "layer_module_template",
        "attention_modules",
    )
    missing = [field for field in required if field not in raw]
    if missing:
        raise BaseManifestError(f"base manifest is missing fields: {missing}")

    num_layers = int(raw["num_hidden_layers"])
    if num_layers != C.NUM_HIDDEN_LAYERS:
        raise BaseManifestError(
            f"base manifest declares {num_layers} hidden layers but this build of the "
            f"protocol is compiled for {C.NUM_HIDDEN_LAYERS}. Layer-group boundaries "
            "are derived from the depth, so this mismatch would change reconstruction. "
            "Update capability_subnet.common.constants.NUM_HIDDEN_LAYERS and bump the "
            "spec version."
        )

    shapes = {
        name: ModuleShape(int(value["in_features"]), int(value["out_features"]))
        for name, value in raw["module_shapes"].items()
    }
    missing_modules = sorted(set(C.CANONICAL_TARGET_MODULES) - set(shapes))
    if missing_modules:
        raise BaseManifestError(f"base manifest has no shapes for modules: {missing_modules}")

    return BaseManifest(
        model_repo=raw["model_repo"],
        revision=raw["revision"],
        license=raw.get("license", "unknown"),
        tokenizer_repo=raw.get("tokenizer_repo", raw["model_repo"]),
        architecture=raw.get("architecture", "unknown"),
        dtype=raw.get("dtype", C.CANONICAL_DTYPE),
        num_hidden_layers=num_layers,
        hidden_size=int(raw["hidden_size"]),
        module_shapes=shapes,
        layer_module_template=raw["layer_module_template"],
        raw=raw,
    )


def require_pinned(manifest: BaseManifest) -> None:
    """Fail loudly when a live component is started against an unpinned base.

    Development and testing run happily against the placeholder; anything that
    writes to a chain must not.
    """
    if not manifest.is_pinned:
        raise BaseManifestError(
            f"the base model revision is still the {UNPINNED_SENTINEL!r} placeholder. "
            f"Pin {manifest.model_repo} to an immutable commit in "
            f"registry/data/{C.BASE_MANIFEST_FILENAME} before running against a live network."
        )
