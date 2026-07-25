"""The certified source-adapter pool.

The pool is frozen: a recipe may only reference adapter IDs that were in the
snapshot the recipe declares. Freezing is what makes reconstruction reproducible
and what stops a miner from pointing at weights nobody has vetted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from capability_subnet.common import constants as C
from capability_subnet.common.hashing import canonical_json_bytes, sha256_bytes


class RegistryError(Exception):
    """Raised when the adapter registry is malformed or used inconsistently."""


@dataclass(frozen=True, slots=True)
class CertificationRecord:
    """Outcome of the capability certification run for one adapter."""

    capability_score: float | None = None
    base_retention: float | None = None
    certified_at: str | None = None
    recertified_after_conversion: bool = False


@dataclass(frozen=True, slots=True)
class AdapterEntry:
    """One certified source adapter."""

    adapter_id: str
    capability: str
    description: str
    license: str
    license_allows_derivatives: bool
    provenance: str
    training_data_ref: str
    training_data_date_range: str
    known_overlaps: tuple[str, ...]
    artifact_uri: str
    artifact_sha256: str
    rank: int
    lora_alpha: int
    converted_from_rank: int | None
    certified: bool
    is_distractor: bool
    certification: CertificationRecord = field(default_factory=CertificationRecord)

    @property
    def scaling(self) -> float:
        """The ``alpha / rank`` factor folded into the effective update."""
        return self.lora_alpha / self.rank

    def snapshot_fields(self) -> dict[str, Any]:
        """The subset of fields that define this adapter's identity.

        Descriptions and provenance notes are deliberately excluded: editing a
        description must not change the snapshot digest and invalidate every
        outstanding recipe. Anything that changes the *weights* or how they are
        interpreted is included.
        """
        return {
            "adapter_id": self.adapter_id,
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "is_distractor": self.is_distractor,
        }


@dataclass(frozen=True, slots=True)
class AdapterRegistry:
    """The complete pool plus the metadata that scopes it."""

    registry_version: int
    workflow_id: str
    base_revision: str
    canonical_rank: int
    canonical_lora_alpha: int
    canonical_target_modules: tuple[str, ...]
    adapters: tuple[AdapterEntry, ...]

    # -- lookup -------------------------------------------------------------

    @property
    def ids(self) -> tuple[str, ...]:
        """Every adapter ID, sorted. Sorted order is the merge load order."""
        return tuple(sorted(entry.adapter_id for entry in self.adapters))

    @property
    def selectable_ids(self) -> tuple[str, ...]:
        """IDs a recipe may reference. Distractors are selectable on purpose —
        recognising that they hurt is part of the composition problem."""
        return self.ids

    def get(self, adapter_id: str) -> AdapterEntry:
        for entry in self.adapters:
            if entry.adapter_id == adapter_id:
                return entry
        raise RegistryError(
            f"adapter {adapter_id!r} is not in the certified pool. Available: {', '.join(self.ids)}"
        )

    def contains(self, adapter_id: str) -> bool:
        return any(entry.adapter_id == adapter_id for entry in self.adapters)

    def unknown_ids(self, adapter_ids: list[str]) -> list[str]:
        """Which of ``adapter_ids`` are not in the pool."""
        return sorted(set(adapter_ids) - set(self.ids))

    def distractors(self) -> tuple[str, ...]:
        return tuple(sorted(e.adapter_id for e in self.adapters if e.is_distractor))

    def capability_adapters(self) -> tuple[str, ...]:
        return tuple(sorted(e.adapter_id for e in self.adapters if not e.is_distractor))

    def fully_certified(self) -> bool:
        return all(entry.certified for entry in self.adapters)

    # -- snapshot -----------------------------------------------------------

    def snapshot_document(self) -> dict[str, Any]:
        """The canonical document the snapshot digest is taken over."""
        return {
            "registry_version": self.registry_version,
            "workflow_id": self.workflow_id,
            "base_revision": self.base_revision,
            "canonical_rank": self.canonical_rank,
            "canonical_lora_alpha": self.canonical_lora_alpha,
            "canonical_target_modules": list(self.canonical_target_modules),
            "adapters": [
                entry.snapshot_fields()
                for entry in sorted(self.adapters, key=lambda e: e.adapter_id)
            ],
        }

    def snapshot_sha256(self) -> str:
        """Digest every recipe must declare.

        A recipe that names a different snapshot is rejected at admission: it was
        built against a pool that no longer exists.
        """
        return sha256_bytes(canonical_json_bytes(self.snapshot_document()))


def _default_registry_path() -> Path:
    return Path(str(resources.files("capability_subnet.registry") / "data")) / (
        C.ADAPTER_REGISTRY_FILENAME
    )


def _parse_entry(raw: dict[str, Any]) -> AdapterEntry:
    cert_raw = raw.get("certification") or {}
    return AdapterEntry(
        adapter_id=raw["adapter_id"],
        capability=raw.get("capability", ""),
        description=raw.get("description", ""),
        license=raw.get("license", "unknown"),
        license_allows_derivatives=bool(raw.get("license_allows_derivatives", False)),
        provenance=raw.get("provenance", ""),
        training_data_ref=raw.get("training_data_ref", ""),
        training_data_date_range=raw.get("training_data_date_range", ""),
        known_overlaps=tuple(raw.get("known_overlaps", ())),
        artifact_uri=raw.get("artifact_uri", ""),
        artifact_sha256=raw.get("artifact_sha256", ""),
        rank=int(raw["rank"]),
        lora_alpha=int(raw["lora_alpha"]),
        converted_from_rank=(
            int(raw["converted_from_rank"]) if raw.get("converted_from_rank") else None
        ),
        certified=bool(raw.get("certified", False)),
        is_distractor=bool(raw.get("is_distractor", False)),
        certification=CertificationRecord(
            capability_score=cert_raw.get("capability_score"),
            base_retention=cert_raw.get("base_retention"),
            certified_at=cert_raw.get("certified_at"),
            recertified_after_conversion=bool(cert_raw.get("recertified_after_conversion", False)),
        ),
    )


@lru_cache(maxsize=4)
def load_registry(path: str | Path | None = None) -> AdapterRegistry:
    """Load and structurally validate the certified pool."""
    registry_path = Path(path) if path is not None else _default_registry_path()
    if not registry_path.is_file():
        raise RegistryError(f"adapter registry not found at {registry_path}")

    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(
            f"adapter registry at {registry_path} is not valid JSON: {exc}"
        ) from exc

    entries = tuple(_parse_entry(item) for item in raw.get("adapters", []))
    if not entries:
        raise RegistryError("adapter registry contains no adapters")

    ids = [entry.adapter_id for entry in entries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise RegistryError(f"duplicate adapter IDs in registry: {duplicates}")

    canonical_rank = int(raw.get("canonical_rank", C.CANONICAL_RANK))
    off_rank = [e.adapter_id for e in entries if e.rank != canonical_rank]
    if off_rank:
        raise RegistryError(
            f"adapters are not at the canonical rank {canonical_rank}: {off_rank}. "
            "Convert them offline and recertify before admission."
        )

    return AdapterRegistry(
        registry_version=int(raw.get("registry_version", 1)),
        workflow_id=raw.get("workflow_id", C.DEFAULT_WORKFLOW_ID),
        base_revision=raw.get("base_revision", ""),
        canonical_rank=canonical_rank,
        canonical_lora_alpha=int(raw.get("canonical_lora_alpha", C.CANONICAL_LORA_ALPHA)),
        canonical_target_modules=tuple(
            raw.get("canonical_target_modules", C.CANONICAL_TARGET_MODULES)
        ),
        adapters=entries,
    )
