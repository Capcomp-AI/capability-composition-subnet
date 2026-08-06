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


#: Whether an adapter must be capability-measured before a recipe may select it.
#:
#: False under the leaderboard contract, and the reasoning is that the two
#: contracts need different things from this gate. When a submission had to beat
#: an absolute reference bar, an unmeasured adapter was a hole in the argument —
#: the bar was a claim about value, and a claim built on uncharacterised weights
#: is not one. When the highest score simply wins, a miner who selects a poor
#: adapter is answered by its score, immediately and without the operator having
#: to have measured it first. The market does the work the gate was doing, and
#: does it over a far larger pool than an operator can characterise by hand.
#:
#: What does *not* relax is structural admission. These tensors are loaded into a
#: process that also holds hidden evaluation material, and no incentive argument
#: touches that. Capability measurements, where the operator has them, are still
#: published — as information a miner can use rather than a door they must pass.
REQUIRE_CAPABILITY_CERTIFICATION: bool = False


@dataclass(frozen=True, slots=True)
class CertificationRecord:
    """Outcome of the capability certification run for one adapter."""

    capability_score: float | None = None
    base_retention: float | None = None
    certified_at: str | None = None
    recertified_after_conversion: bool = False

    @property
    def is_measured(self) -> bool:
        """Whether both capability measurements have actually been taken.

        A missing measurement is not a passing one. The floors themselves are
        applied by :func:`certification.check_capability_certification`; what
        this answers is the prior question of whether anybody looked.
        """
        return self.capability_score is not None and self.base_retention is not None


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
    #: Where this adapter came from, at the revision it was taken at. Recorded
    #: because the pool has to be reproducible: an operator materialising it on a
    #: new host fetches exactly these, and without the revision "the same
    #: adapter" means whatever that repository holds today.
    #:
    #: Deliberately outside `snapshot_fields`, so recording provenance never
    #: moves the digest every outstanding recipe declares — and defaulted, so a
    #: registry written before these were recorded still loads.
    source_repo: str = ""
    source_revision: str = ""
    source_rank: int = 0
    source_lora_alpha: int = 0

    @property
    def scaling(self) -> float:
        """The ``alpha / rank`` factor folded into the effective update."""
        return self.lora_alpha / self.rank

    @property
    def selectable(self) -> bool:
        """Whether a recipe may reference this adapter.

        Certification answers two different questions and they gate different
        things. The structural gates — safetensors only, the canonical config,
        the pinned base revision, correct shapes, finite values, a licence that
        permits derivatives — decide whether these tensors may be loaded into a
        process that also holds hidden evaluation material. They are not
        negotiable, and ``certified`` records them.

        The capability gates ask whether the adapter is any *good* at what it
        claims, and whether it destroyed the base model's general ability
        getting there. That is a measurement, and it arrives per adapter on its
        own schedule.

        Collapsing the two meant a pool could not open until every last adapter
        had been measured — so one unmeasured adapter blocked the network rather
        than blocking itself. Separating them lets the pool grow: an adapter
        that is structurally sound but unmeasured sits in the registry, safe to
        load and not selectable, until someone measures it.
        """
        if REQUIRE_CAPABILITY_CERTIFICATION:
            return self.certified and self.certification.is_measured
        return self.certified

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
            # Included because it decides which recipes are *valid*. Two engines
            # disagreeing about whether an adapter may be selected would admit
            # different submissions while claiming the same pool, and the
            # divergence would look like one of them misbehaving. Certifying an
            # adapter therefore changes the snapshot digest and invalidates
            # outstanding recipes — which is correct: the pool changed.
            "selectable": self.selectable,
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
        """IDs a recipe may reference.

        Distractors are selectable on purpose — recognising that they hurt is
        part of the composition problem. Unmeasured adapters are not, because a
        miner cannot be asked to reason about an adapter nobody has characterised
        and the engine cannot defend a score that depends on one.
        """
        return tuple(sorted(e.adapter_id for e in self.adapters if e.selectable))

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

    def unselectable_ids(self, adapter_ids: list[str]) -> list[str]:
        """Which of ``adapter_ids`` are in the pool but not yet selectable."""
        return sorted((set(adapter_ids) & set(self.ids)) - set(self.selectable_ids))

    def distractors(self) -> tuple[str, ...]:
        return tuple(
            sorted(e.adapter_id for e in self.adapters if e.selectable and e.is_distractor)
        )

    def capability_adapters(self) -> tuple[str, ...]:
        """Selectable non-distractors — what the reference merges are built from.

        Selectable rather than merely present: a reference built from an
        unmeasured adapter would set the bar every miner has to clear using
        weights nobody has characterised.
        """
        return tuple(
            sorted(e.adapter_id for e in self.adapters if e.selectable and not e.is_distractor)
        )

    def retention_anchor(self) -> str | None:
        """The adapter that exists to preserve general ability under merging.

        Identified by declared capability rather than by name so repinning the
        pool does not silently leave the miner tooling advising about an adapter
        that is no longer there.
        """
        for entry in sorted(self.adapters, key=lambda e: e.adapter_id):
            if entry.capability == "general_reasoning_retention":
                return entry.adapter_id
        return None

    def adapters_with_capability(self, capability: str) -> tuple[str, ...]:
        return tuple(sorted(e.adapter_id for e in self.adapters if e.capability == capability))

    def fully_certified(self) -> bool:
        """Whether every adapter is structurally admitted *and* measured."""
        return all(entry.selectable for entry in self.adapters)

    def structurally_admitted(self) -> bool:
        """Whether every adapter is safe to load.

        This one is not negotiable: the merge engine loads these tensors into a
        process that also holds hidden evaluation material.
        """
        return all(entry.certified for entry in self.adapters)

    def unmeasured(self) -> tuple[str, ...]:
        """Structurally sound adapters nobody has characterised yet."""
        return tuple(
            sorted(e.adapter_id for e in self.adapters if e.certified and not e.selectable)
        )

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
        source_repo=raw.get("source_repo", ""),
        source_revision=raw.get("source_revision", ""),
        source_rank=int(raw.get("source_rank", raw.get("rank", 0))),
        source_lora_alpha=int(raw.get("source_lora_alpha", raw.get("lora_alpha", 0))),
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
