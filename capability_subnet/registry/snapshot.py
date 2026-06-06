"""Frozen pool snapshots.

A snapshot is the (registry, base manifest) pair a recipe was built against,
identified by one digest. Freezing the pool is what makes "the miner selected
adapters that existed and had these exact weights" a checkable statement rather
than a promise.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from capability_subnet.common.hashing import canonical_json_bytes, sha256_bytes
from capability_subnet.registry.adapters import AdapterRegistry, RegistryError, load_registry
from capability_subnet.registry.base_model import BaseManifest, load_base_manifest


@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    """An immutable view of the certified pool plus the base it targets."""

    registry: AdapterRegistry
    manifest: BaseManifest

    @property
    def sha256(self) -> str:
        """The digest every recipe declares as ``source_snapshot_sha256``."""
        return self.registry.snapshot_sha256()

    @property
    def base_revision(self) -> str:
        return self.manifest.revision

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return self.registry.ids

    def document(self) -> dict:
        """Human-readable snapshot description for publication."""
        return {
            "source_snapshot_sha256": self.sha256,
            "base_revision": self.manifest.revision,
            "base_model_repo": self.manifest.model_repo,
            "base_manifest_sha256": self.manifest.digest(),
            "canonical_rank": self.registry.canonical_rank,
            "adapter_count": len(self.registry.adapters),
            "adapters": list(self.adapter_ids),
            "distractors": list(self.registry.distractors()),
            "fully_certified": self.registry.fully_certified(),
        }

    def digest_of_document(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.document()))

    def validate_recipe_scope(self, base_revision: str, snapshot_sha: str) -> list[str]:
        """Check a recipe's declared scope against this snapshot.

        Returns:
            A list of human-readable problems. Empty means the recipe was built
            against exactly this pool.
        """
        problems: list[str] = []
        if base_revision != self.manifest.revision:
            problems.append(
                f"recipe targets base revision {base_revision!r}, "
                f"this network is pinned to {self.manifest.revision!r}"
            )
        if snapshot_sha != self.sha256:
            problems.append(
                f"recipe declares source snapshot {snapshot_sha[:19]}…, "
                f"the frozen pool is {self.sha256[:19]}…"
            )
        return problems

    def require_certified(self) -> None:
        """Refuse to serve an uncertified pool to a live network."""
        uncertified = sorted(
            entry.adapter_id for entry in self.registry.adapters if not entry.certified
        )
        if uncertified:
            raise RegistryError(
                f"{len(uncertified)} adapters have not completed certification: {uncertified}. "
                "Run the certification gates and record the results before freezing the pool."
            )


@lru_cache(maxsize=4)
def load_snapshot(
    registry_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> PoolSnapshot:
    """Load the frozen pool. Cached: the pool does not change while a process runs."""
    return PoolSnapshot(
        registry=load_registry(registry_path),
        manifest=load_base_manifest(manifest_path),
    )


def current_snapshot_sha256() -> str:
    """Convenience accessor used by miner tooling to fill in a recipe."""
    return load_snapshot().sha256
