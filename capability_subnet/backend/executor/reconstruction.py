"""Reconstruction with a cache and a cross-worker check.

Two concerns meet here.

**Caching.** Artifacts are addressed by their content, so a recipe that has been
built before is never built again — which matters because the references are
rebuilt every window and are identical every time.

**Cross-worker agreement.** Reconstruction is meant to be deterministic, and the
protocol treats a disagreement as an engine failure rather than a candidate
failure. Several workers build the same recipe independently and their artifact
digests must match. If they do not, evaluation of that candidate *pauses*: the
candidate is neither scored nor terminated, because scoring one of two
disagreeing artifacts would mean paying for a result nobody can reproduce.

The workers run in the same process here. That catches the failure modes
determinism actually has in practice — an unpinned kernel, a thread-count
dependence, an uninitialised buffer — while a genuine cross-machine check is what
a multi-host deployment adds on top.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from capability_subnet.common.schemas import Recipe
from capability_subnet.merge_engine.engine import (
    ReconstructionError,
    ReconstructionResult,
    reconstruct,
)
from capability_subnet.merge_engine.loader import AdapterSource
from capability_subnet.registry.snapshot import PoolSnapshot

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BuildOutcome:
    """The result of building one recipe."""

    artifact_sha256: str
    artifact_dir: Path
    size_bytes: int
    from_cache: bool
    workers_agreed: bool
    agreement_detail: str
    result: ReconstructionResult

    @property
    def usable(self) -> bool:
        return self.workers_agreed


class ArtifactCache:
    """Content-addressed artifact storage."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, artifact_sha256: str) -> Path:
        digest = artifact_sha256.split(":", 1)[-1]
        return self.root / digest[:2] / digest

    def contains(self, artifact_sha256: str) -> bool:
        return (self.path_for(artifact_sha256) / "adapter_model.safetensors").is_file()

    def size_of(self, artifact_sha256: str) -> int:
        weights = self.path_for(artifact_sha256) / "adapter_model.safetensors"
        return weights.stat().st_size if weights.is_file() else 0

    def evict(self, artifact_sha256: str) -> None:
        target = self.path_for(artifact_sha256)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def total_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())


class Reconstructor:
    """Builds candidate artifacts, with caching and the agreement check."""

    def __init__(
        self,
        snapshot: PoolSnapshot,
        source: AdapterSource,
        cache: ArtifactCache,
        *,
        workers: int = 2,
    ) -> None:
        self.snapshot = snapshot
        self.source = source
        self.cache = cache
        self.workers = max(1, workers)
        if self.workers < 2:
            log.warning(
                "running with a single reconstruction worker; the cross-worker artifact "
                "hash check is disabled and a determinism regression would go unnoticed"
            )

    def build(self, recipe: Recipe) -> BuildOutcome:
        """Build ``recipe``, or return the cached artifact if it exists.

        Raises:
            ReconstructionError: if the recipe cannot be reconstructed at all.
                That is a candidate failure and scores zero, unlike a worker
                disagreement, which is reported on the outcome instead.
        """
        # A dry build computes the digest without writing, which is how the cache
        # is consulted: the artifact's address is not known until it is built.
        probe = reconstruct(recipe, self.snapshot, self.source, output_dir=None)
        digest = probe.artifact_sha256

        if self.cache.contains(digest):
            log.info("artifact %s already cached", digest[:19])
            return BuildOutcome(
                artifact_sha256=digest,
                artifact_dir=self.cache.path_for(digest),
                size_bytes=self.cache.size_of(digest),
                from_cache=True,
                workers_agreed=True,
                agreement_detail="served from the artifact cache",
                result=probe,
            )

        agreed, detail = self._cross_check(recipe, probe)

        target = self.cache.path_for(digest)
        written = reconstruct(recipe, self.snapshot, self.source, output_dir=target)

        if written.artifact_sha256 != digest:
            # The in-memory digest and the written file disagree, which means the
            # serialisation path is not deterministic. Refuse to use either.
            self.cache.evict(digest)
            raise ReconstructionError(
                f"the artifact digest changed between the dry build ({digest[:19]}…) and "
                f"the written file ({written.artifact_sha256[:19]}…); serialisation is "
                "not deterministic on this host"
            )

        return BuildOutcome(
            artifact_sha256=digest,
            artifact_dir=target,
            size_bytes=written.size_bytes,
            from_cache=False,
            workers_agreed=agreed,
            agreement_detail=detail,
            result=written,
        )

    def _cross_check(self, recipe: Recipe, first: ReconstructionResult) -> tuple[bool, str]:
        """Rebuild the recipe and compare digests."""
        if self.workers < 2:
            return True, "cross-worker check disabled (single worker)"

        digests = {first.artifact_sha256}
        for index in range(1, self.workers):
            # A different thread count is the cheapest way to expose an
            # accumulation-order dependence, which is the determinism bug this
            # check is most likely to catch.
            repeat = reconstruct(
                recipe,
                self.snapshot,
                self.source,
                output_dir=None,
                threads=1 + index,
            )
            digests.add(repeat.artifact_sha256)

        if len(digests) == 1:
            return True, f"{self.workers} workers agree on {first.artifact_sha256[:19]}…"

        return False, (
            f"{self.workers} workers produced {len(digests)} distinct artifacts: "
            + ", ".join(sorted(digest[:19] for digest in digests))
        )
