"""Deterministic merge engine.

Turns a declarative recipe into a merged LoRA artifact, byte-identically on
every machine that runs the same software image **on the same compute device**.

That last clause is measured, not hedging. `linear` reproduces byte-for-byte
across CPU and GPU. Every method that factorises — anything ending `_svd` — does
not: LAPACK and cuSOLVER return different factorisations of the same matrix, and
the difference is far too large to round away. On a 12288x4096 delta the singular
values disagree by up to 9e-4 relative, the rank-64 reconstruction by up to
6.5e-3, and **65% of elements still differ after bfloat16 rounding** — bfloat16's
own epsilon is 7.8e-3, so quantisation does not close it.

The practical consequence is narrow but real: an artifact digest identifies a
recipe *and the device that merged it*. Validators are unaffected — they re-score
from published traces and never re-merge — but a miner reproducing a published
digest must merge on the same kind of device the engine did.
"""

from capability_subnet.merge_engine.canonical_writer import write_artifact
from capability_subnet.merge_engine.engine import (
    AdapterUnavailableError,
    ReconstructionError,
    ReconstructionResult,
    ReconstructionStats,
    artifact_hashes_agree,
    reconstruct,
)
from capability_subnet.merge_engine.factorize import factorize
from capability_subnet.merge_engine.loader import (
    AdapterLoadError,
    AdapterSource,
    InMemoryAdapterSource,
    SafetensorsAdapterSource,
)
from capability_subnet.merge_engine.methods import PIPELINES, describe, pipeline_for

__all__ = [
    "PIPELINES",
    "AdapterLoadError",
    "AdapterUnavailableError",
    "AdapterSource",
    "InMemoryAdapterSource",
    "ReconstructionError",
    "ReconstructionResult",
    "ReconstructionStats",
    "SafetensorsAdapterSource",
    "artifact_hashes_agree",
    "describe",
    "factorize",
    "pipeline_for",
    "reconstruct",
    "write_artifact",
]
