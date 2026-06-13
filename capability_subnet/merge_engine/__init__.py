"""Deterministic merge engine.

Turns a declarative recipe into a merged LoRA artifact, byte-identically on
every machine that runs the same software image.
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
