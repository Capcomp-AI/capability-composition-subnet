"""Permanent reference champions a challenger must also clear."""

from capability_subnet.backend.baselines.references import (
    ReferencePackage,
    build_references,
    collapse_single_adapters,
    is_reference,
    owner_reference_recipe,
    single_adapter_references,
)

__all__ = [
    "ReferencePackage",
    "build_references",
    "collapse_single_adapters",
    "is_reference",
    "owner_reference_recipe",
    "single_adapter_references",
]
