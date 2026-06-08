"""Certified base model and source-adapter registry."""

from capability_subnet.registry.adapters import (
    AdapterEntry,
    AdapterRegistry,
    RegistryError,
    load_registry,
)
from capability_subnet.registry.base_model import (
    BaseManifest,
    BaseManifestError,
    load_base_manifest,
    require_pinned,
)
from capability_subnet.registry.snapshot import PoolSnapshot, current_snapshot_sha256, load_snapshot

__all__ = [
    "AdapterEntry",
    "AdapterRegistry",
    "BaseManifest",
    "BaseManifestError",
    "PoolSnapshot",
    "RegistryError",
    "current_snapshot_sha256",
    "load_base_manifest",
    "load_registry",
    "load_snapshot",
    "require_pinned",
]
