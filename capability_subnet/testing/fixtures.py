"""Reusable test fixtures for anyone building against this protocol.

Published as a pytest plugin rather than left in this repository's ``tests/``
directory, because the evaluation engine ships separately and both sides need the
same miniature pool. Two copies would drift, and a drifting fixture makes the two
suites test subtly different systems.

Enable with ``pytest_plugins = ["capability_subnet.testing"]`` in a conftest.

The objects themselves are built by :mod:`capability_subnet.testing.pool`, which
imports no test framework. Anything that needs the miniature pool outside a test
run imports from there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from capability_subnet.registry.adapters import AdapterRegistry
from capability_subnet.registry.base_model import BaseManifest
from capability_subnet.registry.snapshot import PoolSnapshot
from capability_subnet.testing.pool import (
    MAINTENANCE_WORKFLOW_ID,
    TINY_ADAPTERS,
    TINY_LAYERS,
    TINY_RANK,
    TINY_WIDTH,
    build_recipe,
    make_manifest,
    make_registry,
    make_results,
    make_snapshot,
    make_tensors,
    write_pool,
)

__all__ = [
    "MAINTENANCE_WORKFLOW_ID",
    "TINY_ADAPTERS",
    "TINY_LAYERS",
    "TINY_RANK",
    "TINY_WIDTH",
    "build_recipe",
    "make_manifest",
    "make_registry",
    "make_results",
    "make_snapshot",
    "make_tensors",
    "write_pool",
]


@pytest.fixture(scope="session")
def tiny_manifest() -> BaseManifest:
    return make_manifest()


@pytest.fixture(scope="session")
def tiny_registry() -> AdapterRegistry:
    return make_registry()


@pytest.fixture(scope="session")
def tiny_snapshot(tiny_registry: AdapterRegistry, tiny_manifest: BaseManifest) -> PoolSnapshot:
    return PoolSnapshot(registry=tiny_registry, manifest=tiny_manifest)


@pytest.fixture(scope="session")
def tiny_tensors(tiny_manifest: BaseManifest):
    return make_tensors(tiny_manifest)


@pytest.fixture
def tiny_source(tiny_tensors):
    from capability_subnet.merge_engine.loader import InMemoryAdapterSource

    return InMemoryAdapterSource(tiny_tensors)


@pytest.fixture
def tiny_pool_dir(tmp_path_factory, tiny_tensors, tiny_manifest) -> Path:
    return write_pool(tmp_path_factory.mktemp("pool"), tiny_tensors, tiny_manifest)


@pytest.fixture
def recipe_factory(tiny_snapshot: PoolSnapshot):
    """A callable that builds recipes against the tiny pool."""

    def _build(**kwargs):
        return build_recipe(tiny_snapshot, **kwargs)

    return _build


@pytest.fixture(scope="session")
def workflow():
    from capability_subnet.workflows import get_workflow

    return get_workflow(MAINTENANCE_WORKFLOW_ID)


@pytest.fixture
def instance(workflow):
    """One deterministic workflow instance."""
    return workflow.generate_instance(4242, split="hidden")
