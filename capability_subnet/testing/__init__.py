"""The miniature pool, and the pytest fixtures that wrap it.

Split so that importing the builders does not import a test framework. Anything
running outside a test — seeding a store, reproducing a merge, driving the engine
headlessly — imports from here directly and needs nothing else installed.

For the fixtures, add to a conftest:

    pytest_plugins = ["capability_subnet.testing.fixtures"]
"""

from __future__ import annotations

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
