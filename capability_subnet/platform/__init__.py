"""Platform services around the evaluation engine.

Object storage for published artifacts and reports, the compatibility history and
its analyses, and the public dashboard.

The read-only HTTP interface lives in ``capability_subnet.backend.api`` rather
than here: it serves engine state directly and has no business being a second
process with its own view of the database.
"""

from capability_subnet.platform.compatibility_graph import GraphSummary, build_graph, render_summary
from capability_subnet.platform.storage import LocalObjectStore, S3ObjectStore, StorageError

__all__ = [
    "GraphSummary",
    "LocalObjectStore",
    "S3ObjectStore",
    "StorageError",
    "build_graph",
    "render_summary",
]
