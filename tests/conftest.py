"""Test configuration.

The miniature-pool fixtures live in ``capability_subnet.testing`` so the
separately-shipped evaluation engine can use the identical ones. Only fixtures
that are specific to this repository's suite belong here.
"""

from __future__ import annotations

pytest_plugins = ["capability_subnet.testing.fixtures"]
