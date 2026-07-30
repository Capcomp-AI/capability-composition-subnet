"""A naive starting recipe.

Draws a random valid recipe from the pool. It is a working starting point and
nothing more — no attempt is made to choose adapters that belong together, to
tune coefficients, or to set per-layer-group overrides.

Finding compositions that score is the competition. Replace this.
"""

from __future__ import annotations

import random

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import Recipe
from capability_subnet.miner.recipe import new_recipe
from capability_subnet.registry.snapshot import PoolSnapshot, load_snapshot

#: Merge methods a baseline draws from.
METHODS = (C.MERGE_LINEAR, C.MERGE_TIES_SVD, C.MERGE_DARE_TIES_SVD)


def random_recipe(
    seed: int = 0,
    *,
    adapter_count: int = 4,
    snapshot: PoolSnapshot | None = None,
) -> Recipe:
    """One random valid recipe.

    Args:
        seed: chooses the draw. The same seed gives the same recipe.
        adapter_count: how many adapters to combine, clamped to the pool bounds.
        snapshot: the pool to draw from. Defaults to the shipped one.

    Returns:
        A recipe that passes validation. Whether it scores well is another matter.
    """
    pool = snapshot or load_snapshot()
    selectable = list(pool.registry.selectable_ids)

    count = max(C.MIN_SELECTED_ADAPTERS, min(adapter_count, len(selectable)))
    rng = random.Random(seed)

    adapters = sorted(rng.sample(selectable, count))
    method = rng.choice(METHODS)

    return new_recipe(
        adapters,
        combination_type=method,
        density=None if method == C.MERGE_LINEAR else round(rng.uniform(0.3, 0.8), 3),
        majority_sign_method=None if method == C.MERGE_LINEAR else "total",
        global_weights={a: round(rng.uniform(0.4, 1.0), 3) for a in adapters},
        snapshot=pool,
    )
