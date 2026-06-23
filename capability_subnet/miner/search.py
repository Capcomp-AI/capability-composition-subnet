"""A reference composition search.

This is a starting point, not a strategy. It enumerates a coarse grid over the
recipe space so a miner can get a first candidate without writing anything, and
so the shape of the search problem is concrete rather than abstract.

Nobody wins with this. The search space is large — adapter subsets times merge
methods times densities times coefficients times per-group overrides times ranks
— and the interesting structure is in the interactions between adapters, which a
grid does not see. Finding that structure is the competition; this module just
makes the first hour cheap.

Search happens entirely on the miner's own hardware. The engine never sees it and
does not care how a recipe was found.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import Recipe
from capability_subnet.miner.recipe import new_recipe
from capability_subnet.registry.snapshot import PoolSnapshot, load_snapshot

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchSpace:
    """The grid a coarse search walks."""

    methods: tuple[str, ...] = (
        C.MERGE_LINEAR,
        C.MERGE_TIES_SVD,
        C.MERGE_DARE_TIES_SVD,
        C.MERGE_MAGNITUDE_PRUNE_SVD,
    )
    densities: tuple[float, ...] = (0.3, 0.5, 0.7)
    output_ranks: tuple[int, ...] = (32, 64)
    clamp_quantiles: tuple[float, ...] = (1.0, 0.99)
    emphasis_weights: tuple[float, ...] = (1.0, 1.15)
    #: Adapters worth emphasising first. Structural stages gate end-to-end
    #: success, so they are the cheapest place to look for a first improvement.
    emphasis_candidates: tuple[str, ...] = (
        "strict-json-v1",
        "tool-calling-v1",
        "safety-policy-v1",
    )
    seeds: tuple[int, ...] = (0,)


@dataclass(slots=True)
class SearchResult:
    """One evaluated point."""

    recipe: Recipe
    score: float
    detail: str = ""


@dataclass(slots=True)
class SearchReport:
    """What a search run found."""

    evaluated: int = 0
    best: SearchResult | None = None
    history: list[SearchResult] = field(default_factory=list)

    def record(self, result: SearchResult) -> None:
        self.evaluated += 1
        self.history.append(result)
        if self.best is None or result.score > self.best.score:
            self.best = result
            log.info("new best: %.4f (%s)", result.score, result.detail or "no detail")


def default_adapter_sets(snapshot: PoolSnapshot | None = None) -> list[list[str]]:
    """A few adapter subsets worth trying first.

    The full subset lattice over a twelve-adapter pool has four thousand members,
    most of which are obviously bad. These four are the ones a competent engineer
    would try before searching: everything, the capability adapters, the
    capability adapters without the retention anchor, and the structural core.
    """
    pool = snapshot or load_snapshot()
    capability = list(pool.registry.capability_adapters())

    without_retention = [a for a in capability if a != "general-reasoning-retention-v1"]
    structural = [
        adapter
        for adapter in capability
        if adapter
        in {
            "strict-json-v1",
            "tool-calling-v1",
            "safety-policy-v1",
            "german-technical-v1",
            "general-reasoning-retention-v1",
        }
    ]

    return [capability, without_retention, structural, list(pool.adapter_ids)]


def enumerate_recipes(
    adapter_sets: list[list[str]],
    space: SearchSpace | None = None,
    snapshot: PoolSnapshot | None = None,
) -> Iterator[Recipe]:
    """Walk the grid.

    Yields lazily: the full grid over four adapter sets is in the thousands, and
    a miner will usually want to stop early rather than materialise all of it.
    """
    space = space or SearchSpace()
    pool = snapshot or load_snapshot()

    for adapters, method, output_rank, clamp, seed in itertools.product(
        adapter_sets, space.methods, space.output_ranks, space.clamp_quantiles, space.seeds
    ):
        needs_density = method in C.DENSITY_METHODS
        densities: tuple[float | None, ...] = space.densities if needs_density else (None,)
        sign_method = "total" if method in (C.MERGE_TIES_SVD, C.MERGE_DARE_TIES_SVD) else None

        for density in densities:
            for emphasis in space.emphasis_weights:
                weights = (
                    {
                        adapter: emphasis
                        for adapter in space.emphasis_candidates
                        if adapter in adapters
                    }
                    if emphasis != 1.0
                    else {}
                )
                yield new_recipe(
                    adapters,
                    combination_type=method,
                    density=density,
                    majority_sign_method=sign_method,
                    random_seed=seed,
                    global_weights=weights,
                    output_rank=output_rank,
                    svd_clamp_quantile=clamp,
                    snapshot=pool,
                )


def coarse_search(
    score_recipe: Callable[[Recipe], float],
    *,
    adapter_sets: list[list[str]] | None = None,
    space: SearchSpace | None = None,
    budget: int = 24,
    snapshot: PoolSnapshot | None = None,
    shuffle_seed: int | None = None,
) -> SearchReport:
    """Evaluate up to ``budget`` grid points.

    Args:
        score_recipe: the miner's own objective. Local end-to-end completion is
            the obvious one, but it is noisy on small instance counts; a miner
            with limited compute usually does better scoring a cheaper proxy
            first and only running the full workflow on the survivors.
        shuffle_seed: randomise the order before truncating to the budget. A
            grid walked in order spends its whole budget varying the last
            dimension, which is rarely the informative one.
    """
    report = SearchReport()
    candidates = list(
        enumerate_recipes(adapter_sets or default_adapter_sets(snapshot), space, snapshot)
    )

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(candidates)

    log.info("coarse search over %d points, evaluating %d", len(candidates), budget)

    for recipe in candidates[:budget]:
        try:
            score = score_recipe(recipe)
        except Exception as exc:  # noqa: BLE001 - one bad point must not end the search
            log.warning("scoring failed for %s: %s", recipe.digest()[:19], exc)
            continue

        report.record(
            SearchResult(
                recipe=recipe,
                score=score,
                detail=(
                    f"{recipe.merge.combination_type} density={recipe.merge.density} "
                    f"rank={recipe.compression.output_rank} "
                    f"adapters={len(recipe.selected_adapters)}"
                ),
            )
        )

    return report


def refine_layer_groups(
    base: Recipe,
    score_recipe: Callable[[Recipe], float],
    *,
    adapters: list[str],
    groups: list[str] | None = None,
    multipliers: tuple[float, ...] = (0.85, 1.15),
    snapshot: PoolSnapshot | None = None,
) -> SearchReport:
    """Vary one adapter's coefficient in one layer group at a time.

    Coordinate descent over the per-group coefficients, starting from a recipe a
    coarse search already liked. This is where the per-layer structure the
    network is meant to discover actually lives: whether a domain adapter belongs
    early, whether a code adapter belongs late, and which pairs interfere at
    which depth.
    """
    pool = snapshot or load_snapshot()
    group_names = groups or list(sorted(pool.manifest.layer_groups))
    report = SearchReport()

    report.record(SearchResult(recipe=base, score=score_recipe(base), detail="starting point"))

    for adapter in adapters:
        for group in group_names:
            for multiplier in multipliers:
                assert report.best is not None
                current = report.best.recipe
                overrides = {
                    name: dict(values)
                    for name, values in current.layer_group_overrides.items()
                }
                overrides.setdefault(group, {})

                proposed = current.effective_weight(adapter, group) * multiplier
                if not (C.ADAPTER_WEIGHT_MIN <= proposed <= C.ADAPTER_WEIGHT_MAX):
                    continue
                overrides[group][adapter] = round(proposed, 4)

                candidate = new_recipe(
                    current.selected_adapters,
                    combination_type=current.merge.combination_type,
                    density=current.merge.density,
                    majority_sign_method=current.merge.majority_sign_method,
                    random_seed=current.merge.random_seed,
                    global_weights=dict(current.global_weights),
                    layer_group_overrides=overrides,
                    output_rank=current.compression.output_rank,
                    svd_clamp_quantile=current.compression.svd_clamp_quantile,
                    snapshot=pool,
                )

                try:
                    score = score_recipe(candidate)
                except Exception as exc:  # noqa: BLE001
                    log.warning("scoring failed: %s", exc)
                    continue

                report.record(
                    SearchResult(
                        recipe=candidate,
                        score=score,
                        detail=f"{adapter} in {group} ×{multiplier}",
                    )
                )

    return report
