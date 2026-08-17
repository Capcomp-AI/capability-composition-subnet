"""Permanent reference champions.

A plain king-of-the-hill contest only requires beating whoever holds the throne.
That is not a high enough bar for this commodity: at genesis the throne is empty,
and later a mediocre champion could hold it simply because nothing better
challenged. So the network keeps a set of references on the board permanently.

They are evaluated on the same hidden instances every window, alongside the
incumbent, and a challenger must beat the *strongest* of them. None of them can
be terminated and none of them earn emission — if a reference is the best thing
on the board, the workflow share is burned, because the network has not yet
produced anything worth paying for.

The set spans what a competent engineer would try before reaching for a search:
the untouched base model, the single best specialist, and three standard
equal-weight merges. If a miner cannot beat all of those, composition has not
added value and the correct outcome is that nobody gets paid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CompressionSpec, MergeSpec, OutputSpec, Recipe
from capability_subnet.registry.snapshot import PoolSnapshot

log = logging.getLogger(__name__)

ReferenceKind = Literal["base", "single_adapter", "recipe"]

#: Identifiers of the permanent references, in the order they are reported.
BASE_MODEL = "reference:base_model"
BEST_SINGLE = "reference:best_single_adapter"
EQUAL_LINEAR = "reference:equal_linear_merge"
EQUAL_TIES = "reference:equal_ties_svd_merge"
EQUAL_DARE_TIES = "reference:equal_dare_ties_svd_merge"
INCUMBENT = "incumbent"

#: Fixed seed for the stochastic reference merge. The reference must be the same
#: package every window, or the bar a challenger has to clear would drift.
REFERENCE_SEED = 1_000_003


@dataclass(frozen=True, slots=True)
class ReferencePackage:
    """One thing a challenger has to beat."""

    reference_id: str
    kind: ReferenceKind
    description: str
    adapter_id: str | None = None
    recipe: Recipe | None = None

    @property
    def is_evaluable(self) -> bool:
        return self.kind == "base" or self.adapter_id is not None or self.recipe is not None


def _equal_weight_recipe(
    snapshot: PoolSnapshot,
    adapters: list[str],
    *,
    combination_type: str,
    density: float | None = None,
    sign_method: str | None = None,
) -> Recipe:
    """An equal-weight merge over ``adapters``.

    Every coefficient is left at the implicit default of 1.0 and no layer group
    is singled out. That is the point: these references represent what you get
    without doing any composition research, so any tuning would make them a
    weaker bar than they are meant to be.

    Capped at the same number of adapters a miner may select. A reference built
    from more of the pool than any submission is allowed to use is not a bar a
    submission can clear — and on a pool larger than the cap it is not even a
    valid recipe, which is why a pool of more than twelve capability adapters
    could not be started against at all.

    Which twelve is decided by sorting on adapter id and taking the first: an
    arbitrary rule, but a fixed one that every engine reproduces and no operator
    chooses. Selecting by measured score instead would let the bar move whenever
    a certification landed.
    """
    return Recipe(
        workflow_id=snapshot.registry.workflow_id,
        base_revision=snapshot.manifest.revision,
        source_snapshot_sha256=snapshot.sha256,
        selected_adapters=sorted(adapters)[: C.MAX_SELECTED_ADAPTERS],
        merge=MergeSpec(
            combination_type=combination_type,
            density=density,
            majority_sign_method=sign_method,  # type: ignore[arg-type]
            random_seed=REFERENCE_SEED,
        ),
        compression=CompressionSpec(
            output_rank=snapshot.registry.canonical_rank, svd_clamp_quantile=1.0
        ),
        output=OutputSpec(adapter_name="reference"),
    )


def build_references(snapshot: PoolSnapshot) -> list[ReferencePackage]:
    """Every permanent reference for this pool."""
    capability_adapters = list(snapshot.registry.capability_adapters())

    references: list[ReferencePackage] = [
        ReferencePackage(
            reference_id=BASE_MODEL,
            kind="base",
            description="The pinned base model with no adapter applied.",
        )
    ]

    references.append(
        ReferencePackage(
            reference_id=EQUAL_LINEAR,
            kind="recipe",
            description="Equal-weight linear merge of every capability adapter.",
            recipe=_equal_weight_recipe(
                snapshot, capability_adapters, combination_type=C.MERGE_LINEAR
            ),
        )
    )
    references.append(
        ReferencePackage(
            reference_id=EQUAL_TIES,
            kind="recipe",
            description="Equal-weight trimmed merge with sign election.",
            recipe=_equal_weight_recipe(
                snapshot,
                capability_adapters,
                combination_type=C.MERGE_TIES_SVD,
                density=0.5,
                sign_method="total",
            ),
        )
    )
    references.append(
        ReferencePackage(
            reference_id=EQUAL_DARE_TIES,
            kind="recipe",
            description="Equal-weight drop-and-rescale merge with sign election.",
            recipe=_equal_weight_recipe(
                snapshot,
                capability_adapters,
                combination_type=C.MERGE_DARE_TIES_SVD,
                density=0.5,
                sign_method="total",
            ),
        )
    )
    return references


def single_adapter_references(snapshot: PoolSnapshot) -> list[ReferencePackage]:
    """One reference per capability adapter.

    Evaluated separately so the best single adapter can be identified rather than
    assumed. Which specialist is strongest on the whole workflow is exactly the
    kind of thing that is obvious in retrospect and wrong in advance.
    """
    return [
        ReferencePackage(
            reference_id=f"{BEST_SINGLE}:{adapter_id}",
            kind="single_adapter",
            description=f"The {adapter_id} adapter alone.",
            adapter_id=adapter_id,
        )
        for adapter_id in snapshot.registry.capability_adapters()
    ]


def collapse_single_adapters(
    scores: dict[str, float], *, include_incumbent: bool = True
) -> dict[str, float]:
    """Fold the per-adapter references into one ``best single adapter`` entry.

    A challenger has to beat the best specialist, not each one individually, so
    the reference set the comparator sees carries a single collapsed entry. The
    per-adapter numbers stay in the report, because which specialist was
    strongest is one of the more useful things the network learns each window.

    Args:
        include_incumbent: whether the reigning champion counts as a reference.
            False when picking the bar a challenger must clear by the absolute
            margin. The permanent references answer "did composition add value
            at all", and that question does not get harder because someone
            already answered it — folding the incumbent in made every successive
            champion clear the previous one by a further fixed margin, which
            walks the bar up until nothing can move it.
    """
    collapsed: dict[str, float] = {}
    singles: dict[str, float] = {}

    for name, value in scores.items():
        if name.startswith(BEST_SINGLE + ":"):
            singles[name] = value
        elif name == INCUMBENT and not include_incumbent:
            continue
        else:
            collapsed[name] = value

    if singles:
        collapsed[BEST_SINGLE] = max(singles.values())

    return collapsed


def is_reference(candidate_id: str) -> bool:
    """Whether a candidate identifier belongs to a permanent reference.

    References never earn emission. If one holds the throne the workflow share is
    burned, so this check is what stands between "no miner has beaten an
    equal-weight merge yet" and "the operator's own recipe is collecting emission".
    """
    return candidate_id.startswith("reference:")
