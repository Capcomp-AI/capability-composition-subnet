"""What each merge method does, as description rather than as arithmetic.

Separate from the implementation because the published contract needs it and the
implementation needs torch. A validator installs neither the tensor stack nor a
GPU, and it still has to be able to read and check a contract that names every
method and its stages — so this module imports nothing heavier than the protocol
constants.

The arithmetic that carries these names out lives in
:mod:`capability_subnet.merge_engine.methods`, which imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from capability_subnet.common import constants as C

SparsifyMode = Literal["none", "magnitude", "random_rescale"]
SignMode = Literal["none", "total", "frequency"]
AggregateMode = Literal["sum", "disjoint_mean"]


@dataclass(frozen=True, slots=True)
class MergePipeline:
    """The resolved stage configuration for one merge method."""

    sparsify: SparsifyMode
    sign_election: SignMode
    aggregate: AggregateMode
    factor_space: bool
    description: str


#: Each allowed method name resolved to its pipeline.
PIPELINES: dict[str, MergePipeline] = {
    C.MERGE_LINEAR: MergePipeline(
        sparsify="none",
        sign_election="none",
        aggregate="sum",
        factor_space=True,
        description=(
            "Weighted sum performed on the factors themselves. The cheapest path: "
            "no full update is ever materialised. Because the factors are summed "
            "rather than their products, the result carries cross terms between "
            "adapters, which is what distinguishes it from the delta-space sum."
        ),
    ),
    C.MERGE_SVD: MergePipeline(
        sparsify="none",
        sign_election="none",
        aggregate="sum",
        factor_space=False,
        description=(
            "Exact weighted sum of effective updates, then a rank reduction back "
            "to the declared output rank."
        ),
    ),
    # Deliberately the same pipeline as `svd`. Concatenating factorisations is
    # algebraically the sum of the updates, so the two cannot produce different
    # weights — and because the engine now decomposes exactly this concatenation
    # rather than a materialised sum, they no longer even differ in rounding.
    #
    # Kept as an accepted spelling rather than removed, so recipes written
    # against the published contract stay valid. `canonical_method` folds it
    # into `svd` before anything hashes a recipe, which is what stops two miners
    # who independently chose different names for the same package from
    # colliding on the artifact digest and having the later one terminated for
    # copying work it never saw.
    C.MERGE_CAT_SVD: MergePipeline(
        sparsify="none",
        sign_election="none",
        aggregate="sum",
        factor_space=False,
        description=(
            "Concatenates the adapters' factorisations and reduces the result. "
            "An accepted alias of `svd`: concatenating factors is algebraically "
            "the sum of the updates, so both names describe one package."
        ),
    ),
    C.MERGE_TIES_SVD: MergePipeline(
        sparsify="magnitude",
        sign_election="total",
        aggregate="disjoint_mean",
        factor_space=False,
        description=(
            "Trims each update to its largest-magnitude entries, elects one sign "
            "per entry, and averages only the adapters that agree with it."
        ),
    ),
    C.MERGE_DARE_TIES_SVD: MergePipeline(
        sparsify="random_rescale",
        sign_election="total",
        aggregate="disjoint_mean",
        factor_space=False,
        description=(
            "Drops a random subset of each update and rescales the survivors so "
            "the expected update is preserved, then elects signs and averages "
            "the agreeing adapters."
        ),
    ),
    C.MERGE_DARE_LINEAR_SVD: MergePipeline(
        sparsify="random_rescale",
        sign_election="none",
        aggregate="sum",
        factor_space=False,
        description="Random drop-and-rescale followed by a plain weighted sum.",
    ),
    C.MERGE_MAGNITUDE_PRUNE_SVD: MergePipeline(
        sparsify="magnitude",
        sign_election="none",
        aggregate="sum",
        factor_space=False,
        description="Magnitude trimming followed by a plain weighted sum.",
    ),
}


#: Merge names that describe the same package, folded to one spelling.
METHOD_ALIASES: dict[str, str] = {C.MERGE_CAT_SVD: C.MERGE_SVD}


def canonical_method(method: str) -> str:
    """The canonical spelling of ``method``.

    Applied before a recipe is hashed so that two recipes describing the same
    merge share a digest. Without it the anti-copy check sees them as distinct
    submissions that happen to build identical bytes, and terminates whichever
    arrived second for copying.

    Here rather than beside the pipelines because a recipe is validated far more
    often than it is merged, and validating must not need a merge engine.
    Folding a name through ``merge_engine.methods`` pulled in torch by way of
    the package, so ``capcomp init`` - the first command in the miner guide -
    failed on the install the guide documents with ``No module named 'torch'``.
    """
    return METHOD_ALIASES.get(method, method)
