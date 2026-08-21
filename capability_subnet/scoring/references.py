"""The permanent reference the network measures against.

A plain king-of-the-hill contest only requires beating whoever holds the throne.
That is not a high enough bar for this commodity: at genesis the throne is
empty, and later a mediocre champion could hold it simply because nothing better
challenged. So the network keeps the untouched base model on the board
permanently. It is evaluated on the same hidden instances every window,
alongside the incumbent, and a challenger must beat it. It cannot be terminated
and it earns no emission — if the base model is the best thing on the board, the
workflow share is burned, because the network has not yet produced anything
worth paying for.

The set used to be wider: the base model, the best single adapter, and three
standard equal-weight merges. Measuring them retired all but the base. Over a
full 1350-instance window the base model scored 0.1133 end-to-end and every
other reference scored below it — the equal-weight TIES merge 0.0926, the
equal-weight linear merge 0.0000, single adapters 0.1067 and 0.0815. None ever
bound, so each spent a card every window raising a bar the base had already
raised higher.

The linear merge is worth recording plainly. Linear aggregation sums the
weighted updates without normalising, so ten adapters at the implicit
coefficient of 1.0 is a tenfold sum rather than an average. It served without
error over all 1350 instances and answered in fragments — "By eighty For By
ByxFE ByxFE" — scoring zero end-to-end and 0/40 on the retention probe. It was
not what a competent engineer would try before reaching for a search; it was a
configuration nobody would ship.

What this gives up, recorded so it is not rediscovered as a surprise:

- A recipe identical to a naive equal-weight merge used to be terminated for
  matching a reference. It is now an ordinary candidate that has to beat the
  base by the margin. It still loses on measurement, but it loses on its score
  rather than on its identity.
- The single-adapter references are the bar that actually bound, and they are
  gone. On a 250-item paired benchmark over this pool the best single adapter
  scored 0.132 against the base model's 0.100 — the only reference measured to
  beat the base — and no merge has beaten it. Removing it lowers the bar from
  "beat the best specialist" to "beat the untouched model", which is a
  materially easier thing to do.

  Concretely, on the run-411 draw the base scored 0.1133 and the strongest
  challenger 0.1496. Scaling the benchmark's best single adapter to that draw
  puts it near 0.150, so that challenger clears the base by a wide margin and
  would not clear the specialist at all. It is eligible because this reference
  was removed.

  The specialist is exactly what the reference set was for: a merge worse than
  one adapter used alone is composition with negative value, and it can now
  take the throne and earn emission. This was removed knowingly, for the card
  time, and it is the first thing to restore if the network starts paying for
  merges nobody would deploy.

The remedy is to measure the pool's twenty-seven adapters once and append a
reference here for anything that clears the base.
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


def build_references(snapshot: PoolSnapshot) -> list[ReferencePackage]:
    """Every permanent reference for this pool.

    One, now. Kept as a list because the audit path, the disclosure and the
    engine all iterate it, and because restoring a reference should be a matter
    of appending here rather than reshaping its callers.
    """
    return [
        ReferencePackage(
            reference_id=BASE_MODEL,
            kind="base",
            description="The pinned base model with no adapter applied.",
        )
    ]


def bar_scores(
    scores: dict[str, float], *, include_incumbent: bool = True
) -> dict[str, float]:
    """The reference scores a bar is taken from.

    Was ``collapse_single_adapters``, and folded the per-adapter references into
    one "best single adapter" entry before applying the rule below. There are no
    per-adapter references left to fold, so only the rule remains.

    Args:
        include_incumbent: whether the reigning champion counts as a reference.
            False when picking the bar a challenger must clear by the absolute
            margin. The permanent reference answers "did composition add value
            at all", and that question does not get harder because someone
            already answered it — folding the incumbent in made every successive
            champion clear the previous one by a further fixed margin, which
            walks the bar up until nothing can move it.
    """
    if include_incumbent:
        return dict(scores)
    return {name: value for name, value in scores.items() if name != INCUMBENT}


def is_reference(candidate_id: str) -> bool:
    """Whether a candidate identifier belongs to a permanent reference.

    References never earn emission. If one holds the throne the workflow share is
    burned, so this check is what stands between "no miner has beaten an
    equal-weight merge yet" and "the operator's own recipe is collecting emission".
    """
    return candidate_id.startswith("reference:")
