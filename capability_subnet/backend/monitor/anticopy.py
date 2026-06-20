"""Anti-copy.

Recipes are public — they have to be, or nobody could verify an evaluation. That
makes copying trivially easy, so the protocol makes it worthless rather than
impossible.

Three mechanisms do the work together:

* **Earliest commit wins.** Two identical recipes are one submission; the first
  hotkey to commit it is eligible and every later one scores zero. This is
  checked on the recipe digest at admission and again on the reconstructed
  artifact digest, because two differently-worded recipes that build the same
  bytes are the same package.
* **One shot per hotkey.** A hotkey gets one evaluation. Copying costs a
  registration and buys nothing.
* **Defender advantage.** Dethroning requires a real margin, so a copy that
  exactly reproduces the champion's scores loses by construction. This is
  enforced in the comparator rather than here, but it is what makes the other
  two sufficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from capability_subnet.backend.store import Store

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CopyVerdict:
    """Whether a submission duplicates one already on the board."""

    is_copy: bool
    detail: str
    original_hotkey: str | None = None
    original_block: int | None = None


def check_for_copy(
    store: Store, *, hotkey: str, recipe_sha256: str, first_block: int
) -> CopyVerdict:
    """Check a recipe digest against everything already admitted.

    A submission is a copy when an *earlier* commitment carries the same digest.
    Comparing blocks rather than arrival order matters: the engine may read the
    chain long after both commitments were made, and the chain's ordering is the
    only one either miner could have relied on.

    Ties on the same block resolve by hotkey, which is arbitrary but fixed and
    identical for every observer.
    """
    original = store.earliest_with_recipe(recipe_sha256)

    if original is None or original.hotkey == hotkey:
        return CopyVerdict(False, "recipe digest is unique among admitted submissions")

    if (original.first_block, original.hotkey) < (first_block, hotkey):
        return CopyVerdict(
            True,
            f"identical recipe was committed at block {original.first_block} by "
            f"{original.hotkey[:12]}…; the earliest valid commitment is eligible",
            original_hotkey=original.hotkey,
            original_block=original.first_block,
        )

    return CopyVerdict(False, "this is the earliest commitment of the recipe")


def check_artifact_copy(
    store: Store, *, hotkey: str, artifact_sha256: str, first_block: int
) -> CopyVerdict:
    """Check a reconstructed artifact against everything already built.

    Run after reconstruction, because it catches what the recipe check cannot: a
    recipe that reorders its adapter list, renames its output or picks a
    different random seed for a method that ignores randomness produces different
    recipe bytes and identical weights. The artifact digest sees through all of
    that.
    """
    original = store.earliest_with_artifact(artifact_sha256)

    if original is None or original.hotkey == hotkey:
        return CopyVerdict(False, "reconstructed artifact is unique")

    if (original.first_block, original.hotkey) < (first_block, hotkey):
        return CopyVerdict(
            True,
            f"reconstructs to the same artifact as {original.hotkey[:12]}…, committed "
            f"earlier at block {original.first_block}",
            original_hotkey=original.hotkey,
            original_block=original.first_block,
        )

    return CopyVerdict(False, "this is the earliest commitment producing this artifact")


def is_champion_artifact(store: Store, artifact_sha256: str) -> bool:
    """Whether a challenger rebuilt exactly the champion's package.

    Not a rejection on its own — the comparator will fail it anyway, since an
    identical package cannot beat itself by a margin. Detecting it explicitly
    lets the engine skip a full evaluation and say so in the report, which is
    both cheaper and clearer than letting it run and reporting a statistical
    tie.
    """
    champion = store.get_champion()
    if champion is None or not champion.artifact_sha256:
        return False
    return champion.artifact_sha256 == artifact_sha256
