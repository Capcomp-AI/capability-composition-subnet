"""Per-window hidden instance draws.

Each window draws a fresh set of hidden instances and a fresh out-of-distribution
set. Both are derived from a secret root the operator holds, so:

* nobody outside the engine can predict which instances a window will use,
* the draw is reproducible inside the engine, so a disputed evaluation can be
  replayed exactly,
* a candidate cannot be tuned to a fixed test set, because the set moves every
  window and the incumbent is re-measured on the new one.

Refreshing per window is what replaces the heavier commit-and-reveal machinery a
round-based design would need. There is nothing to reveal: the instances simply
did not exist in a form anyone could see before the window opened.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Range instance seeds are drawn from. Wide enough that the same instance
#: recurring across windows is effectively impossible.
SEED_SPACE = 2**48


@dataclass(frozen=True, slots=True)
class WindowSample:
    """The instances one window is decided on."""

    window_id: int
    hidden_seeds: tuple[int, ...]
    ood_seeds: tuple[int, ...]

    @property
    def total(self) -> int:
        return len(self.hidden_seeds) + len(self.ood_seeds)


def _derive(root: int, window_id: int, label: str) -> random.Random:
    """A generator keyed on the secret root, the window and a label.

    Hashing rather than arithmetic on the root: an attacker who learned one
    window's seeds must not be able to walk backwards to the root or forwards to
    the next window's draw.
    """
    material = f"{root}|{window_id}|{label}".encode()
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest, "big"))


def draw_window(
    window_id: int,
    *,
    root: int,
    hidden_count: int,
    ood_count: int,
) -> WindowSample:
    """Draw the hidden and out-of-distribution seeds for one window."""
    if root == 0:
        log.warning(
            "drawing window %s from the default seed root; hidden instances are "
            "predictable and this deployment must not be used on a live network",
            window_id,
        )

    hidden_rng = _derive(root, window_id, "hidden")
    ood_rng = _derive(root, window_id, "ood")

    return WindowSample(
        window_id=window_id,
        hidden_seeds=tuple(_distinct(hidden_rng, hidden_count)),
        ood_seeds=tuple(_distinct(ood_rng, ood_count)),
    )


def _distinct(rng: random.Random, count: int) -> list[int]:
    seen: set[int] = set()
    while len(seen) < count:
        seen.add(rng.randrange(1, SEED_SPACE))
    return sorted(seen)


def build_instances(sample: WindowSample, workflow) -> tuple[list, list]:
    """Materialise a window's instances.

    Returns:
        ``(hidden, ood)`` instance lists, in seed order so two workers building
        the same window iterate them identically.
    """
    hidden = [workflow.generate_instance(seed, split="hidden") for seed in sample.hidden_seeds]
    ood = [workflow.generate_instance(seed, split="ood") for seed in sample.ood_seeds]
    return hidden, ood


def common_instance_ids(*results_sets: list) -> list[str]:
    """Instance identifiers every candidate in ``results_sets`` has a valid row for.

    The comparator is paired: it compares two packages on the *same* instances,
    not on their respective averages. An instance where either side hit a harness
    failure is dropped from both, because a difference measured on different
    problems is not a difference in capability.
    """
    if not results_sets:
        return []

    common: set[str] | None = None
    for results in results_sets:
        ids = {row.instance_id for row in results if row.is_valid_sample}
        common = ids if common is None else (common & ids)
    return sorted(common or set())
