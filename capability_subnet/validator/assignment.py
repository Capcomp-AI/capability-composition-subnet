"""Which instances a particular validator measures.

Every validator evaluating every instance is the obvious design and the wrong
one. A window is around seventeen GPU-hours; replicating it per validator
multiplies the network's cost by the number of validators while producing the
same numbers, and prices the role at a 48 GB card running most of the day. That
does not decentralise the role, it just makes it expensive enough that few can
hold it.

The split here buys both properties for a fraction of the work:

* a **core** every validator measures, identical for all of them. This is what
  makes validators comparable: two honest validators on different hardware
  should land within the cross-device band on exactly these instances, and one
  that does not is visible without anybody re-running its work.
* a **tail** unique to each validator, drawn from what the core left. Coverage
  of the window grows with the number of validators rather than being repeated,
  and no miner can know which validator will see which instances.

Both parts derive from public material — the beacon and the validator's hotkey —
so any party can recompute exactly what a given validator was supposed to
measure and check it did. There is nothing private in the assignment, which is
the point: the only thing a validator can do unobserved is compute a wrong
answer, and the core is what catches that.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

#: Share of a window every validator measures. Sized so two validators share
#: enough paired instances for their disagreement to mean something: the paired
#: comparison needs samples, and a core far below this makes "these two disagree"
#: indistinguishable from ordinary sampling noise.
DEFAULT_CORE_FRACTION: float = 0.25

#: Share of a window each validator measures alone, drawn from outside the core.
DEFAULT_TAIL_FRACTION: float = 0.15


@dataclass(frozen=True, slots=True)
class Assignment:
    """The instances one validator is accountable for this window."""

    core: tuple[int, ...]
    tail: tuple[int, ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        """Everything this validator measures, in a stable order."""
        return tuple(sorted({*self.core, *self.tail}))

    def __len__(self) -> int:
        return len(self.seeds)


def _rng(*parts: object) -> random.Random:
    material = "|".join(str(p) for p in parts).encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def core_instances(
    seeds: tuple[int, ...] | list[int],
    *,
    beacon: str,
    fraction: float = DEFAULT_CORE_FRACTION,
) -> tuple[int, ...]:
    """The instances every validator measures.

    Keyed on the beacon alone — no hotkey — so it is the same set for everyone
    and nobody can be handed an easier core than anybody else.
    """
    ordered = sorted(seeds)
    if not ordered:
        return ()
    count = max(1, round(len(ordered) * _checked_fraction(fraction, "core")))
    return tuple(sorted(_rng("core", beacon).sample(ordered, min(count, len(ordered)))))


def assign(
    seeds: tuple[int, ...] | list[int],
    *,
    hotkey: str,
    beacon: str,
    core_fraction: float = DEFAULT_CORE_FRACTION,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> Assignment:
    """Split a window's instances into this validator's core and tail.

    Args:
        hotkey: the validator's hotkey. Only the tail depends on it, so a
            validator cannot shop for a favourable core by changing keys.
    """
    ordered = sorted(seeds)
    core = core_instances(ordered, beacon=beacon, fraction=core_fraction)

    remaining = [s for s in ordered if s not in set(core)]
    tail_count = round(len(ordered) * _checked_fraction(tail_fraction, "tail"))
    tail_count = min(tail_count, len(remaining))
    tail = (
        tuple(sorted(_rng("tail", beacon, hotkey).sample(remaining, tail_count)))
        if tail_count
        else ()
    )
    return Assignment(core=core, tail=tail)


def _checked_fraction(value: float, name: str) -> float:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name}_fraction must be in (0, 1], got {value}")
    return value


def coverage(assignments: list[Assignment], total: int) -> float:
    """Share of a window measured by at least one validator.

    The number worth watching as validators join: the core is repeated by
    everyone and does not grow, so this rising above the core fraction is the
    only evidence that adding validators is buying coverage rather than
    duplicating it.
    """
    if total <= 0:
        return 0.0
    seen: set[int] = set()
    for a in assignments:
        seen.update(a.seeds)
    return len(seen) / total
