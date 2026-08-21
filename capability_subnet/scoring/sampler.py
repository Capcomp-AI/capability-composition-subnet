"""Per-run hidden instance draws.

Each run draws a fresh set of hidden instances and a fresh out-of-distribution
set. Both are derived from a secret root the operator holds, so:

* nobody outside the engine can predict which instances a run will use,
* the draw is reproducible inside the engine, so a disputed evaluation can be
  replayed exactly,
* a candidate cannot be tuned to a fixed test set, because the set moves every
  run and the incumbent is re-measured on the new one.

Refreshing per run defends against a *miner* tuning to a fixed set. On its own
it does not defend against the operator, and the difference matters.

Seeds derive deterministically from ``(root, run_id)``. If the root were the
operator's private choice and nothing bound them to it, they could try many roots
for one run, keep whichever draw suited a candidate they had already seen, and
publish the resulting seeds. A replay would verify that the published instances
match the published seeds and still miss it entirely, because what was chosen was
the *seeds*.

Two things close that, and both are needed:

* **The root is committed.** ``root_commitment`` is a hash of the root, published
  in the contract and in every disclosure. One root produces every run, so an
  operator cannot re-grind per run without the commitment changing — and a
  changing commitment is visible to everyone. Revealing the root at any point
  lets an auditor recompute every historical run at once.
* **The draw is bound to a value the operator does not choose.** Each run mixes
  in a public ``beacon`` — the hash of the block the run opened at. It is not
  known until the run opens, so the draw cannot be precomputed, and it is
  publicly checkable, so a fabricated one is caught.

With both, choosing the draw requires either breaking the commitment or choosing
a block hash.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Range instance seeds are drawn from. Wide enough that the same instance
#: recurring across runs is effectively impossible.
SEED_SPACE = 2**48


@dataclass(frozen=True, slots=True)
class RunSample:
    """The instances one run is decided on."""

    run_id: int
    hidden_seeds: tuple[int, ...]
    ood_seeds: tuple[int, ...]
    #: Seed for this run's general-capability probe. Derived from the same
    #: secret root under its own label, so learning the probe reveals nothing
    #: about the hidden instances and vice versa.
    probe_seed: int = 0
    #: The public value this draw was bound to — the hash of the block the run
    #: opened at. Published so anyone can check the draw against the chain.
    beacon: str = ""
    #: Hash of the seed root. Constant across every run of a deployment.
    root_commitment: str = ""

    @property
    def total(self) -> int:
        return len(self.hidden_seeds) + len(self.ood_seeds)


def root_commitment(root: int) -> str:
    """The public commitment to a seed root.

    Published in the contract and in every disclosure. It reveals nothing about
    the root — it is a hash — but it binds the operator to one: every run must
    derive from the root behind this value, so a commitment that changes between
    runs is an operator changing the draw, in public.
    """
    return "sha256:" + hashlib.sha256(f"capsub-seed-root|{root}".encode()).hexdigest()


def _derive(root: int, run_id: int, label: str, beacon: str = "") -> random.Random:
    """A generator keyed on the secret root, the run, a label and the beacon.

    Hashing rather than arithmetic on the root: an attacker who learned one
    run's seeds must not be able to walk backwards to the root or forwards to
    the next run's draw.

    The beacon is the public part. Without it the operator alone decides the draw;
    with it they would also have to choose the block hash it comes from.
    """
    material = f"{root}|{run_id}|{label}|{beacon}".encode()
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest, "big"))


def draw_run(
    run_id: int,
    *,
    root: int,
    hidden_count: int,
    ood_count: int,
    beacon: str = "",
) -> RunSample:
    """Draw the hidden and out-of-distribution seeds for one run.

    Args:
        beacon: a public value the operator does not choose — the hash of the
            block this run opened at. Empty means the draw rests on the
            operator's root alone, which is a weaker guarantee and is logged.
    """
    if not beacon:
        log.warning(
            "run %s drawn with no beacon; the draw rests entirely on the "
            "operator's secret root and cannot be shown to be unchosen",
            run_id,
        )
    if root == 0:
        log.warning(
            "drawing run %s from the default seed root; hidden instances are "
            "predictable and this deployment must not be used on a live network",
            run_id,
        )

    hidden_rng = _derive(root, run_id, "hidden", beacon)
    ood_rng = _derive(root, run_id, "ood", beacon)

    return RunSample(
        run_id=run_id,
        hidden_seeds=tuple(_distinct(hidden_rng, hidden_count)),
        ood_seeds=tuple(_distinct(ood_rng, ood_count)),
        probe_seed=_derive(root, run_id, "probe", beacon).randrange(1, SEED_SPACE),
        beacon=beacon,
        root_commitment=root_commitment(root),
    )


def _derive_open(run_id: int, label: str, beacon: str) -> random.Random:
    """A generator keyed on nothing anyone holds privately.

    Same construction as ``_derive`` with the root removed, so the material is
    entirely public and every party computes byte-identical seeds.
    """
    material = f"open|{run_id}|{label}|{beacon}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def draw_run_open(
    run_id: int,
    *,
    beacon: str,
    hidden_count: int,
    ood_count: int,
) -> RunSample:
    """Draw a run from the beacon alone, with no operator secret.

    The secret root exists to stop a miner tuning to the instances. It buys that
    at the cost of an operator who alone knows the draw, and every other party
    reduced to checking that published instances match published seeds — which
    cannot detect a draw that was *chosen*, only one that was misreported. The
    commitment and the beacon narrow that, but they do not remove it, and they
    only work if somebody is watching.

    Removing the root removes the operator. Seeds become a pure function of the
    hash of the block the run opened at, so every validator derives the same
    instances independently, with nothing to trust and nothing to publish.

    What replaces the secret is ordering, not concealment: a miner cannot tune to
    a draw it cannot see *at the time it must commit*. The beacon is the hash of
    the opening block, unknown until that block exists, and the commitment being
    evaluated is the one standing at that block. Learning the instances a moment
    later is worth nothing, because the recipe is already fixed and one recipe
    per hotkey is final.

    This is the stronger guarantee of the two. The secret-root draw is unverifiable
    by construction and merely watched; this one is checkable by anyone holding a
    block hash, including someone auditing a run years later.
    """
    if not beacon:
        raise ValueError(
            "an open draw has no secret to fall back on, so an empty beacon would "
            "make every run of every deployment draw the same instances"
        )

    return RunSample(
        run_id=run_id,
        hidden_seeds=tuple(_distinct(_derive_open(run_id, "hidden", beacon), hidden_count)),
        ood_seeds=tuple(_distinct(_derive_open(run_id, "ood", beacon), ood_count)),
        probe_seed=_derive_open(run_id, "probe", beacon).randrange(1, SEED_SPACE),
        beacon=beacon,
        root_commitment="",  # nothing is committed to, because nothing is held
    )


def _distinct(rng: random.Random, count: int) -> list[int]:
    seen: set[int] = set()
    while len(seen) < count:
        seen.add(rng.randrange(1, SEED_SPACE))
    return sorted(seen)


def build_instances(sample: RunSample, workflow) -> tuple[list, list]:
    """Materialise a run's instances.

    Returns:
        ``(hidden, ood)`` instance lists, in seed order so two workers building
        the same run iterate them identically.
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
