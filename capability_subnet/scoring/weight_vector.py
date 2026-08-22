"""Building the weight vector.

The engine computes weights; it never sets them. What it publishes is a signed
vector, and validators independently decide to submit it. That separation is what
keeps a centralised evaluation engine honest at the chain layer: the engine can
publish whatever it likes, but emission only moves if validators — who verify the
signature, check the vector against the published reports, and answer to their own
stake — agree to write it.

The rule the vector expresses, in full:

* ``BURN_SHARE`` of every run burns. What is left is the miner share.
* The run pays nothing unless its leader takes the throne, which means
  exceeding the reigning champion's grade by ``CHAMPION_DETHRONE_MARGIN``. A
  run whose best candidate cannot do that offered the network nothing it did
  not already have, and the whole miner share burns.
* When the leader does take it, the miner share is split by rank across the
  field behind them: ``RANK_SHARES`` for the first five and ``TAIL_SHARE``
  across ranks six to ``PAID_RANKS``, in proportion to grade.
* A permanent reference on the throne earns nothing, and an unfilled rank burns
  rather than being redistributed. "Best of a bad run" is not paid for.
"""

from __future__ import annotations

import logging

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import WeightEntry, WeightVector
from capability_subnet.scoring.references import is_reference

log = logging.getLogger(__name__)


def burn_entry(weight: float, burn_uid: int = C.BURN_UID) -> WeightEntry:
    return WeightEntry(uid=burn_uid, hotkey="", weight=weight, role="burn")


def dethrone_threshold(champion_grade: float | None) -> float:
    """The grade the leading candidate must exceed for the run to pay anyone.

    With no champion the threshold is zero and the field is ranked as it
    stands, which is how the first throne is filled.
    """
    if champion_grade is None:
        return 0.0
    return champion_grade + C.CHAMPION_DETHRONE_MARGIN


def rank_shares(count: int) -> list[float]:
    """The share of the miner pool each of ``count`` ranks receives.

    Ranks beyond ``PAID_RANKS`` receive nothing. The tail is returned as an
    equal split here and re-weighted by grade in :func:`champion_ladder`, which
    is the only caller that knows the grades.
    """
    if count <= 0:
        return []
    shares = [0.0] * count
    for index in range(min(count, len(C.RANK_SHARES))):
        shares[index] = C.RANK_SHARES[index]
    tail = [i for i in range(len(C.RANK_SHARES), min(count, C.PAID_RANKS))]
    if tail:
        for index in tail:
            shares[index] = C.TAIL_SHARE / len(tail)
    return shares


def champion_ladder(
    ranked: list[tuple[int, str, float]],
    *,
    run_id: int,
    block: int,
    champion_grade: float | None,
    burn_uid: int = C.BURN_UID,
    burn_share: float = C.BURN_SHARE,
    workflow_id: str = C.DEFAULT_WORKFLOW_ID,
) -> WeightVector:
    """Split a run's emission across the miners that took the throne.

    Args:
        ranked: ``(uid, hotkey, grade)`` for every candidate that cleared every
            hard gate, best grade first. A candidate that failed a gate does not
            belong here; it is not promoted into a free rank.
        champion_grade: the reigning champion's grade, or ``None`` when the
            throne is empty and this run fills it.

    Returns:
        A vector in which every share not paid to a qualifying miner is burned.

    Raises:
        ValueError: if ``ranked`` is not ordered by descending grade, contains a
            permanent reference, or ``burn_share`` is outside ``[0, 1]``. Each
            would produce a vector that looks reasonable and pays the wrong
            miners, so none of them is corrected quietly.
    """
    if not 0.0 <= burn_share <= 1.0:
        raise ValueError(f"burn_share is {burn_share}; expected a fraction in [0, 1]")
    references = [hotkey for _, hotkey, _ in ranked if is_reference(hotkey)]
    if references:
        raise ValueError(
            f"the field contains permanent references ({', '.join(references)}); "
            "a reference is the bar, not a competitor, and paying one would pay "
            "the operator for the package the network already has"
        )
    grades = [grade for _, _, grade in ranked]
    if grades != sorted(grades, reverse=True):
        raise ValueError(
            "ranked must be ordered by descending grade; the rank ladder pays "
            "position, so an unsorted field pays the wrong miners"
        )

    # The bar is on the leader, and on the leader alone. Either the run
    # produced something better than what the network already has, in which
    # case the field behind the winner is paid by rank, or it did not, in which
    # case the run bought nothing and pays nobody.
    threshold = dethrone_threshold(champion_grade)
    took_the_throne = bool(ranked) and ranked[0][2] > threshold
    qualifying = ranked[: C.PAID_RANKS] if took_the_throne else []

    pool = 1.0 - burn_share
    entries: list[WeightEntry] = []
    paid = 0.0

    shares = rank_shares(len(qualifying))
    tail_grades = [row[2] for row in qualifying[len(C.RANK_SHARES) :]]
    tail_total = sum(tail_grades)

    for index, (uid, hotkey, grade) in enumerate(qualifying):
        if index < len(C.RANK_SHARES):
            share = shares[index]
        elif tail_total > 0.0:
            # By grade, not evenly: an even split would pay a candidate that
            # barely qualified the same as one that nearly placed fifth.
            share = C.TAIL_SHARE * (grade / tail_total)
        else:
            share = shares[index]
        allocation = share * pool
        paid += allocation
        entries.append(
            WeightEntry(
                uid=uid,
                hotkey=hotkey,
                weight=allocation,
                role="champion" if index == 0 else "runner_up",
            )
        )

    burned = 1.0 - paid
    if burned > 0.0:
        entries.append(burn_entry(burned, burn_uid))

    if not took_the_throne:
        best = f"{ranked[0][2]:.6f}" if ranked else "no measured candidate"
        log.info(
            "run %d: the leading grade was %s against a throne at %.6f; "
            "nobody took it, so the whole miner share burns",
            run_id,
            best,
            threshold,
        )

    return WeightVector(
        workflow_id=workflow_id,
        run_id=run_id,
        computed_at_block=block,
        burn_percentage=burn_share,
        entries=_normalise(entries),
        champion_hotkey=qualifying[0][1] if qualifying else None,
    )


def _normalise(entries: list[WeightEntry]) -> list[WeightEntry]:
    """Merge duplicate UIDs and correct rounding drift.

    The chain rejects a vector with a repeated UID, and the schema requires the
    weights to sum to one. Both can be violated by arithmetic that is otherwise
    correct — a champion registered on the burn UID produces a duplicate, and
    three shares of a burn-adjusted total rarely sum to exactly one in binary
    floating point.
    """
    if not entries:
        return [burn_entry(1.0)]

    merged: dict[int, WeightEntry] = {}
    for entry in entries:
        existing = merged.get(entry.uid)
        if existing is None:
            merged[entry.uid] = entry.model_copy()
        else:
            existing.weight += entry.weight

    ordered = [merged[uid] for uid in sorted(merged)]
    total = sum(entry.weight for entry in ordered)
    if total <= 0.0:
        return [burn_entry(1.0)]

    for entry in ordered:
        entry.weight = entry.weight / total

    # Push the residue into the largest entry so the sum is exactly one.
    residue = 1.0 - sum(entry.weight for entry in ordered)
    if abs(residue) > 0.0:
        largest = max(ordered, key=lambda entry: entry.weight)
        largest.weight = min(1.0, max(0.0, largest.weight + residue))

    return ordered


def apply_validator_burn(vector: WeightVector, extra_burn: float, burn_uid: int) -> WeightVector:
    """Apply a validator's own additional burn to a published vector.

    A validator may burn more than the engine asked for but never less. Allowing
    less would let a validator quietly override an operator's incident response;
    allowing more is a validator declining to pay a champion it does not trust,
    which is a decision it is entitled to make with its own stake.
    """
    extra = max(0.0, min(1.0, extra_burn))
    if extra <= 0.0:
        return vector

    # Everything already in the vector is scaled by what remains, burn included,
    # and the extra is added on top. Adding the extra to an *unscaled* burn
    # over-burns: a vector that already burned four fifths, asked for half
    # again, would pay the miners a seventh less than the half they were owed
    # and the sum would only come back to one because it is normalised after.
    scaled: list[WeightEntry] = []
    burned = extra
    for entry in vector.entries:
        remaining = entry.weight * (1.0 - extra)
        if entry.role == "burn" or entry.uid == burn_uid:
            burned += remaining
        else:
            scaled.append(entry.model_copy(update={"weight": remaining}))

    scaled.append(burn_entry(burned, burn_uid))

    return vector.model_copy(
        update={"entries": _normalise(scaled), "burn_percentage": min(1.0, extra)}
    )
