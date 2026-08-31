"""Building the weight vector.

Whoever computes a vector never sets it. A validator measures the field on its
own cards, derives this vector from what it measured, and writes that; a vector
published by somebody else is evidence to check against, not an instruction. So
emission only moves on numbers a validator either produced or verified — and it
answers for them with its own stake.

The rule the vector expresses, in full:

* ``BURN_SHARE`` of every run burns. What is left is the miner share.
* A run pays whatever cleared its hard gates. The bar is absolute — the entry
  gate requires ``DEFAULT_END_TO_END_MARGIN`` of completion over the strongest
  permanent reference — so it is a statement about the package rather than
  about whoever happens to hold the throne. A field where nobody clears it
  bought nothing, and the whole miner share burns.
* The miner share is split by rank: ``RANK_SHARES`` for the first five and
  ``TAIL_SHARE`` across ranks six to ``PAID_RANKS``, in proportion to grade.
* A permanent reference is not a competitor, and an unfilled rank among the
  first five burns rather than being redistributed.

Payment used to require dethroning the incumbent by
``CHAMPION_DETHRONE_MARGIN``. That made emission depend on a number no
candidate in the run could see or affect — the grade of a package measured on a
different draw — so a field that cleared every absolute bar was paid nothing
because a previous run had been strong. Run 415 burned entirely with five
qualified packages in it. The throne is still recorded, and the margin still
decides whether it changes hands; it no longer decides whether anyone is paid.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import EvaluationReport, WeightEntry, WeightVector
from capability_subnet.scoring.contribution import ContributionInputs, contribution_score
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
    # Capped at PAID_RANKS, which the named shares were not. champion_ladder
    # slices the field to PAID_RANKS before calling, so the two agree in
    # practice — but this says "ranks beyond PAID_RANKS receive nothing" while
    # handing out five named shares regardless, and a second caller reading the
    # docstring at its word would pay ranks that earn nothing.
    for index in range(min(count, len(C.RANK_SHARES), C.PAID_RANKS)):
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

    # The bar is the hard gates, which every entry in `ranked` has already
    # cleared, and the entry gate among them is an absolute margin over the
    # strongest permanent reference. So a non-empty field is by construction a
    # field that beat the bar, and it is paid.
    #
    # `champion_grade` no longer gates payment. It still says whether the
    # throne changes hands, which is recorded and carried into the next run.
    threshold = dethrone_threshold(champion_grade)
    took_the_throne = bool(ranked) and ranked[0][2] > threshold
    qualifying = ranked[: C.PAID_RANKS]

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

    if not ranked:
        log.info(
            "run %d: no candidate cleared every hard gate, so the whole miner share burns",
            run_id,
        )
    elif not took_the_throne:
        log.info(
            "run %d: %d candidate(s) cleared the gates and are paid; the leading "
            "grade was %.6f against a throne at %.6f, so the throne does not "
            "change hands",
            run_id,
            len(qualifying),
            ranked[0][2],
            threshold,
        )

    return WeightVector(
        workflow_id=workflow_id,
        run_id=run_id,
        computed_at_block=block,
        burn_percentage=burn_share,
        entries=_normalise(entries),
        # The throne, not the pay slot. Rank one always carries the
        # "champion" role because it takes the champion's share, but the
        # throne only changes hands when the leader actually beats the
        # reigning grade — so this is None on a run that paid a full ladder
        # and dethroned nobody. Setting it to the leader regardless would
        # record every run's best candidate as having taken a throne it did
        # not take, and the next run would be measured against the wrong bar.
        champion_hotkey=qualifying[0][1] if (qualifying and took_the_throne) else None,
    )


def vector_from_reports(
    reports: Iterable[EvaluationReport],
    *,
    run_id: int,
    block: int,
    champion_grade: float | None,
    burn_uid: int = C.BURN_UID,
    burn_share: float = C.BURN_SHARE,
    workflow_id: str = C.DEFAULT_WORKFLOW_ID,
) -> WeightVector:
    """Derive the weight vector from a run's published evaluation reports.

    This is the "anyone can re-derive the vector from a stream of reports" path
    those reports exist for. A validator in endpoint mode fetches the signed
    reports, verifies them, and calls this to compute the vector itself rather
    than trusting a vector somebody else computed. It is the same ladder local
    mode applies to its own measurements; only the source of the grades differs.

    Only reports that cleared every hard gate are ranked. A permanent reference
    is the bar, not a competitor, and is dropped. Each grade is recomputed from
    the report's own scores against the reference it was measured on, so a report
    whose grade does not follow from its scores earns nothing it did not measure.
    """
    graded: list[tuple[int, str, float]] = []
    for report in reports:
        if not report.gates_passed:
            continue
        hotkey = report.miner_hotkey
        if not hotkey or is_reference(hotkey) or report.miner_uid is None:
            continue
        grade = contribution_score(
            ContributionInputs(
                scores=report.scores,
                reference_e2e=report.strongest_reference_score,
            )
        )
        graded.append((report.miner_uid, hotkey, grade))
    graded.sort(key=lambda row: -row[2])
    return champion_ladder(
        graded,
        run_id=run_id,
        block=block,
        champion_grade=champion_grade,
        burn_uid=burn_uid,
        burn_share=burn_share,
        workflow_id=workflow_id,
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
