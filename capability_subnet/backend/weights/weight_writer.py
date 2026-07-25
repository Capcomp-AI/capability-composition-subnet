"""Building the weight vector.

The engine computes weights; it never sets them. What it publishes is a signed
vector, and validators independently decide to submit it. That separation is what
keeps a centralised evaluation engine honest at the chain layer: the engine can
publish whatever it likes, but emission only moves if validators — who verify the
signature, check the vector against the published reports, and answer to their own
stake — agree to write it.

Two rules govern every vector produced here:

* a permanent reference holding the throne earns nothing. The share burns.
* unfilled shares burn rather than being redistributed. "Best of a bad window"
  is not a thing this network pays for.
"""

from __future__ import annotations

import logging

from capability_subnet.backend.baselines.references import is_reference
from capability_subnet.common import constants as C
from capability_subnet.common.schemas import ChampionRecord, WeightEntry, WeightVector

log = logging.getLogger(__name__)


def burn_entry(weight: float, burn_uid: int = C.BURN_UID) -> WeightEntry:
    return WeightEntry(uid=burn_uid, hotkey="", weight=weight, role="burn")


def winner_take_all(
    champion: ChampionRecord | None,
    *,
    window_id: int,
    block: int,
    spec_version: int,
    burn_percentage: float = 0.0,
    burn_uid: int = C.BURN_UID,
    champion_report_sha256: str | None = None,
    workflow_id: str = C.DEFAULT_WORKFLOW_ID,
) -> WeightVector:
    """Assign the whole workflow share to the reigning champion.

    Args:
        champion: the incumbent, or ``None`` when nothing holds the throne.
        burn_percentage: an operational safety valve. Routes part of the share to
            the burn UID without changing who the champion is, which is what an
            operator reaches for during an incident rather than stopping the
            engine and freezing emission entirely.

    Returns:
        A vector summing to exactly 1.0.
    """
    entries: list[WeightEntry] = []
    champion_hotkey: str | None = None

    payable = (
        champion is not None
        and champion.uid is not None
        and champion.hotkey
        and not champion.is_reference
        and not is_reference(champion.candidate_id)
    )

    if not payable:
        reason = "no champion" if champion is None else f"{champion.candidate_id} is a reference"
        log.info("burning the full workflow share: %s", reason)
        entries.append(burn_entry(1.0, burn_uid))
    else:
        assert champion is not None and champion.uid is not None  # narrowed above
        burn = max(0.0, min(1.0, burn_percentage))
        champion_hotkey = champion.hotkey

        if burn > 0.0:
            if champion.uid == burn_uid:
                # The champion is registered on the burn UID. Splitting would be
                # meaningless, so the whole share goes to it as one entry.
                entries.append(
                    WeightEntry(
                        uid=champion.uid,
                        hotkey=champion.hotkey or "",
                        weight=1.0,
                        role="champion",
                    )
                )
            else:
                entries.append(
                    WeightEntry(
                        uid=champion.uid,
                        hotkey=champion.hotkey or "",
                        weight=1.0 - burn,
                        role="champion",
                    )
                )
                entries.append(burn_entry(burn, burn_uid))
        else:
            entries.append(
                WeightEntry(
                    uid=champion.uid,
                    hotkey=champion.hotkey or "",
                    weight=1.0,
                    role="champion",
                )
            )

    return WeightVector(
        workflow_id=workflow_id,
        window_id=window_id,
        computed_at_block=block,
        spec_version=spec_version,
        mode=C.MODE_WINNER_TAKE_ALL,
        burn_percentage=burn_percentage,
        entries=_normalise(entries),
        champion_hotkey=champion_hotkey,
        champion_report_sha256=champion_report_sha256,
    )


def graded_top3(
    ranked: list[tuple[int, str]],
    *,
    window_id: int,
    block: int,
    spec_version: int,
    burn_percentage: float = 0.0,
    burn_uid: int = C.BURN_UID,
    workflow_id: str = C.DEFAULT_WORKFLOW_ID,
) -> WeightVector:
    """Split the share across up to three qualified miners.

    Args:
        ranked: ``(uid, hotkey)`` pairs of the qualified miners, best first. Only
            miners that cleared every gate belong here — an unqualified miner
            does not get promoted into a lower rank because a slot was free.

    Returns:
        A vector where any unfilled rank's share has been burned.
    """
    entries: list[WeightEntry] = []
    roles: tuple[str, ...] = ("champion", "runner_up", "third")

    burn = max(0.0, min(1.0, burn_percentage))
    payable_share = 1.0 - burn
    unfilled = 0.0

    for index, share in enumerate(C.GRADED_TOP3_SHARES):
        allocation = share * payable_share
        if index < len(ranked):
            uid, hotkey = ranked[index]
            entries.append(
                WeightEntry(uid=uid, hotkey=hotkey, weight=allocation, role=roles[index])
            )
        else:
            unfilled += allocation

    burned = burn + unfilled
    if burned > 0.0:
        entries.append(burn_entry(burned, burn_uid))

    return WeightVector(
        workflow_id=workflow_id,
        window_id=window_id,
        computed_at_block=block,
        spec_version=spec_version,
        mode=C.MODE_GRADED_TOP3,
        burn_percentage=burn_percentage,
        entries=_normalise(entries),
        champion_hotkey=ranked[0][1] if ranked else None,
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

    scaled: list[WeightEntry] = []
    burned = extra
    for entry in vector.entries:
        if entry.role == "burn" or entry.uid == burn_uid:
            burned += entry.weight
        else:
            scaled.append(entry.model_copy(update={"weight": entry.weight * (1.0 - extra)}))

    scaled.append(burn_entry(burned, burn_uid))

    return vector.model_copy(
        update={"entries": _normalise(scaled), "burn_percentage": min(1.0, extra)}
    )
