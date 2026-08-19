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

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import ChampionRecord, WeightEntry, WeightVector
from capability_subnet.scoring.references import is_reference

log = logging.getLogger(__name__)


def burn_entry(weight: float, burn_uid: int = C.BURN_UID) -> WeightEntry:
    return WeightEntry(uid=burn_uid, hotkey="", weight=weight, role="burn")


def survival_tail(
    uids_and_hotkeys: list[tuple[int, str]],
    total_share: float,
    *,
    exclude_uid: int | None = None,
    max_entries: int = C.MAX_TAIL_ENTRIES,
) -> list[WeightEntry]:
    """Split ``total_share`` across queued miners as a linear taper.

    This is deregistration protection, not payment for work. Bittensor prunes by
    emission, lowest first, so a miner holding exactly zero is the one the chain
    evicts when a slot is needed — and under a pure winner-take-all split that is
    every challenger still waiting in the queue. The engine evaluates roughly one
    challenger per window, so a miner can easily wait days for its single
    evaluation; being pruned during that wait means the network loses the
    submission it was about to judge, and the queue drains itself.

    A taper rather than an equal split so the ordering still carries information:
    the front of the queue, which is closest to being evaluated, is worth most.
    """
    payable = [(uid, hotkey) for uid, hotkey in uids_and_hotkeys if uid != exclude_uid and hotkey][
        :max_entries
    ]
    if not payable or total_share <= 0.0:
        return []

    count = len(payable)
    # Weights count, count-1, ..., 1 — normalised to `total_share`.
    denominator = count * (count + 1) / 2
    return [
        WeightEntry(
            uid=uid,
            hotkey=hotkey,
            weight=total_share * (count - index) / denominator,
            role="queued",
        )
        for index, (uid, hotkey) in enumerate(payable)
    ]


def winner_take_all(
    champion: ChampionRecord | None,
    *,
    window_id: int,
    block: int,
    burn_percentage: float = 0.0,
    burn_uid: int = C.BURN_UID,
    champion_report_sha256: str | None = None,
    workflow_id: str = C.DEFAULT_WORKFLOW_ID,
    tail: list[tuple[int, str]] | None = None,
    tail_share: float = C.DEFAULT_TAIL_SHARE,
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

    # The tail is paid whether or not anyone holds the throne. With an empty
    # throne it is the *only* thing standing between a queue of unevaluated
    # submissions and the pruning mechanism, which is precisely the moment the
    # network can least afford to lose them.
    tail_entries = survival_tail(
        tail or [],
        tail_share,
        exclude_uid=champion.uid if champion is not None else None,
    )
    tail_total = sum(entry.weight for entry in tail_entries)
    entries.extend(tail_entries)
    remaining = max(0.0, 1.0 - tail_total)

    if not payable:
        reason = "no champion" if champion is None else f"{champion.candidate_id} is a reference"
        log.info("burning the workflow share: %s", reason)
        entries.append(burn_entry(remaining, burn_uid))
    else:
        assert champion is not None and champion.uid is not None  # narrowed above
        burn = max(0.0, min(1.0, burn_percentage))
        champion_hotkey = champion.hotkey

        if burn > 0.0 and champion.uid != burn_uid:
            entries.append(
                WeightEntry(
                    uid=champion.uid,
                    hotkey=champion.hotkey or "",
                    weight=remaining * (1.0 - burn),
                    role="champion",
                )
            )
            entries.append(burn_entry(remaining * burn, burn_uid))
        else:
            # Either nothing is being burned, or the champion happens to sit on
            # the burn UID and splitting would be meaningless.
            entries.append(
                WeightEntry(
                    uid=champion.uid,
                    hotkey=champion.hotkey or "",
                    weight=remaining,
                    role="champion",
                )
            )

    return WeightVector(
        workflow_id=workflow_id,
        window_id=window_id,
        computed_at_block=block,
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
        mode=C.MODE_GRADED_TOP3,
        burn_percentage=burn_percentage,
        entries=_normalise(entries),
        champion_hotkey=ranked[0][1] if ranked else None,
    )


def graded_contribution(
    champion: ChampionRecord | None,
    contributors: list[tuple[int, str, float]],
    *,
    window_id: int,
    block: int,
    champion_base_share: float = C.CHAMPION_BASE_SHARE,
    burn_percentage: float = 0.0,
    burn_uid: int = C.BURN_UID,
    champion_report_sha256: str | None = None,
    workflow_id: str = C.DEFAULT_WORKFLOW_ID,
    tail: list[tuple[int, str]] | None = None,
    tail_share: float = C.DEFAULT_TAIL_SHARE,
    no_champion_burn_share: float = C.NO_CHAMPION_BURN_SHARE,
) -> WeightVector:
    """Pay the champion a fixed share and split the rest by graded contribution.

    Args:
        contributors: ``(uid, hotkey, contribution)`` for every miner whose
            evaluation cleared all hard gates recently. Grades come from
            :func:`scorer.contribution.contribution_score`; a candidate that
            failed a gate is not here at all, because grading applies within the
            qualified set rather than as a consolation prize for producing
            something undeployable.

    Returns:
        A vector summing to exactly 1.0.

    Three claims on the emission, in priority order: the queue tail, which keeps
    unevaluated submissions from being pruned; the champion's base share; and the
    graded pool for everyone who demonstrated something. Whatever the graded pool
    cannot allocate — because nobody qualified — is burned rather than handed to
    the champion, so an empty field does not silently become a bonus for holding
    an uncontested throne.
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
    champion_uid = champion.uid if champion is not None else None

    tail_entries = survival_tail(tail or [], tail_share, exclude_uid=champion_uid)
    entries.extend(tail_entries)
    remaining = max(0.0, 1.0 - sum(entry.weight for entry in tail_entries))

    burn = max(0.0, min(1.0, burn_percentage))
    burned = remaining * burn
    payable_pool = remaining - burned

    graded = [
        (uid, hotkey, grade)
        for uid, hotkey, grade in contributors
        if grade > 0.0 and hotkey and uid != champion_uid
    ]
    graded.sort(key=lambda item: (-item[2], item[0]))
    graded = graded[: C.MAX_GRADED_CONTRIBUTORS]

    # The leader's share is set aside whether or not anyone holds the throne.
    #
    # With a champion it goes to them. Without one, half the window burns and the
    # best measured package leads what remains, on the same terms a champion
    # leads the whole. So a leaderless window pays its best miner roughly half
    # what a crowned window pays its champion: the field is still ranked and
    # still paid, and the throne is still the thing worth taking.
    #
    # The share is never handed to the runners-up intact. Doing that would pay
    # *more* in exactly the windows where the field was weakest — measured
    # before that was fixed, a single contributor took 0.80 of a leaderless
    # window against 0.36 of a contested one, which is the incentive pointing
    # backwards.
    if payable:
        champion_share = payable_pool * champion_base_share
        graded_pool = payable_pool - champion_share
    else:
        burned += payable_pool * no_champion_burn_share
        leaderless_pool = payable_pool * (1.0 - no_champion_burn_share)
        champion_share = leaderless_pool * champion_base_share
        graded_pool = leaderless_pool - champion_share

    if payable:
        assert champion is not None and champion.uid is not None  # narrowed above
        champion_hotkey = champion.hotkey
        entries.append(
            WeightEntry(
                uid=champion.uid,
                hotkey=champion.hotkey or "",
                weight=champion_share,
                role="champion",
            )
        )

    leader: tuple[int, str, float] | None = None
    if not payable and graded:
        # `graded` is already sorted by grade, so the head is the best measured
        # package in the window. It leads in the throne's absence.
        leader, graded = graded[0], graded[1:]
        entries.append(
            WeightEntry(
                uid=leader[0],
                hotkey=leader[1],
                # Not role="champion": nothing was crowned, and a vector that
                # claimed otherwise would disagree with the champion record an
                # auditor replays it against.
                weight=champion_share,
                role="contributor",
            )
        )

    count = len(graded)
    if count > 0 and graded_pool > 0.0:
        # Split by rank, not by grade: the runner-up takes the most and it tapers
        # down the ranked field. Weights count, count-1, ..., 1 normalised to the
        # graded pool, so a higher place always pays more than a lower one — by
        # position, not by how the grades happened to cluster. `graded` is
        # already sorted best-first, so index 0 is the runner-up.
        denominator = count * (count + 1) / 2
        for index, (uid, hotkey, _grade) in enumerate(graded):
            entries.append(
                WeightEntry(
                    uid=uid,
                    hotkey=hotkey,
                    weight=graded_pool * (count - index) / denominator,
                    role="contributor",
                )
            )
    else:
        # Nobody left to share it. Burn rather than promoting the leader into it:
        # a window with one qualified package is not a bigger achievement.
        burned += graded_pool

    if burned > 0.0:
        entries.append(burn_entry(burned, burn_uid))

    if not entries:
        entries.append(burn_entry(1.0, burn_uid))

    log.info(
        "graded window %d: champion=%s, %d contributor(s), %.3f burned",
        window_id,
        champion_hotkey[:12] if champion_hotkey else "none",
        len(graded),
        burned,
    )

    return WeightVector(
        workflow_id=workflow_id,
        window_id=window_id,
        computed_at_block=block,
        mode=C.MODE_GRADED_CONTRIBUTION,
        burn_percentage=burn_percentage,
        entries=_normalise(entries),
        champion_hotkey=champion_hotkey,
        champion_report_sha256=champion_report_sha256,
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
