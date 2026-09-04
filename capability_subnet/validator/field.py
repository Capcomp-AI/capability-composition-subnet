"""Fetching a run's field of submissions, from the chain.

A miner seals a recipe to the drand round its run closes at and writes it into
the commitments pallet. The chain decrypts it there, on its own schedule, and
the plaintext lands in public storage. That is where a validator reads it, and
there is no other route: no service to ask, no credential to hold, nothing the
operator could withhold or substitute.

This replaces reading a submission service. That service held the bodies
privately and served them on a schedule it chose, which meant a validator's
field was whatever the operator decided to return - checkable against a digest
the same operator stored beside it, which is no check at all if the two were
written together. Every validator now derives the same field from the same
chain state, and a body that does not unseal, parse and hash correctly is
refused by each of them identically.

What a validator gains beyond independence is timing. The chain opens a run's
commitments when the run that measures them opens, so the field is readable for
exactly as long as it is needed, and a validator measuring run N+1 has run N's
recipes for the whole of it.

Every body is checked before it becomes a candidate: the reveal round must be
the one the run pins, the payload must decompress and parse, and the digest is
taken over the canonical form rather than over what was sealed. The chain
stores whatever a miner puts there and vouches for none of it, so all of that
happens here. :mod:`capability_subnet.common.sealed` is where it lives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from capability_subnet.common import constants as C
from capability_subnet.common.chain import measured_in_run

log = logging.getLogger(__name__)


class FieldError(RuntimeError):
    """The field could not be read, or could not be trusted once it was."""


class FieldPending(RuntimeError):
    """The field is on chain but the chain has not opened it yet.

    Deliberately not a :class:`FieldError`. Both mean "no usable field right
    now" and they call for opposite responses: an unreadable chain is a
    permanent condition this validator cannot fix, so it burns and says so; an
    unopened one resolves by itself within the hour, and burning through it
    would pay nobody for a run a field was in fact entered for.

    A run's commitments unseal ``REVEAL_MARGIN_BLOCKS`` after the measuring run
    opens - an hour, at twelve seconds a block. A validator whose weight
    interval lands in that hour sees a chain holding every recipe and readable
    payloads for none of them.
    """


def _pending_reveals(records, measuring_run: int, *, run_blocks: int) -> list[str]:
    """Commitments this run must measure that the chain has not opened yet.

    Keyed on the reveal round each source run pins, not on "sealed" alone. A
    commitment sealed to some other round is not this run's to wait for - a
    miner who sealed to the wrong run would otherwise hold every later run
    hostage, since their payload never opens on a schedule this run cares
    about.
    """
    from capability_subnet.common.timelock import reveal_round_for_run

    expected = {
        reveal_round_for_run(source, run_blocks=run_blocks)
        for source in (measuring_run - 2, measuring_run - 1)
        if source >= 0
    }
    return [
        getattr(record, "hotkey", "")
        for record in records
        if int(getattr(record, "reveal_round", 0) or 0) in expected
        and not getattr(record, "revealed", None)
    ]


@dataclass(slots=True)
class FetchedSubmission:
    """One submission, unsealed from the chain and checked."""

    hotkey: str
    uid: int
    recipe_sha256: str
    recipe_raw: bytes
    first_block: int
    submission_count: int


def field_for_run(
    subtensor,
    measuring_run: int,
    *,
    netuid: int = 103,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
) -> list[FetchedSubmission]:
    """Everything ``measuring_run`` is supposed to measure.

    Two source runs, not one. A submission is measured in the run after the one
    it was made in - unless it was made inside the settling window, in which
    case it is held over one further run. So this run's field is the settled
    part of run N-1 plus the late part of N-2, and a validator that read only
    N-1 would leave the held-over miners unmeasured and unpaid.

    :func:`measured_in_run` decides which is which, from the commitment's block
    alone. Every validator reads the same chain and selects the same field, so
    there is nothing here to disagree about.

    Refusals are logged rather than raised. One miner sealing something
    malformed is that miner's run to lose, not a reason to stop measuring the
    rest of the field - but it is said out loud, because the chain gave them no
    error and this is the only place the reason exists.

    Raises:
        FieldError: if the chain cannot be read at all. An unreadable chain and
            an empty field produce the same weight vector, and reporting the
            second as the first would describe a subnet nobody entered.
    """
    from capability_subnet.common.chain import fetch_metagraph
    from capability_subnet.common.sealed import field_from_commitments

    try:
        view = fetch_metagraph(subtensor, netuid)
    except Exception as exc:  # noqa: BLE001 - reported, never absorbed
        raise FieldError(f"could not read netuid {netuid} from the chain: {exc}") from exc

    # Before anything is read: is there anything left to open? An unopened
    # commitment and no commitment at all are indistinguishable further down -
    # field_from_commitments skips a record with no revealed payload - so the
    # question has to be asked here, while the sealed records are still in hand.
    pending = _pending_reveals(view.commitment_records, measuring_run, run_blocks=run_blocks)
    if pending:
        raise FieldPending(
            f"run {measuring_run}: {len(pending)} commitment(s) for this run have "
            f"not unsealed yet. They open at the round their run pins; until then "
            f"the field is incomplete and measuring it would score a fraction of "
            f"it and burn the rest."
        )

    registered = set(view.hotkeys)
    seen: dict[str, FetchedSubmission] = {}
    refusals: list[tuple[str, str]] = []

    for source in (measuring_run - 2, measuring_run - 1):
        if source < 0:
            continue
        admitted, refused = field_from_commitments(
            view.commitment_records,
            source,
            registered=registered,
            run_blocks=run_blocks,
        )
        refusals.extend(refused)
        for entry in admitted:
            if not measured_in_run(entry.block, measuring_run, run_blocks):
                continue
            # A hotkey with a commitment in both source runs re-sealed; the one
            # this run measures is the one whose block says so, and both have
            # already been filtered to exactly that.
            seen[entry.hotkey] = FetchedSubmission(
                hotkey=entry.hotkey,
                uid=entry.uid if entry.uid is not None else -1,
                recipe_sha256=entry.recipe_sha256,
                recipe_raw=entry.recipe_bytes,
                first_block=entry.block,
                submission_count=1,
            )

    out = sorted(seen.values(), key=lambda e: e.first_block)
    log.info(
        "run %d: field of %d from the chain, drawn from runs %d and %d",
        measuring_run,
        len(out),
        measuring_run - 2,
        measuring_run - 1,
    )
    for hotkey, reason in refusals:
        log.info("run %d: %s refused - %s", measuring_run, hotkey[:12], reason)
    return out
