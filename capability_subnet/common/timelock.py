"""When a run's recipes unseal, derived from constants alone.

A recipe reaches the chain as a ``TimelockEncrypted`` field carrying its own
``reveal_round``: the drand round whose signature decrypts it. The pallet
decrypts on finalize once that round's pulse arrives, and writes the plaintext
into public storage. Nobody chooses who reads it - drand is a public beacon, so
the round chooses only *when*, and when it comes everyone has it at once.

Two things follow, and they are why this module exists.

The round travels in the commitment, which means the **miner** supplies it. An
early round unseals a recipe while the field is still submitting; a late one
unseals it after the run that was supposed to measure it has finished, and the
submission is lost. Neither needs bad intent to happen - both are what a wrong
constant or a drifting estimate produces on its own. So the round is not read
from the commitment and trusted: it is derived here, and a commitment carrying
any other value is not a submission.

And the round cannot be read from the chain, because the block it corresponds
to has not happened. ``run_opens_block`` gives the close of run N as a height;
drand needs an instant. Bridging the two is an estimate, and the estimate is
made from ``RUN_EPOCH_UNIX`` and ``BLOCK_SECONDS`` rather than from the caller's
clock or a chain read, so that a miner committing on Tuesday and an engine
checking on Wednesday compute the same number with nothing to agree on.

Everything here is a pure function of the constants. That is the point: it is
the one part of the protocol where a miner and the engine must arrive at an
identical value independently, and a shared service to ask would be a shared
thing to break.
"""

from __future__ import annotations

import math

from capability_subnet.common import constants as C
from capability_subnet.common.chain import run_opens_block

#: Unix time of drand quicknet round 1, and the seconds between rounds.
#:
#: Mirrored from ``bittensor.timelock`` rather than imported, so that deriving a
#: reveal round costs nothing but arithmetic - the miner CLI, the engine and an
#: auditor's own script all reach the same value without a bittensor install.
#: :func:`tests.unit.test_the_reveal_round_is_pinned` holds them to the SDK.
QUICKNET_GENESIS: int = 1_692_803_367
QUICKNET_PERIOD: int = 3


class RevealRoundError(ValueError):
    """A commitment's reveal round is not the one its run pins."""


def round_at(unix_time: float) -> int:
    """The earliest quicknet round emitted at or after ``unix_time``.

    At-or-after, not the round in force: the round is being chosen so that a
    recipe is *still sealed* until an instant, so rounding must never land
    earlier than asked. Matches ``bittensor.timelock.round_at`` exactly.

    Raises:
        ValueError: for a time before quicknet's genesis.
    """
    if unix_time < QUICKNET_GENESIS:
        raise ValueError(f"{unix_time} precedes quicknet genesis {QUICKNET_GENESIS}")
    elapsed = math.ceil(unix_time) - QUICKNET_GENESIS
    return -(-elapsed // QUICKNET_PERIOD) + 1


def round_time(reveal_round: int) -> int:
    """The Unix time quicknet emits ``reveal_round``. Inverse of :func:`round_at`."""
    if reveal_round < 1:
        raise ValueError(f"round {reveal_round} is before quicknet round 1")
    return QUICKNET_GENESIS + (reveal_round - 1) * QUICKNET_PERIOD


def block_instant(
    block: int,
    *,
    epoch_block: int = C.RUN_EPOCH_BLOCK,
    epoch_unix: int = C.RUN_EPOCH_UNIX,
    block_seconds: int = C.BLOCK_SECONDS,
) -> int:
    """When the chain is expected to reach ``block``, as a Unix timestamp.

    An estimate, and only ever used for blocks that have not happened. For one
    that has, read its real timestamp from the chain - this is the future-facing
    half, and it is deliberately anchored rather than clock-relative so that two
    callers days apart agree.
    """
    return epoch_unix + (block - epoch_block) * block_seconds


def reveal_round_for_run(
    run_id: int,
    *,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
    margin_blocks: int = C.REVEAL_MARGIN_BLOCKS,
) -> int:
    """The one round at which run ``run_id``'s recipes unseal.

    Its close plus :data:`~capability_subnet.common.constants.REVEAL_MARGIN_BLOCKS`,
    converted to an instant and then to a round. The margin is not slack for its
    own sake - it is what keeps a slow chain from unsealing the field while the
    field is still submitting. See the constant for why that direction is the
    dangerous one.

    A run's close is the block its successor opens at, so this rides the same
    anchored schedule as everything else and needs no separate notion of when a
    run ends.

    Raises:
        ValueError: if ``run_blocks`` is not positive.
    """
    close_block = run_opens_block(run_id + 1, run_blocks)
    return round_at(block_instant(close_block + margin_blocks))


def opened_in_run(
    reveal_block: int,
    run_id: int,
    *,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
    margin_blocks: int = C.REVEAL_MARGIN_BLOCKS,
    tolerance_blocks: int = 600,
) -> bool:
    """Whether a commitment opened at ``reveal_block`` belongs to ``run_id``.

    For a record the chain has already opened. The pallet reports a sealed
    commitment's reveal round and, once it opens it, reports ``None`` - the
    round is gone from the record along with the commit block, which becomes
    the block the reveal landed at. So the round cannot be compared after the
    fact and the run has to be recovered from *when* it opened.

    That is sound, and it is not a weaker check than comparing the round. A run
    pins exactly one round, the chain opens a commitment at that round and at
    no other, and consecutive runs' rounds are a day apart. So the block a
    commitment opened at identifies its run outright, and the tolerance only
    has to absorb the difference between the round's wall-clock instant and the
    block that carried it.

    ``tolerance_blocks`` is two hours against a day of separation. Reveals have
    been observed a few blocks either side: the round fires on wall-clock time
    while the block height it lands on depends on how fast the chain is
    running, which is the same drift REVEAL_MARGIN_BLOCKS exists for.
    """
    expected = run_opens_block(run_id + 1, run_blocks) + margin_blocks
    return abs(reveal_block - expected) <= tolerance_blocks


def check_reveal_round(
    run_id: int,
    reveal_round: int,
    *,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
    margin_blocks: int = C.REVEAL_MARGIN_BLOCKS,
) -> None:
    """Raise unless ``reveal_round`` is the round ``run_id`` pins.

    Exact equality, with no tolerance window. A near-miss is not a rounding
    difference to be forgiven: both sides compute this from the same constants
    with integer arithmetic, so they either agree or one of them is running
    different code, and accepting a round "close enough" to the pinned one is
    accepting a reveal at a time nobody chose.

    Raises:
        RevealRoundError: with both rounds and the drift between them, because
            the useful question when this fires is which side is stale.
    """
    expected = reveal_round_for_run(run_id, run_blocks=run_blocks, margin_blocks=margin_blocks)
    if reveal_round == expected:
        return
    drift = (reveal_round - expected) * QUICKNET_PERIOD
    raise RevealRoundError(
        f"reveal_round {reveal_round} is not run {run_id}'s pinned round {expected} "
        f"({drift:+d}s); this commitment is not a submission to run {run_id}"
    )


def frame_payload(payload: bytes) -> bytes:
    """Prefix ``payload`` with its SCALE compact length, for encryption.

    The pallet stores exactly the bytes that were sealed, and a direct
    ``query(Commitments.RevealedCommitments, ...)`` hands them back whole. The
    metagraph's commitment records do not: that path strips a SCALE compact
    length prefix on the way out, on the assumption that whatever sealed the
    payload wrote one.

    So an unframed payload survives one reader and not the other, silently.
    Verified on testnet 544: a zlib stream sealed as ``78 da`` came back from
    the metagraph beginning ``da``, because ``0x78 & 0b11`` reads as a one-byte
    compact header. A recipe short by one to four bytes fails its digest check
    and looks exactly like a miner who submitted garbage.

    Writing the frame deliberately makes both readers agree: the metagraph eats
    it, and a direct reader drops it with :func:`unframe_payload`.
    """
    length = len(payload)
    if length < 0b0100_0000:
        prefix = bytes([length << 2])
    elif length < 0b0100_0000_0000_0000:
        prefix = ((length << 2) | 0b01).to_bytes(2, "little")
    elif length < 0b0100_0000_0000_0000_0000_0000_0000_0000:
        prefix = ((length << 2) | 0b10).to_bytes(4, "little")
    else:  # pragma: no cover - a recipe this size cannot reach the chain
        raise ValueError(f"payload of {length} bytes is too large to frame")
    return prefix + payload


def unframe_payload(framed: bytes) -> bytes:
    """Recover a payload written by :func:`frame_payload`.

    The inverse, for reading ``RevealedCommitments`` directly rather than
    through an SDK that has already stripped the frame.

    Raises:
        ValueError: if the frame's length does not describe what follows it,
            which means the bytes are not one of ours.
    """
    if not framed:
        raise ValueError("empty payload carries no compact length prefix")
    mode = framed[0] & 0b11
    if mode == 0b11:
        raise ValueError("big-integer compact lengths are not written by this subnet")
    width = {0b00: 1, 0b01: 2, 0b10: 4}[mode]
    if len(framed) < width:
        raise ValueError(f"payload is {len(framed)} bytes, too short for a {width}-byte prefix")
    declared = int.from_bytes(framed[:width], "little") >> 2
    body = framed[width:]
    if len(body) != declared:
        raise ValueError(f"compact prefix declares {declared} bytes, {len(body)} follow")
    return body
