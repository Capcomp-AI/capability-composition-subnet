"""Reading a run's field out of the chain's revealed commitments.

A miner seals a recipe to the drand round its run closes at and writes it into
the commitments pallet. The chain decrypts it on its own when that round
arrives - the operator has no part in it and cannot hold it back - and the
plaintext lands in public storage where anyone can read it.

This is the reader for that. It takes what the chain hands over and returns the
submissions a run is entitled to measure, having checked every one of them, or
refuses it with a reason that says which check failed and what the miner should
do about it.

Four things are checked and none of them is optional:

**The reveal round is the one the run pins.** It travels inside the commitment,
so the miner supplies it, and a wrong one is not a formatting slip: a round
already past means the recipe was public the moment it landed, and a round in
the future means the run it was meant for cannot read it. Neither is a
submission to this run.

**The frame is intact.** A payload that does not carry its own length is not
one of ours, and one whose length disagrees with what follows it has been
truncated somewhere between the miner and here.

**The body is a recipe.** Decompressed, parsed, and canonicalised - a
commitment holding arbitrary bytes is bytes, not a submission.

**The digest matches the canonical form.** Not the bytes as sealed: the
canonical form of what was parsed. Two recipes that differ only in whitespace
seal differently and reconstruct identically, and the digest a run is identified
by has to be the second one.

What this module deliberately does not do is decide *which* run a commitment
belongs to. That is the settling rule, and it is applied on the commitment's
block by the caller, from the same helpers the API path uses.
"""

from __future__ import annotations

import hashlib
import logging
import zlib
from dataclasses import dataclass

from capability_subnet.common import constants as C
from capability_subnet.common import timelock as T

log = logging.getLogger(__name__)

#: The largest plaintext a commitment can hold, decompressed. A sealed field is
#: bounded, but zlib is not: a few hundred bytes of ciphertext can carry a
#: payload that expands without limit, and a reader that decompresses first and
#: checks second has already spent the memory.
MAX_DECOMPRESSED_BYTES = 1024 * 1024


class SealedError(Exception):
    """A revealed commitment is not a submission. The message says why."""


@dataclass(frozen=True, slots=True)
class Revealed:
    """One commitment the chain has opened, checked and parsed."""

    hotkey: str
    uid: int | None
    block: int
    reveal_round: int
    recipe_sha256: str
    recipe_bytes: bytes


def unseal(
    payload: bytes,
    *,
    hotkey: str,
    uid: int | None,
    block: int,
    reveal_round: int,
    run_id: int,
    framed: bool = True,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
) -> Revealed:
    """Turn one revealed payload into a checked submission.

    Args:
        payload: the plaintext the chain published.
        framed: whether ``payload`` still carries its SCALE compact length. A
            direct storage read does; the metagraph's decoder has already eaten
            it. Getting this wrong is the failure that reads as a miner
            submitting garbage, so it is a parameter rather than a guess.
        run_id: the run this commitment is claimed to belong to, used to check
            the reveal round against the one that run pins.

    Raises:
        SealedError: on any failed check, naming the check.
    """
    # Two ways to bind a commitment to a run, because the pallet reports
    # different things before and after it opens one. While sealed it carries
    # the reveal round, and that is compared exactly. Once opened it reports no
    # round at all - and no commit block either, `block` becoming the block the
    # reveal landed at - so the run is recovered from when it opened instead.
    #
    # Not a relaxation. A run pins one round, the chain opens at that round and
    # no other, and consecutive runs are a day apart; the block a commitment
    # opened at names its run as surely as the round did. Requiring the round
    # here refused every commitment in the first run that ever reached this
    # code with real revealed data - all 41 of them, for a round the pallet had
    # already discarded.
    if reveal_round:
        try:
            T.check_reveal_round(run_id, reveal_round, run_blocks=run_blocks)
        except T.RevealRoundError as exc:
            raise SealedError(str(exc)) from exc
    elif not T.opened_in_run(block, run_id, run_blocks=run_blocks):
        raise SealedError(
            f"opened at block {block}, which is not when run {run_id}'s "
            f"commitments open; this commitment is not a submission to run {run_id}"
        )

    if framed:
        try:
            payload = T.unframe_payload(payload)
        except ValueError as exc:
            raise SealedError(
                f"{hotkey[:12]}… sealed a payload whose length prefix does not describe "
                f"it ({exc}); it was truncated or was never one of ours"
            ) from exc

    body = _decompress(payload, hotkey)
    recipe_sha256, canonical = _canonicalise(body, hotkey)

    return Revealed(
        hotkey=hotkey,
        uid=uid,
        block=block,
        reveal_round=reveal_round,
        recipe_sha256=recipe_sha256,
        recipe_bytes=canonical,
    )


def _decompress(payload: bytes, hotkey: str) -> bytes:
    """Inflate a sealed payload, bounded.

    Bounded because the ratio is the miner's to choose. The ciphertext is
    capped by the pallet, the plaintext is not, and a reader that inflates
    first has already committed the memory by the time it could object.
    """
    try:
        machine = zlib.decompressobj()
        body = machine.decompress(payload, MAX_DECOMPRESSED_BYTES)
    except zlib.error as exc:
        raise SealedError(
            f"{hotkey[:12]}… sealed something that is not a compressed recipe: {exc}"
        ) from exc

    if machine.unconsumed_tail:
        raise SealedError(
            f"{hotkey[:12]}… sealed a payload that expands past "
            f"{MAX_DECOMPRESSED_BYTES:,} bytes; a recipe is a few hundred"
        )
    return body


def _canonicalise(body: bytes, hotkey: str) -> tuple[str, bytes]:
    """Parse the body as a recipe and return its canonical digest and bytes.

    The digest is taken over the canonical form rather than over what was
    sealed. A miner who seals the same recipe with different whitespace must
    arrive at the same identity, because the anti-copy check compares digests
    and two spellings of one recipe are one submission.
    """
    from capability_subnet.common.schemas import Recipe

    try:
        recipe = Recipe.model_validate_json(body)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same answer
        raise SealedError(
            f"{hotkey[:12]}… sealed {len(body):,} bytes that do not parse as a recipe: "
            f"{str(exc).splitlines()[0]}"
        ) from exc

    canonical = recipe.canonical_bytes()
    return "sha256:" + hashlib.sha256(canonical).hexdigest(), canonical


def field_from_commitments(
    records,
    run_id: int,
    *,
    registered: set[str] | None = None,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
) -> tuple[list[Revealed], list[tuple[str, str]]]:
    """Every submission ``run_id`` is entitled to measure, and every refusal.

    Args:
        records: the metagraph's commitment records. Their payloads have
            already had the frame stripped by the SDK's decoder, which is why
            ``framed=False`` below.
        registered: hotkeys on the subnet now. A commitment from a hotkey that
            has deregistered is dropped rather than measured - there is nobody
            left to pay - but it is reported, because silently shrinking a
            field is how a run of nothing looks like a run of nobody.

    Returns:
        ``(admitted, refused)``, where each refusal is ``(hotkey, reason)``.
        Both are returned because a refusal a miner never hears about is a run
        they lost for a reason nobody can tell them.
    """
    admitted: list[Revealed] = []
    refused: list[tuple[str, str]] = []

    for record in records:
        hotkey = getattr(record, "hotkey", "")
        reveals = getattr(record, "revealed", None)
        if not reveals:
            continue

        if registered is not None and hotkey not in registered:
            refused.append((hotkey, "hotkey is no longer registered on this subnet"))
            continue

        block, data = reveals[-1]
        payload = (
            bytes.fromhex(data[2:])
            if isinstance(data, str) and data.startswith("0x")
            else (data if isinstance(data, bytes) else str(data).encode())
        )
        try:
            admitted.append(
                unseal(
                    payload,
                    hotkey=hotkey,
                    uid=getattr(record, "uid", None),
                    block=int(block),
                    reveal_round=int(getattr(record, "reveal_round", 0) or 0),
                    run_id=run_id,
                    framed=False,
                    run_blocks=run_blocks,
                )
            )
        except SealedError as exc:
            refused.append((hotkey, str(exc)))

    if refused:
        log.info(
            "run %s: %d commitment(s) refused, %d admitted", run_id, len(refused), len(admitted)
        )
    return admitted, refused
