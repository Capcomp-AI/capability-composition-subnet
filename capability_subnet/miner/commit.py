"""Sealing a recipe and committing it on chain.

A submission is one extrinsic: the recipe, compressed and timelocked to the
round its run unseals at, written into the commitments pallet under the miner's
hotkey. Nothing is sent to any service of ours, and nobody - the operator
included - can read it until the chain opens it.

The whole of this module is arranged around one problem: **the chain is a bad
place to learn you made a mistake.** An API refuses in the same breath as the
request and says why. A commitment is accepted by the pallet as long as it is
well-formed bytes, so a recipe naming an adapter that does not exist, or sealed
to the wrong round, is taken by the chain, costs the miner their run, and
surfaces a day later as a line in a report. There is no retry, because by then
the run has closed.

So every check that can be made before the extrinsic is made before the
extrinsic, and every refusal names the limit, the actual value, and the way
out. A miner should never have to read the pallet to understand why we said no.

Sealing is deterministic given the recipe and the run, so ``seal`` takes no
wallet and touches no network - it is the part worth testing exhaustively, and
:func:`commit` is the thin part that signs.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass

from capability_subnet.common import constants as C
from capability_subnet.common import timelock as T

log = logging.getLogger(__name__)

#: Compression level. Recipes are small, structured JSON - the difference
#: between level 9 and the default is a few dozen bytes on a payload that has
#: several hundred to spare, and the cost is microseconds once per run.
COMPRESSION_LEVEL = 9


class CommitError(Exception):
    """A recipe cannot be committed. The message is written for the miner."""


@dataclass(frozen=True, slots=True)
class Sealed:
    """A recipe ready to go on chain, and the sizes it took to get there.

    The sizes are carried because a miner who is near a limit needs to see
    which one, and because ``capcomp commit`` prints them whether or not it is
    about to send anything.
    """

    run_id: int
    recipe_sha256: str
    reveal_round: int
    ciphertext: bytes
    canonical_bytes: int
    compressed_bytes: int

    @property
    def epoch_cost(self) -> int:
        """What this commitment charges against the epoch budget."""
        return max(len(self.ciphertext), C.MIN_COMMIT_SPACE_BYTES)

    @property
    def commits_per_epoch(self) -> int:
        """How many commitments this size fit in one epoch."""
        return C.MAX_EPOCH_COMMIT_BYTES // self.epoch_cost


def seal(recipe, run_id: int, *, run_blocks: int = C.DEFAULT_RUN_BLOCKS) -> Sealed:
    """Compress, frame and timelock a recipe for ``run_id``.

    Pure: no wallet, no chain, no clock. The reveal round comes from the run
    schedule, so two miners sealing the same recipe for the same run produce
    payloads that unseal at the same instant.

    Raises:
        CommitError: if the recipe is too large to commit, or seals to more
            than one commitment can carry. Both messages name the limit and
            what to do about it.
    """
    from capability_subnet.miner.submit import canonical_body, digest_of

    body = canonical_body(recipe)
    if len(body) > C.MAX_ONCHAIN_RECIPE_BYTES:
        raise CommitError(
            f"this recipe is {len(body):,} canonical bytes and the on-chain limit is "
            f"{C.MAX_ONCHAIN_RECIPE_BYTES:,}.\n"
            f"A commitment carries at most {C.MAX_COMMITMENT_FIELDS} fields of "
            f"{C.MAX_TIMELOCK_FIELD_BYTES:,} sealed bytes, and sealing adds a fixed 254 "
            "bytes, so a larger recipe cannot be committed.\n"
            "Shorten it: drop a selected adapter, or shorten output.adapter_name. "
            "`capcomp canonicalise <recipe>` prints the bytes that are counted."
        )

    compressed = zlib.compress(body, COMPRESSION_LEVEL)
    reveal_round = T.reveal_round_for_run(run_id, run_blocks=run_blocks)
    check_round_is_ahead(reveal_round)
    ciphertext = _encrypt(T.frame_payload(compressed), reveal_round)

    # One field, not three. The pallet allows three, but a commitment is written
    # as a single sealed payload here, and a ciphertext larger than one field
    # cannot be stored in one: the chain would refuse the extrinsic. Checking
    # against the field rather than the commitment is what keeps seal() from
    # handing commit() something the chain will not take.
    if len(ciphertext) > C.MAX_TIMELOCK_FIELD_BYTES:
        raise CommitError(
            f"this recipe seals to {len(ciphertext):,} bytes and one commitment field "
            f"holds {C.MAX_TIMELOCK_FIELD_BYTES:,}.\n"
            f"It is {len(body):,} canonical bytes and compressed to "
            f"{len(compressed):,}, which is a poorer ratio than most recipes manage - "
            "sealing then adds a fixed 254 bytes.\n"
            "Recipes compress on their repetition. Weights that repeat across layer "
            "groups cost almost nothing; a distinct six-decimal weight for every "
            "adapter in every group is what pushes one over. Reuse a weight where the "
            "value does not matter, round to fewer decimals, or select fewer adapters."
        )

    return Sealed(
        run_id=run_id,
        recipe_sha256=digest_of(body),
        reveal_round=reveal_round,
        ciphertext=ciphertext,
        canonical_bytes=len(body),
        compressed_bytes=len(compressed),
    )


def check_round_is_ahead(reveal_round: int, *, now: float | None = None) -> None:
    """Refuse to seal to a round drand has already published.

    A recipe sealed to a past round is not sealed at all: its decryption key is
    already public, so the chain unseals it the moment it lands - or refuses it
    outright once the pulse has been pruned. Either way the miner has published
    their recipe rather than committed to it, and nothing in the pallet says so.

    The realistic cause is not a miner picking a silly round. It is running
    against a network whose block heights do not match the run schedule's
    anchors - a testnet, or a stale ``RUN_EPOCH_BLOCK`` - where the derived run
    id is wildly wrong and the round with it. That is worth catching loudly,
    because everything else about the command looks healthy.

    Raises:
        CommitError: if ``reveal_round`` is not still in the future.
    """
    import time

    current = T.round_at(now if now is not None else time.time())
    if reveal_round > current:
        return
    behind = (current - reveal_round) * T.QUICKNET_PERIOD
    raise CommitError(
        f"this recipe would be sealed to drand round {reveal_round}, which passed "
        f"{behind / 3600:,.1f} hours ago (drand is at {current}).\n"
        "A recipe sealed to a past round is public the instant it is committed - its "
        "decryption key is already published - so this would disclose the recipe "
        "rather than commit to it.\n"
        "This normally means the run schedule does not match the network you are "
        "pointed at: check --subtensor.network and --netuid. Nothing was committed."
    )


def _encrypt(payload: bytes, reveal_round: int) -> bytes:
    """Timelock ``payload`` to ``reveal_round``.

    Isolated so the failure has somewhere to be explained. ``bittensor_drand``
    is a compiled extension, and the way it usually fails on a miner's box is
    absence rather than error.

    Raises:
        CommitError: if the timelock library is missing or refuses the payload.
    """
    try:
        import bittensor_drand
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise CommitError(
            "the timelock library is not installed, so this recipe cannot be sealed.\n"
            "Install it with `pip install bittensor` (it ships the bittensor_drand "
            "extension), then run this command again. Nothing was committed."
        ) from exc

    try:
        ciphertext, _ = bittensor_drand.encrypt_at_round(payload, reveal_round)
    except Exception as exc:  # noqa: BLE001 - surfaced to the miner verbatim
        raise CommitError(
            f"the recipe could not be sealed to round {reveal_round}: {exc}\n"
            "Nothing was committed. If this repeats, check that your bittensor "
            "install is current."
        ) from exc
    return bytes(ciphertext)


def run_for_commit(commit_block: int, *, run_blocks: int = C.DEFAULT_RUN_BLOCKS) -> int:
    """The run whose field a commitment made at ``commit_block`` joins.

    Derived rather than asked for, and this is the fix for a whole class of
    miner error rather than a convenience. The reveal round is a property of
    the run, so a recipe sealed for one run and committed into another unseals
    at the wrong time - sealed for a run already closed, it unseals the moment
    it lands, or is refused outright as an expired pulse. Neither is a mistake
    the pallet catches.

    Inside the settling window this is the *next* run, not the current one: a
    commitment made in the last hour is not measured by the run about to open,
    so the field it actually joins is the one after. Sealing follows that,
    which keeps the round and the field in agreement no matter when a miner
    runs the command.
    """
    from capability_subnet.common.chain import measuring_run_for

    return measuring_run_for(commit_block, run_blocks) - 1


def check_not_closing(
    commit_block: int,
    *,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
    cutoff_blocks: int = C.COMMIT_CUTOFF_BLOCKS,
) -> None:
    """Refuse a commitment made too close to the run's close.

    The chain would take it, and it would be sealed and filed correctly - for
    the *next* run, because the settling window has passed. What it also does is
    overwrite whatever this hotkey has sealed for the run now closing, since a
    hotkey holds one commitment and the pallet keeps no history. That one is
    unrecoverable: it never opens, so it is never measured, and nothing reports
    it.

    Waiting until the next run opens costs a miner the remainder of the window
    and keeps both submissions.

    Raises:
        CommitError: if ``commit_block`` is within ``cutoff_blocks`` of the
            close of the run it sits in.
    """
    from capability_subnet.common.chain import run_id_for_block, run_opens_block

    run = run_id_for_block(commit_block, run_blocks)
    close = run_opens_block(run + 1, run_blocks)
    left = close - commit_block
    if left > cutoff_blocks:
        # The other half of the boundary, reached only when the exact check
        # could not read the chain. Distance to this run's close says nothing
        # here: the run before this one has not opened its commitments yet, so
        # a commitment now is a correct entry to this run that destroys that
        # one on the way in.
        since_open = commit_block - run_opens_block(run, run_blocks)
        if since_open >= C.COMMIT_HOLD_AFTER_OPEN_BLOCKS:
            return
        opens = run_opens_block(run, run_blocks) + C.REVEAL_MARGIN_BLOCKS
        raise CommitError(
            f"run {run} opened {since_open} blocks ago and run {run - 1}'s commitments "
            f"open at block {opens}. If this hotkey holds one for run {run - 1}, "
            f"committing now REPLACES it - a hotkey holds exactly one, the pallet keeps "
            f"no history, and a sealed commitment replaced before it opens is never "
            f"measured. Nothing was committed.\n"
            f"  Commit from block "
            f"{run_opens_block(run, run_blocks) + C.COMMIT_HOLD_AFTER_OPEN_BLOCKS}, an "
            f"hour past that, so you are clear of the reveal rather than racing it.\n"
            f"  Pass --run {run} to commit now and accept losing anything held for "
            f"run {run - 1}."
        )

    # Standing exactly MIN_COMMITMENT_AGE_BLOCKS counts as settled:
    # measuring_run_for compares block + the window against the next opening
    # and takes equality as in time. Off by one here would tell a miner their
    # commitment had moved runs when it had not.
    minutes = left * C.BLOCK_SECONDS // 60
    head = (
        f"run {run} closes in {left} blocks (~{minutes} minutes) and this command "
        f"stops {cutoff_blocks} blocks out, so nothing was committed."
    )

    if left >= C.MIN_COMMITMENT_AGE_BLOCKS:
        # The run does not change here: a commitment still joins run `run`, and
        # replacing one already held for that same run is ordinary
        # resubmission, not a loss. What the margin buys is the few blocks
        # between reading the head and the extrinsic landing, which is the only
        # way a miner crosses the protocol cutoff without meaning to.
        raise CommitError(
            f"{head}\n"
            f"  A commitment now would still join run {run} - the protocol's cutoff "
            f"is {C.MIN_COMMITMENT_AGE_BLOCKS} blocks and {left} remain - so this is "
            f"a margin, not a rule. Between reading the block and the extrinsic "
            f"landing you could cross it, and on the other side a commitment joins "
            f"run {run + 1} and destroys whatever this hotkey holds for run {run}.\n"
            f"  Pass --run {run} to commit to run {run} anyway, which is safe while "
            f"{left} blocks remain, or wait for run {run + 1} to open."
        )

    raise CommitError(
        f"{head}\n"
        f"  A commitment now joins run {run + 1}, not run {run}: it has to stand "
        f"{C.MIN_COMMITMENT_AGE_BLOCKS} blocks before the measuring run opens and it "
        f"would stand {left}.\n"
        f"  If this hotkey holds a commitment for run {run}, writing one now DESTROYS "
        f"it - a hotkey holds exactly one, the pallet keeps no history, and a sealed "
        f"commitment replaced before it opens is never measured and never reported. "
        f"Run {run} is already lost to you either way; this decides whether you also "
        f"lose what you have.\n"
        f"  Wait {left} blocks for run {run + 1} to open and commit there, or pass "
        f"--run {run + 1} to commit to run {run + 1} now and accept losing anything "
        f"held for run {run}."
    )


def held_reveal_round(subtensor, netuid: int, hotkey: str) -> int | None:
    """The round a sealed, unopened commitment of this hotkey's opens at.

    ``None`` when the hotkey holds nothing sealed, and ``None`` when the chain
    cannot be read - the caller falls back to a time window there rather than
    treating an unreadable chain as permission.
    """
    from capability_subnet.common.chain import fetch_metagraph

    try:
        view = fetch_metagraph(subtensor, netuid)
    except Exception:  # noqa: BLE001 - the caller degrades to the time window
        return None
    for record in getattr(view, "commitment_records", ()) or ():
        if getattr(record, "hotkey", None) != hotkey:
            continue
        if getattr(record, "status", "") != "sealed":
            return None
        return int(getattr(record, "reveal_round", 0) or 0) or None
    return None


def check_not_overwriting(
    held_round: int | None,
    joins_run: int,
    *,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
) -> None:
    """Refuse a commitment that would destroy a sealed one made for another run.

    A hotkey holds exactly one commitment and the pallet keeps no history, so
    writing a second replaces the first. Replacing one made for the run being
    committed to is ordinary resubmission and the newest is what gets measured.
    Replacing one made for a *different* run is not a replacement at all - the
    miner meant to enter both - and the older never opens and is never measured.

    Thirteen of run 424's sixty-nine entrants were lost that way in a single
    window, eight of them in the hour after the next run opened, where every
    time-based check said yes because the distance to this run's close had gone
    quiet.

    Raises:
        CommitError: if ``held_round`` belongs to a run other than ``joins_run``.
    """
    from capability_subnet.common.timelock import reveal_round_for_run

    if not held_round:
        return
    if held_round == reveal_round_for_run(joins_run, run_blocks=run_blocks):
        return
    raise CommitError(
        f"this hotkey holds a sealed commitment that opens at round {held_round}, and it "
        f"was not made for run {joins_run}. A hotkey holds exactly one commitment and the "
        f"pallet keeps no history, so committing now REPLACES it: that one never opens, is "
        f"never measured, and is never reported. Nothing was committed.\n"
        f"  Wait for it to open - it does so an hour into the run that measures it - and "
        f"commit afterwards. Both submissions survive that way.\n"
        f"  Pass --run {joins_run} to commit anyway and accept losing the sealed one."
    )


def check_window(commit_block: int, run_id: int, *, run_blocks: int = C.DEFAULT_RUN_BLOCKS) -> None:
    """Refuse a commitment that would land in a different run than it is sealed for.

    Only reachable when a miner names a run explicitly. The refusal explains
    what sealing to a stale round would actually do, because "commit anyway"
    is the intuitive move here and it is the one that loses the submission.

    Raises:
        CommitError: if ``run_id`` is not the run this commitment would join.
    """
    joins = run_for_commit(commit_block, run_blocks=run_blocks)
    if joins == run_id:
        return

    from capability_subnet.common.chain import run_opens_block

    close = run_opens_block(run_id + 1, run_blocks)
    stale = joins > run_id
    detail = (
        (
            f"Run {run_id} closed at block {close:,}, so its reveal round has passed. "
            "A recipe sealed to a round in the past is not kept secret at all - the "
            "chain unseals it as soon as it lands, or refuses it as an expired pulse."
        )
        if stale
        else (
            f"Run {run_id} does not open until block "
            f"{run_opens_block(run_id, run_blocks):,}, so a commitment made now would "
            f"be measured before its recipe unsealed, and counted as unreadable."
        )
    )
    raise CommitError(
        f"you asked to submit to run {run_id}, but a commitment made at block "
        f"{commit_block:,} joins run {joins}'s field.\n"
        f"{detail}\n"
        f"Run this again without --run to seal for run {joins}, which is the run this "
        "commitment actually competes in. Nothing was committed."
    )


@dataclass(frozen=True, slots=True)
class Budget:
    """A hotkey's remaining commitment space for the current epoch."""

    used: int
    limit: int
    blocks_to_reset: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def refusal(self, needed: int) -> str:
        """Why this commitment will not fit, and when it will."""
        minutes = self.blocks_to_reset * C.BLOCK_SECONDS / 60
        return (
            f"this hotkey has used {self.used:,} of its {self.limit:,} bytes of commitment "
            f"space this epoch, and this recipe needs {needed:,} more.\n"
            f"The budget is per hotkey per epoch and resets in about {self.blocks_to_reset} "
            f"blocks ({minutes:.0f} minutes). Nothing was committed - run this again then.\n"
            "There is no limit on how many times you may submit; this is the chain's rate "
            "limit, not ours."
        )


def read_budget(subtensor, netuid: int, hotkey: str) -> Budget:
    """This hotkey's remaining commitment space, read from the chain.

    Read before the extrinsic rather than after, so a miner who is out of space
    is told when to come back instead of being handed a pallet error code.

    ``used_space`` is stored with the epoch it was accrued in and is *not*
    zeroed when that epoch ends - the pallet resets it lazily, on the next
    commitment. So a stale epoch means a fresh budget, and reading the number
    without checking the epoch beside it would refuse a miner who has their
    whole allowance.
    """
    from bittensor._generated import storage as st

    def read(item, params, name: str) -> object:
        """One storage read that must succeed.

        Defaulting here would be worse than failing. A missed ``MaxSpace`` read
        substitutes our idea of the limit for the chain's, and a missed epoch
        index makes the used-space number below unreadable - which resolves to
        "no space used", the one answer that lets a miner spend a block on a
        commitment the pallet then rejects.
        """
        try:
            value = subtensor.query(item, params) if params else subtensor.query(item)
        except Exception as exc:  # noqa: BLE001 - reported, never absorbed
            raise CommitError(
                f"could not read {name} from the chain: {exc}\n"
                "Nothing was committed. This is a connection problem rather than a "
                "problem with the recipe - try again, or check --subtensor.network."
            ) from exc
        if value is None:
            raise CommitError(
                f"the chain returned nothing for {name} on netuid {netuid}.\n"
                "Nothing was committed. Check that --netuid names a subnet that "
                "exists on the network you are pointed at."
            )
        return value

    limit = int(read(st.Commitments.MaxSpace, None, "the commitment space limit"))
    epoch = read(st.SubtensorModule.SubnetEpochIndex, [netuid], "the current epoch index")
    tempo = int(read(st.SubtensorModule.Tempo, [netuid], "the subnet tempo"))
    last_epoch_block = int(
        read(st.SubtensorModule.LastEpochBlock, [netuid], "the last epoch boundary")
    )

    # The one read allowed to come back empty, because empty is an answer: a
    # hotkey that has never committed on this subnet has no usage row at all.
    usage = subtensor.query(st.Commitments.UsedSpaceOf, [netuid, hotkey]) or {}
    used = int(usage.get("used_space", 0)) if usage.get("last_epoch") == epoch else 0

    return Budget(
        used=used,
        limit=limit,
        blocks_to_reset=max(0, (last_epoch_block + tempo) - subtensor.block),
    )


def check_budget(budget: Budget, sealed: Sealed) -> None:
    """Refuse before spending a block on a commitment the pallet will reject.

    Raises:
        CommitError: if the epoch budget cannot absorb this commitment.
    """
    if sealed.epoch_cost > budget.remaining:
        raise CommitError(budget.refusal(sealed.epoch_cost))


def commit(subtensor, wallet, netuid: int, sealed: Sealed) -> str:
    """Write ``sealed`` on chain, signed by the hotkey.

    Signed with the hotkey rather than the coldkey: a submission is a statement
    by the neuron, and no coldkey should be unlocked to make one.

    Returns:
        A one-line confirmation naming the run and the round it unseals at.

    Raises:
        CommitError: with the pallet's refusal translated into something a
            miner can act on.
    """
    from bittensor._generated import calls

    call = calls.Commitments.set_commitment(
        netuid=netuid,
        info={
            "fields": [
                [
                    {
                        "TimelockEncrypted": {
                            "encrypted": sealed.ciphertext,
                            "reveal_round": sealed.reveal_round,
                        }
                    }
                ]
            ]
        },
    )

    log.info(
        "committing %d sealed bytes for run %s, unsealing at round %s",
        len(sealed.ciphertext),
        sealed.run_id,
        sealed.reveal_round,
    )
    try:
        result = subtensor.submit_call(call, wallet, signer="hotkey")
    except Exception as exc:  # noqa: BLE001 - translated below
        raise CommitError(_explain(str(exc), netuid)) from exc

    if not bool(getattr(result, "success", False)):
        raise CommitError(_explain(str(getattr(result, "message", "") or ""), netuid))

    return (
        f"committed for run {sealed.run_id}: {sealed.recipe_sha256}\n"
        f"  {sealed.canonical_bytes:,} canonical bytes -> {sealed.compressed_bytes:,} "
        f"compressed -> {len(sealed.ciphertext):,} sealed\n"
        f"  unseals at drand round {sealed.reveal_round}, when run {sealed.run_id} closes\n"
        f"  nobody can read it before then, including us"
    )


def _explain(message: str, netuid: int) -> str:
    """The pallet's refusal, in terms a miner can act on."""
    lowered = message.lower()

    if "spacelimitexceeded" in lowered.replace(" ", ""):
        return (
            "this hotkey has no commitment space left this epoch, so nothing was "
            f"committed.\nThe chain allows {C.MAX_EPOCH_COMMIT_BYTES:,} bytes per hotkey "
            "per epoch and an epoch is about 72 minutes. Wait for the next epoch and run "
            "this again - there is no limit on how many times you may submit overall."
        )
    if "accountnotallowedcommit" in lowered.replace(" ", "") or "notregistered" in lowered.replace(
        " ", ""
    ):
        return (
            f"this hotkey is not registered on netuid {netuid}, so it cannot commit.\n"
            f"Register it with `btcli subnet register --netuid {netuid}`, then run this "
            "again. Nothing was committed."
        )
    if "toomanyfields" in lowered.replace(" ", ""):
        return (
            "the chain refused this commitment for carrying too many fields, which "
            "should not happen for a single sealed recipe.\nNothing was committed. "
            "Please report this with the recipe that caused it."
        )
    if "inability to pay" in lowered or "insufficient" in lowered:
        return (
            "the transaction could not be paid for, so nothing was committed.\n"
            "Committing is free, but the account still needs a small existential "
            "balance. Fund the coldkey and try again."
        )
    return f"the chain refused this commitment: {message}\nNothing was committed."
