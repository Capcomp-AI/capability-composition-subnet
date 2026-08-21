"""Chain access helpers.

Thin wrappers over the Bittensor SDK so the rest of the codebase never has to
reason about storage layouts or SDK version differences. Two things matter here:

* reading every commitment on the subnet *together with the block it was made
  at* — commit order is what assigns challenger roles, so the block is not
  optional metadata,
* writing weights with sane retry and rate-limit handling.

This module targets the **Bittensor 11 SDK**, which replaced the method-per-call
``Subtensor`` of the 10.x line with two surfaces: typed *reads* grouped into
namespaces (``subtensor.subnets.metagraph(...)``) and *intents* submitted
through ``subtensor.execute(...)``. The shape of the change matters here rather
than just the spelling: a v11 metagraph carries every neuron's commitment
inline, so the per-hotkey "when was this committed" round trip the 10.x code
needed is gone, and with it the failure mode where a commitment could not be
queued because its block could not be read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from capability_subnet.common.commitments import (
    CommitmentError,
    CommitmentPayload,
    decode_commitment,
    encode_commitment,
    is_subnet_commitment,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for type checkers
    import bittensor as bt

log = logging.getLogger(__name__)

#: Largest payload the Commitments pallet accepts in one ``Raw`` field.
MAX_RAW_FIELD_BYTES = 128


@dataclass(frozen=True, slots=True)
class ChainCommitment:
    """One miner's commitment as read from the chain."""

    hotkey: str
    uid: int | None
    block: int
    raw: str
    payload: CommitmentPayload


@dataclass(frozen=True, slots=True)
class MetagraphView:
    """The subset of a metagraph this subnet actually uses.

    Carried as a plain snapshot rather than a live SDK object so the engine,
    the validator and the tests all read the same shape, and so a metagraph
    fetched once per pass cannot change under a caller mid-decision.
    """

    netuid: int
    block: int
    hotkeys: list[str]
    owner_hotkey: str
    commitments: list[ChainCommitment]

    @property
    def size(self) -> int:
        return len(self.hotkeys)

    def uid_of(self, hotkey: str) -> int | None:
        try:
            return self.hotkeys.index(hotkey)
        except ValueError:
            return None

    def owner_uid(self) -> int | None:
        """UID of the subnet owner's hotkey, when it holds one.

        This is the burn target. UID 0 is *not* a burn address — it is whichever
        neuron happens to occupy the first slot — so routing emission there pays
        a stranger. Resolving the owner from the metagraph is the only way to
        burn to something the operator actually controls.
        """
        return self.uid_of(self.owner_hotkey)


def fetch_metagraph(subtensor: bt.Subtensor, netuid: int) -> MetagraphView:
    """Read the metagraph and every commitment on it in one call.

    Raises:
        ChainError: propagated from the SDK. A caller that cannot read the
            metagraph must not proceed on a stale one without knowing.
    """
    graph = subtensor.subnets.metagraph(netuid=netuid)

    commitments: list[ChainCommitment] = []
    skipped = 0
    sealed = 0

    # Registered neurons first, then commitments whose hotkey has since
    # deregistered. The latter are kept because anti-copy compares against every
    # commitment ever admitted, and a hotkey leaving must not retroactively free
    # its recipe for someone else to claim.
    records: list[tuple[Any, int | None]] = [
        (record, uid) for uid, record in sorted(graph.commitments.items())
    ]
    records += [(record, None) for _, record in sorted(graph.unregistered_commitments.items())]

    for record, uid in records:
        value = record.value
        if value is None:
            # A timelocked payload the chain has not decrypted yet. It is not
            # malformed, it is simply not readable, and it will be on the next
            # pass — so it is neither skipped nor counted against the miner.
            sealed += 1
            continue
        if not is_subnet_commitment(value):
            continue
        try:
            payload = decode_commitment(value)
        except CommitmentError as exc:
            skipped += 1
            log.debug("skipping malformed commitment from %s: %s", record.hotkey[:12], exc)
            continue

        commitments.append(
            ChainCommitment(
                hotkey=record.hotkey,
                uid=uid if uid is not None else record.uid,
                block=int(record.block),
                raw=value,
                payload=payload,
            )
        )

    if skipped:
        log.info("skipped %d malformed commitments on netuid %s", skipped, netuid)
    if sealed:
        log.info("%d commitments on netuid %s are still sealed", sealed, netuid)

    commitments.sort(key=lambda c: (c.block, c.hotkey))

    return MetagraphView(
        netuid=netuid,
        block=int(graph.block),
        hotkeys=list(graph.hotkeys),
        owner_hotkey=str(graph.owner_hotkey),
        commitments=commitments,
    )


def read_commitments(
    subtensor: bt.Subtensor,
    netuid: int,
    *,
    min_block: int = 0,
) -> list[ChainCommitment]:
    """Every commitment belonging to this subnet, in commit order.

    Returns:
        Commitments sorted by the block they were made at, earliest first. That
        ordering is the queue order the scheduler uses.
    """
    try:
        view = fetch_metagraph(subtensor, netuid)
    except Exception:
        log.exception("failed to read commitments for netuid %s", netuid)
        return []
    return [c for c in view.commitments if c.block >= min_block]


def is_registered(view: MetagraphView, hotkey: str) -> bool:
    return hotkey in view.hotkeys


def write_commitment(
    subtensor: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
    *,
    workflow_id: str,
    recipe_sha256: str,
    recipe_uri: str,
) -> tuple[bool, str]:
    """Publish a recipe commitment on-chain.

    v11 exposes no ``set_commitment`` intent, so the Commitments call is
    composed directly and submitted through the raw-call escape hatch. The
    payload goes in a single ``Raw<len>`` field, which is what the pallet's
    ``CommitmentInfo`` expects and what :func:`read_commitments` decodes back.

    Signed with the **hotkey**: a commitment is a statement by the neuron, and
    the default signer is the coldkey.

    Returns:
        ``(success, message)``. The message is the chain's response on failure
        and the encoded payload on success.
    """
    payload = encode_commitment(workflow_id, recipe_sha256, recipe_uri)
    raw = payload.encode("utf-8")

    if len(raw) > MAX_RAW_FIELD_BYTES:
        return False, (
            f"commitment payload is {len(raw)} bytes, above the "
            f"{MAX_RAW_FIELD_BYTES}-byte field limit"
        )

    from bittensor._generated import calls

    log.info("committing %d-byte payload on netuid %s", len(raw), netuid)
    call = calls.Commitments.set_commitment(
        netuid=netuid,
        info={"fields": [[{f"Raw{len(raw)}": raw}]]},
    )

    try:
        result = subtensor.submit_call(call, wallet, signer="hotkey")
    except Exception as exc:  # noqa: BLE001 - reported to the operator verbatim
        log.exception("set_commitment raised")
        return False, str(exc)

    if bool(getattr(result, "success", False)):
        return True, payload
    return False, str(getattr(result, "message", "") or "set_commitment failed")


def submit_weights(
    subtensor: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
    *,
    version_key: int,
) -> tuple[bool, str]:
    """Submit a weight vector.

    v11's ``SetWeights`` intent conforms the vector to the subnet's own
    hyperparameters — max-weight clipping, u16 quantisation, minimum weight
    count — and picks the plaintext or timelocked commit-reveal path from the
    subnet's on-chain configuration. None of that is decided here, which is the
    point: a validator that hard-coded one path would break the day the subnet
    owner enabled the other.

    Rate limiting is reported rather than raised, so callers do not retry in a
    tight loop against a limit that only clears with time.
    """
    if len(uids) != len(weights):
        raise ValueError(f"uids/weights length mismatch: {len(uids)} vs {len(weights)}")
    if not uids:
        raise ValueError("refusing to submit an empty weight vector")

    import bittensor as bt_sdk

    try:
        result = subtensor.execute(
            bt_sdk.SetWeights(netuid=netuid, uids=uids, weights=weights, version_key=version_key),
            wallet,
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if _is_rate_limit(message):
            return False, "rate limited"
        log.exception("set_weights raised")
        return False, message

    if bool(getattr(result, "success", False)):
        return True, "weights set"

    message = str(getattr(result, "message", "") or "set_weights failed")
    return False, "rate limited" if _is_rate_limit(message) else message


def _is_rate_limit(message: str) -> bool:
    """Whether a failure is the chain's own weight rate limit.

    Matched on the message because the SDK surfaces it as an ordinary error.
    Treating it as a failure would have the validator retry every pass until the
    limit cleared; treating it as success would advance state that never landed.
    It is neither, so it gets its own answer.
    """
    lowered = message.lower()
    return "rate limit" in lowered or "settingweightstoofast" in lowered.replace(" ", "")


def current_block(subtensor: bt.Subtensor) -> int:
    """Chain head, or 0 if the endpoint is unreachable."""
    try:
        return int(subtensor.block)
    except Exception:
        log.warning("could not read chain head", exc_info=True)
        return 0


def block_beacon(subtensor: bt.Subtensor, block: int) -> str:
    """The hash of ``block``, to bind a run's instance draw to.

    Instance seeds derive from a root only the operator holds. Mixing in a block
    hash takes the choice of draw away from them: the value is public, it is not
    theirs to pick, and it does not exist until the run opens — so a draw
    cannot be selected after seeing a candidate.

    Returns an empty string when the endpoint cannot supply it. That is a real
    weakening and the sampler logs it rather than pretending otherwise: a draw
    with no beacon rests entirely on the operator's root. It is not fatal,
    because an engine that stopped evaluating whenever the chain hiccuped would
    fail closed against the miners waiting in its queue.
    """
    try:
        info = subtensor.block_info(block)
        for attribute in ("block_hash", "hash", "parent_hash"):
            value = getattr(info, attribute, None) or (
                info.get(attribute) if isinstance(info, dict) else None
            )
            if value:
                return str(value)
    except Exception:
        # block_info reads the block's *state*, which a pruned (non-archive) node
        # discards for old blocks. The beacon only needs the block *hash* — a
        # header field that survives pruning — so fall back to chain_getBlockHash,
        # which returns the identical value on both archive and pruned nodes.
        beacon = _block_hash_via_rpc(subtensor, block)
        if beacon:
            return beacon
        log.warning("could not read block %s for the draw beacon", block, exc_info=True)
        return ""

    log.warning("block %s carried no hash; the draw will not be bound to it", block)
    return ""


def _block_hash_via_rpc(subtensor: bt.Subtensor, block: int) -> str:
    """Fetch a block's hash straight from ``chain_getBlockHash``.

    A header-only lookup that does not touch state, so it works on a pruned node
    where :meth:`Subtensor.block_info` — which reads state — cannot.
    """
    endpoint = getattr(subtensor, "endpoint", None) or getattr(subtensor, "chain_endpoint", None)
    if not endpoint:
        return ""
    url = str(endpoint).replace("ws://", "http://").replace("wss://", "https://")
    try:
        import httpx

        response = httpx.post(
            url,
            json={"id": 1, "jsonrpc": "2.0", "method": "chain_getBlockHash", "params": [block]},
            timeout=15.0,
        )
        response.raise_for_status()
        result = response.json().get("result")
        return str(result) if result else ""
    except Exception:
        log.warning("chain_getBlockHash fallback failed for block %s", block, exc_info=True)
        return ""


def run_id_for_block(block: int, run_blocks: int) -> int:
    """Map a block height to its evaluation run index."""
    if run_blocks <= 0:
        raise ValueError("run_blocks must be positive")
    return block // run_blocks


def measured_in_run(commitment_block: int, run_id: int, run_blocks: int) -> bool:
    """Whether a commitment is the business of this run.

    A commitment is measured once, in the run after the one it was made in.
    Two things follow, and both matter more than they look:

    * a miner earns from one measurement and commits again to earn again, so the
      floor between attempts is a whole run — long enough that iterating on a
      copied recipe costs more than searching properly;
    * a run measures its new arrivals rather than every commitment ever made,
      which is the difference between work that grows with the churn and work
      that grows with the size of the subnet.

    It is derived entirely from the commitment block, which every validator
    reads from the same chain. A validator that restarts, or one that registered
    this morning, selects exactly the same candidates as one that has been
    running for months — no local record of whose turn it has been, and nothing
    to disagree about.

    Raises:
        ValueError: if ``run_blocks`` is not positive.
    """
    if run_blocks <= 0:
        raise ValueError("run_blocks must be positive")
    return commitment_block // run_blocks == run_id - 1


@dataclass(frozen=True, slots=True)
class RunPosition:
    """Where a block sits inside its evaluation run.

    Derived from the block height and the run length alone, so anyone holding
    a chain connection can compute it — a miner deciding whether it is worth
    committing now, a dashboard with no engine behind it, a validator reporting
    what it is working on. Nothing here consults an engine, because in the
    default arrangement each validator evaluates for itself and there is no
    central engine to ask.

    What it deliberately does not claim is a *phase*. Runs are continuous:
    there is no submission cut-off after which the arena stops accepting
    commitments and starts evaluating. A commitment is admitted when it is seen,
    and where a run is in its own span is the only position that exists.
    """

    run_id: int
    opened_block: int
    closes_block: int
    blocks_elapsed: int
    blocks_remaining: int

    @property
    def progress(self) -> float:
        """Fraction of the run elapsed, in ``[0, 1)``."""
        span = self.closes_block - self.opened_block
        return self.blocks_elapsed / span if span else 0.0

    def seconds_remaining(self, block_seconds: float = 12.0) -> float:
        """Wall-clock left in the run, at the chain's nominal block time."""
        return self.blocks_remaining * block_seconds


def run_position(block: int, run_blocks: int) -> RunPosition:
    """Locate ``block`` within its evaluation run.

    Raises:
        ValueError: if ``run_blocks`` is not positive, or ``block`` is
            negative — both of which would otherwise produce a position that
            looks meaningful and is not.
    """
    if run_blocks <= 0:
        raise ValueError("run_blocks must be positive")
    if block < 0:
        raise ValueError("block must not be negative")

    run_id = run_id_for_block(block, run_blocks)
    opened = run_id * run_blocks
    return RunPosition(
        run_id=run_id,
        opened_block=opened,
        closes_block=opened + run_blocks,
        blocks_elapsed=block - opened,
        blocks_remaining=opened + run_blocks - block,
    )
