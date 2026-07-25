"""Chain access helpers.

Thin wrappers over the Bittensor SDK so the rest of the codebase never has to
reason about storage layouts or SDK version differences. Two things matter here:

* reading every commitment on the subnet *together with the block it was made
  at* — commit order is what assigns challenger roles, so the block is not
  optional metadata,
* writing weights with sane retry and rate-limit handling.
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


@dataclass(frozen=True, slots=True)
class ChainCommitment:
    """One miner's commitment as read from the chain."""

    hotkey: str
    uid: int | None
    block: int
    raw: str
    payload: CommitmentPayload


def read_commitments(
    subtensor: bt.Subtensor,
    netuid: int,
    *,
    metagraph: bt.Metagraph | None = None,
    min_block: int = 0,
) -> list[ChainCommitment]:
    """Read and decode every commitment belonging to this subnet.

    Malformed and foreign commitments are skipped with a debug log rather than
    raising — a single bad row must never poison a scan.

    Returns:
        Commitments sorted by the block they were made at, earliest first. That
        ordering is the queue order the scheduler uses.
    """
    try:
        raw_commitments: dict[str, str] = subtensor.get_all_commitments(netuid)
    except Exception:
        log.exception("failed to read commitments for netuid %s", netuid)
        return []

    uid_by_hotkey: dict[str, int] = {}
    if metagraph is not None:
        uid_by_hotkey = {hk: uid for uid, hk in enumerate(metagraph.hotkeys)}

    results: list[ChainCommitment] = []
    skipped = 0
    unreadable = 0
    for hotkey, raw in raw_commitments.items():
        if not is_subnet_commitment(raw):
            continue
        try:
            payload = decode_commitment(raw)
        except CommitmentError as exc:
            skipped += 1
            log.debug("skipping malformed commitment from %s: %s", hotkey[:12], exc)
            continue

        block = _commitment_block(subtensor, netuid, hotkey)
        if block is None:
            # Commit order decides who challenges next, so a commitment whose
            # block cannot be read has no defensible place in the queue. Skipping
            # it leaves it eligible for the next scan; guessing a block would
            # either hand it the front of the queue or silently drop it, and both
            # have happened in systems that defaulted this to zero.
            unreadable += 1
            log.warning(
                "skipping %s this pass: its commitment block could not be read", hotkey[:12]
            )
            continue

        if block < min_block:
            continue

        results.append(
            ChainCommitment(
                hotkey=hotkey,
                uid=uid_by_hotkey.get(hotkey),
                block=block,
                raw=raw,
                payload=payload,
            )
        )

    if skipped:
        log.info("skipped %d malformed commitments on netuid %s", skipped, netuid)
    if unreadable:
        # Loud, because a persistent count here means miners are silently unable
        # to enter the queue.
        log.warning(
            "%d commitments on netuid %s had an unreadable block and were not queued this pass",
            unreadable,
            netuid,
        )

    results.sort(key=lambda c: (c.block, c.hotkey))
    return results


def _commitment_block(subtensor: bt.Subtensor, netuid: int, hotkey: str) -> int | None:
    """Block at which ``hotkey``'s current commitment was written.

    Returns:
        The block, or ``None`` when it cannot be determined.

    ``None`` rather than a default. Commit order is what assigns challenger
    roles, so a wrong block is not a cosmetic inaccuracy — it changes who is
    evaluated next. A zero default sorts to the *front* of an ascending queue
    and would hand priority to exactly the commitments the engine understands
    least, while a large default would silently bury them. Neither is
    acceptable, so the caller skips and retries instead.
    """
    try:
        metadata: Any = subtensor.get_commitment_metadata(netuid, hotkey)
    except Exception:
        log.warning("could not read commitment metadata for %s", hotkey[:12], exc_info=True)
        return None

    if not isinstance(metadata, dict):
        log.warning(
            "commitment metadata for %s has an unexpected shape (%s); cannot determine its block",
            hotkey[:12],
            type(metadata).__name__,
        )
        return None

    block = metadata.get("block")
    if block is None:
        log.warning("commitment metadata for %s carries no block", hotkey[:12])
        return None

    try:
        return int(block)
    except (TypeError, ValueError):
        log.warning("commitment block for %s is not an integer: %r", hotkey[:12], block)
        return None


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

    Returns:
        ``(success, message)``. The message is the chain's response on failure
        and the encoded payload on success.
    """
    payload = encode_commitment(workflow_id, recipe_sha256, recipe_uri)
    log.info("committing %d-byte payload on netuid %s", len(payload.encode()), netuid)
    try:
        response = subtensor.set_commitment(wallet=wallet, netuid=netuid, data=payload)
    except Exception as exc:  # noqa: BLE001 - reported to the operator verbatim
        log.exception("set_commitment raised")
        return False, str(exc)

    success = bool(getattr(response, "success", False))
    message = getattr(response, "message", "") or ""
    return (True, payload) if success else (False, message or "set_commitment failed")


def submit_weights(
    subtensor: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
    *,
    version_key: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """Submit a weight vector.

    The SDK's own rate-limit guard returns ``success=False`` with an empty
    message when the validator has set weights too recently. That is a no-op,
    not an error, and is reported as such so callers do not retry in a tight
    loop.
    """
    if len(uids) != len(weights):
        raise ValueError(f"uids/weights length mismatch: {len(uids)} vs {len(weights)}")
    if not uids:
        raise ValueError("refusing to submit an empty weight vector")

    try:
        response = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("set_weights raised")
        return False, str(exc)

    success = bool(getattr(response, "success", False))
    message = getattr(response, "message", "") or ""
    if success:
        return True, "weights set"
    if not message:
        return False, "rate limited"
    return False, message


def current_block(subtensor: bt.Subtensor) -> int:
    """Chain head, or 0 if the endpoint is unreachable."""
    try:
        return int(subtensor.get_current_block())
    except Exception:
        log.warning("could not read chain head", exc_info=True)
        return 0


def window_id_for_block(block: int, window_blocks: int) -> int:
    """Map a block height to its evaluation window index."""
    if window_blocks <= 0:
        raise ValueError("window_blocks must be positive")
    return block // window_blocks
