"""Admission.

Everything a submission has to survive before it costs the engine a GPU-second.
Each check is cheap and each one is decidable from the commitment, the fetched
bytes and the frozen pool — no model is loaded, no instance is generated, nothing
is merged.

Rejection here is final for that submission but not always for that hotkey: a
malformed recipe is simply not admitted, and the miner can correct it and commit
again. What a miner cannot do is get a second *evaluation*; that is the one-shot
rule, and it is enforced once a candidate reaches the queue.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from capability_subnet.backend.monitor.anticopy import CopyVerdict, check_for_copy
from capability_subnet.backend.monitor.fetch import FetchError, fetch_recipe
from capability_subnet.backend.scorer import gates
from capability_subnet.backend.store import Store
from capability_subnet.common.chain import ChainCommitment
from capability_subnet.common.hashing import digests_equal
from capability_subnet.common.schemas import GateVerdict, QueueEntry, Recipe
from capability_subnet.registry.snapshot import PoolSnapshot

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AdmissionResult:
    """What admission decided about one commitment."""

    hotkey: str
    admitted: bool
    reason: str
    recipe: Recipe | None = None
    verdicts: list[GateVerdict] = field(default_factory=list)
    entry: QueueEntry | None = None
    #: The verified bytes, kept so the engine can store its own copy. A champion
    #: is re-measured every window, and re-fetching from the miner's pointer each
    #: time would let a dead host silently cost someone the throne.
    raw_recipe: bytes | None = None

    def failed_gates(self) -> list[str]:
        return [verdict.name for verdict in self.verdicts if not verdict.passed]


def parse_recipe(raw: bytes) -> tuple[Recipe | None, list[str]]:
    """Parse and validate recipe bytes.

    Returns:
        ``(recipe, problems)``. Every validation error is collected, so a miner
        fixing a recipe sees the whole list rather than discovering the next
        problem only after correcting the previous one.
    """
    try:
        document = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return None, ["recipe bytes are not valid UTF-8"]
    except json.JSONDecodeError as exc:
        return None, [f"recipe is not valid JSON: {exc}"]

    if not isinstance(document, dict):
        return None, ["recipe must be a JSON object"]

    try:
        return Recipe.model_validate(document), []
    except Exception as exc:  # noqa: BLE001 - pydantic raises its own type
        problems = _explain_validation_error(exc)
        return None, problems


def _explain_validation_error(exc: Exception) -> list[str]:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return [str(exc)]
    try:
        return [
            f"{'.'.join(str(part) for part in error.get('loc', ())) or '<root>'}: "
            f"{error.get('msg', 'invalid')}"
            for error in errors()
        ]
    except Exception:  # noqa: BLE001
        return [str(exc)]


def evaluate_commitment(
    commitment: ChainCommitment,
    *,
    snapshot: PoolSnapshot,
    store: Store,
    registered_hotkeys: set[str],
    current_block: int,
    fetcher=fetch_recipe,
) -> AdmissionResult:
    """Run every admission gate against one commitment.

    Args:
        commitment: the decoded on-chain submission.
        snapshot: the frozen pool the recipe must match.
        store: used for the anti-copy check against everything already queued.
        registered_hotkeys: hotkeys currently on the metagraph.
        current_block: chain head, recorded as the admission block.
        fetcher: injected so tests and offline replays can supply bytes without
            a network.
    """
    hotkey = commitment.hotkey
    verdicts: list[GateVerdict] = []

    # -- identity -----------------------------------------------------------
    registered = hotkey in registered_hotkeys and commitment.uid is not None
    verdicts.append(gates.gate_identity(registered, hotkey))
    if not registered:
        return AdmissionResult(hotkey, False, "hotkey is not registered", verdicts=verdicts)

    # -- fetch and digest ---------------------------------------------------
    payload = commitment.payload
    try:
        fetched = fetcher(payload.recipe_uri, payload.recipe_sha256)
    except FetchError as exc:
        verdicts.append(gates.gate_digest(False, payload.recipe_sha256, "unavailable"))
        return AdmissionResult(hotkey, False, str(exc), verdicts=verdicts)

    matches = digests_equal(fetched.sha256, payload.recipe_sha256)
    verdicts.append(gates.gate_digest(matches, payload.recipe_sha256, fetched.sha256))
    if not matches:
        return AdmissionResult(
            hotkey, False, "recipe bytes do not match the commitment", verdicts=verdicts
        )

    # -- schema -------------------------------------------------------------
    recipe, problems = parse_recipe(fetched.raw)
    verdicts.append(gates.gate_schema(problems))
    if recipe is None:
        return AdmissionResult(
            hotkey, False, "; ".join(problems[:3]), verdicts=verdicts
        )

    # A recipe's canonical digest is taken over the parsed document, so a miner
    # cannot commit one set of bytes and have a differently-formatted document
    # scored. Re-deriving it here catches that immediately.
    if not digests_equal(recipe.digest(), payload.recipe_sha256):
        verdicts.append(
            gates.gate_digest(False, payload.recipe_sha256, recipe.digest())
        )
        return AdmissionResult(
            hotkey,
            False,
            "the recipe's canonical form does not match the committed digest; "
            "commit the digest of the canonical document, not of arbitrary formatting",
            verdicts=verdicts,
        )

    # -- pool scope ---------------------------------------------------------
    scope_problems = snapshot.validate_recipe_scope(
        recipe.base_revision, recipe.source_snapshot_sha256
    )
    unknown = snapshot.registry.unknown_ids(recipe.selected_adapters)
    verdicts.append(gates.gate_source_pool(unknown))

    if scope_problems or unknown:
        return AdmissionResult(
            hotkey,
            False,
            "; ".join(scope_problems + ([f"unknown adapters: {unknown}"] if unknown else [])),
            recipe=recipe,
            verdicts=verdicts,
        )

    # -- anti-copy ----------------------------------------------------------
    copy_verdict: CopyVerdict = check_for_copy(
        store, hotkey=hotkey, recipe_sha256=recipe.digest(), first_block=commitment.block
    )
    verdicts.append(gates.gate_anti_copy(not copy_verdict.is_copy, copy_verdict.detail))
    if copy_verdict.is_copy:
        return AdmissionResult(hotkey, False, copy_verdict.detail, recipe=recipe, verdicts=verdicts)

    entry = QueueEntry(
        hotkey=hotkey,
        uid=int(commitment.uid),
        recipe_sha256=recipe.digest(),
        recipe_uri=payload.recipe_uri,
        first_block=commitment.block,
        admitted_at_block=current_block,
        status="queued",
    )

    return AdmissionResult(
        hotkey,
        True,
        "admitted",
        recipe=recipe,
        verdicts=verdicts,
        entry=entry,
        raw_recipe=fetched.raw,
    )


def admit_new_commitments(
    commitments: list[ChainCommitment],
    *,
    snapshot: PoolSnapshot,
    store: Store,
    registered_hotkeys: set[str],
    current_block: int,
    fetcher=fetch_recipe,
    recipe_store=None,
) -> list[AdmissionResult]:
    """Process every commitment the engine has not seen before.

    Hotkeys already in the queue are skipped entirely — including terminated
    ones. That is the one-shot rule: a hotkey that has been evaluated and lost
    cannot re-enter by committing something new. A new package needs a new
    hotkey, which costs a registration, which is what makes brute-forcing the
    champion expensive.
    """
    results: list[AdmissionResult] = []

    for commitment in commitments:
        existing = store.get_queue_entry(commitment.hotkey)
        if existing is not None:
            continue

        result = evaluate_commitment(
            commitment,
            snapshot=snapshot,
            store=store,
            registered_hotkeys=registered_hotkeys,
            current_block=current_block,
            fetcher=fetcher,
        )
        results.append(result)

        if result.admitted and result.entry is not None:
            # Persist before queueing. A queue entry whose recipe the engine
            # cannot load is a candidate that can never be evaluated and never
            # terminated — it would occupy the head of the queue forever and
            # block everyone behind it.
            if recipe_store is not None and result.raw_recipe is not None:
                try:
                    recipe_store.store(result.raw_recipe, result.entry.recipe_sha256)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "not admitting %s: its recipe could not be stored (%s). "
                        "It stays eligible and will be retried on the next chain read.",
                        commitment.hotkey[:12],
                        exc,
                    )
                    results[-1] = AdmissionResult(
                        commitment.hotkey,
                        False,
                        f"recipe could not be stored: {exc}",
                        recipe=result.recipe,
                        verdicts=result.verdicts,
                    )
                    continue

            store.upsert_queue_entry(result.entry)
            log.info(
                "admitted %s (recipe %s, committed at block %d)",
                commitment.hotkey[:12],
                result.entry.recipe_sha256[:19],
                commitment.block,
            )
        else:
            log.info(
                "rejected %s: %s (failed: %s)",
                commitment.hotkey[:12],
                result.reason,
                ", ".join(result.failed_gates()) or "none",
            )

    return results
