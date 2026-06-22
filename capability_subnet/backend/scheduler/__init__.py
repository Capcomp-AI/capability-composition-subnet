"""Challenger scheduling.

Role assignment is mechanical: the queue is ordered by the block each commitment
was made at, and the head is the challenger. Nobody — not the operator, not a
miner, not a validator — chooses who challenges next.

The queue itself lives in the store, so scheduling is a query rather than a
process. This module exposes it under the name the architecture uses.
"""

from capability_subnet.backend.store import Store


def next_challenger(store: Store):
    """The earliest admitted submission still waiting to be evaluated."""
    return store.next_challenger()


def queue_depth(store: Store) -> int:
    """How many challengers are waiting."""
    return len(store.list_queue("queued"))


def position_of(store: Store, hotkey: str) -> int | None:
    """A miner's place in line, or ``None`` if it is not waiting.

    Published so a miner can tell "not yet evaluated" apart from "evaluated and
    terminated" without guessing.
    """
    for index, entry in enumerate(store.list_queue("queued")):
        if entry.hotkey == hotkey:
            return index
    return None


__all__ = ["next_challenger", "position_of", "queue_depth"]
