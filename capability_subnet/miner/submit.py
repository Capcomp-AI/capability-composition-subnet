"""What a recipe *is*, and what it hashes to.

Two pure functions. ``canonical_body`` decides what a recipe is and
``digest_of`` what it hashes to, and the whole protocol agrees on both: the
miner seals those bytes, the engine identifies the recipe by that digest, the
validator checks it again, and the archive records it.

They live here rather than in :mod:`capability_subnet.miner.commit` because
they are not the commit path's to own. Every side of the protocol needs the
same answer, and a definition kept next to one caller drifts toward that
caller.
"""

from __future__ import annotations

import hashlib

__all__ = ["canonical_body", "digest_of"]


def digest_of(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def canonical_body(recipe) -> bytes:
    """The bytes that are sealed, and the bytes that are hashed.

    The protocol's own canonical form, not merely *a* compact one. That makes
    the digest a miner commits identical to the digest the engine identifies
    the recipe by, so there is one number to check rather than two that differ
    for reasons nobody should have to learn.

    Serialised from the parsed recipe rather than read off disk, so a stray
    byte in the miner's editor cannot change what goes on chain: what is sealed
    is what was validated a moment earlier.
    """
    return recipe.canonical_bytes()
