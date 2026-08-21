"""Turning a chain commitment into a recipe a validator can measure.

A commitment is 128 bytes: a workflow code, a digest, and a pointer. The pointer
is not trusted and does not need to be — the bytes it returns are checked against
the digest that was committed on-chain before the run opened, so a mutable
pointer, a hijacked host and a substituted file all fail identically.

Every validator does this independently. That is the point of an ownerless
network and also its one duplicated cost: N validators fetch the same small JSON
document. It is a few kilobytes, and the alternative is one party fetching it and
everyone else believing them.
"""

from __future__ import annotations

import logging

from capability_subnet.common.commitments import CommitmentError, decode_commitment
from capability_subnet.common.fetch import FetchError, fetch_recipe
from capability_subnet.common.schemas import Recipe

log = logging.getLogger(__name__)


class ResolutionError(Exception):
    """Raised when a commitment cannot be turned into a usable recipe."""


def resolve_recipe(commitment, *, timeout: float = 30.0) -> Recipe:
    """Fetch and verify the recipe a commitment points at.

    Args:
        commitment: a chain commitment carrying ``payload`` (the 128-byte string)
            or already-decoded ``recipe_sha256`` and ``recipe_uri`` fields.

    Raises:
        ResolutionError: for a malformed commitment, an unreachable or oversized
            pointer, a digest mismatch, or bytes that are not a valid recipe.
            Every one of these is the miner's to fix, and the message says which.
    """
    # Three shapes reach this function and they disagree about what `payload`
    # means. A ChainCommitment read from the chain carries the *decoded* payload
    # there and keeps the 128-byte string in `raw`; a caller holding only the
    # string passes it as `payload`; a test may pass the two fields directly.
    # Treating the decoded object as a string is how every real commitment used
    # to raise AttributeError out of a function that documents ResolutionError.
    payload = getattr(commitment, "payload", None)

    if payload is not None and not isinstance(payload, str):
        digest = getattr(payload, "recipe_sha256", "")
        uri = getattr(payload, "recipe_uri", "")
    else:
        text = payload if isinstance(payload, str) else getattr(commitment, "raw", "")
        if text:
            try:
                decoded = decode_commitment(text)
            except CommitmentError as exc:
                raise ResolutionError(f"unreadable commitment: {exc}") from exc
            digest, uri = decoded.recipe_sha256, decoded.recipe_uri
        else:
            digest = getattr(commitment, "recipe_sha256", "")
            uri = getattr(commitment, "recipe_uri", "")

    if not digest or not uri:
        raise ResolutionError("commitment carries no digest or no pointer")

    try:
        fetched = fetch_recipe(uri, digest, timeout=timeout)
    except FetchError as exc:
        raise ResolutionError(str(exc)) from exc

    try:
        return Recipe.model_validate_json(fetched.raw)
    except Exception as exc:
        # The bytes matched the digest, so this is a miner who committed to
        # something that is not a recipe — not a transport problem.
        raise ResolutionError(f"{uri} matched its digest but is not a valid recipe: {exc}") from exc
