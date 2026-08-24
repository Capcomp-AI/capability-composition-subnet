"""Sending a recipe to the submission API.

A miner's whole interaction with the network is this: POST the recipe, signed by
the hotkey. Nothing goes on chain — no commitment, no transaction, no fee — and
the hotkey is used for one thing only, signing a short string that binds the
miner to the exact bytes they sent.

Kept apart from the neuron so the signing and the request can be tested without
a wallet or a network, and so a miner reading this file can see the entire
protocol contract in one screen.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: What the miner signs. The run and the digest are both inside it, so a
#: signature cannot be replayed into a later run or against another recipe.
SIGNING_PREFIX = "capcomp-submit:v1"


class SubmitError(Exception):
    """Raised when a submission cannot be sent or is refused."""


@dataclass(frozen=True, slots=True)
class Accepted:
    """What the API says about a submission it took."""

    run_id: int
    uid: int
    recipe_sha256: str
    submission_count: int
    remaining: int
    replaced: str | None
    unchanged: bool


def digest_of(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def signing_message(run_id: int, recipe_sha256: str) -> bytes:
    return f"{SIGNING_PREFIX}:{run_id}:{recipe_sha256}".encode()


def canonical_body(recipe) -> bytes:
    """The bytes to send, and the bytes that are hashed and signed.

    The protocol's own canonical form, not merely *a* compact one. That makes
    the digest a miner signs identical to the digest the engine identifies the
    recipe by, so there is one number to check rather than two that differ for
    reasons nobody should have to learn.

    Serialised from the parsed recipe rather than read off disk, so a stray
    byte in the miner's editor cannot change what they sign, and what is sent
    is what was validated a moment earlier.
    """
    return recipe.canonical_bytes()


def current_run(api_url: str, hotkey: str, *, timeout: float = 30.0) -> tuple[int, dict]:
    """The run the API would file a submission under, and this hotkey's standing.

    Read from the API rather than computed locally. The run is decided by the
    service from the chain head it can see, and a miner signing for a run the
    service disagrees about would have every submission refused.
    """
    import httpx

    try:
        response = httpx.get(f"{api_url.rstrip('/')}/status/{hotkey}", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise SubmitError(f"could not reach the submission API: {exc}") from exc
    return int(payload["run_id"]), payload


def send(
    api_url: str, hotkey: str, body: bytes, signature: str, *, timeout: float = 60.0
) -> Accepted:
    """POST the recipe. Raises SubmitError with the service's own words."""
    import httpx

    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/submit",
            json={
                "hotkey": hotkey,
                "recipe": body.decode(),
                "signature": signature,
            },
            timeout=timeout,
        )
    except Exception as exc:
        raise SubmitError(f"could not reach the submission API: {exc}") from exc

    if response.status_code != 200:
        raise SubmitError(_explain(response))

    payload: dict[str, Any] = response.json()
    return Accepted(
        run_id=payload["run_id"],
        uid=payload["uid"],
        recipe_sha256=payload["recipe_sha256"],
        submission_count=payload["submission_count"],
        remaining=payload["remaining"],
        replaced=payload.get("replaced"),
        unchanged=bool(payload.get("unchanged")),
    )


def _explain(response) -> str:
    """The refusal, in terms a miner can act on."""
    try:
        detail = response.json().get("detail", "")
    except ValueError:
        detail = (response.text or "")[:200]

    if response.status_code == 401:
        return (
            f"the signature was not accepted: {detail}\n"
            "The signed string must name the run the API is in and the digest of "
            "the exact bytes sent."
        )
    if response.status_code == 403:
        return (
            "this hotkey is not registered on the subnet. Register it with "
            "`btcli subnet register --netuid 103` before submitting."
        )
    if response.status_code == 429:
        return f"no attempts left this run: {detail}"
    if response.status_code == 503:
        return f"the service is not ready: {detail}. Nothing was submitted; try again shortly."
    return f"submission refused ({response.status_code}): {detail}"
