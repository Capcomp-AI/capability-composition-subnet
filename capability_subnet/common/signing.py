"""Signing and verification of published artifacts.

The evaluation engine is centralised, so everything it publishes is signed with
the operator's hotkey. A validator that fetches a weight vector verifies three
things before touching the chain: the signature is valid, the signer is on the
operator allow-list it was configured with, and the payload is internally
consistent. That combination is what makes a centralised engine auditable —
a fabricated report cannot be attributed to the operator, and a real one cannot
be repudiated.
"""

from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class Signable(Protocol):
    """Anything that can produce the canonical bytes covered by a signature."""

    def signable_bytes(self) -> bytes: ...


class SignatureError(Exception):
    """Raised when a payload fails verification."""


def sign_payload(keypair, payload: Signable) -> str:
    """Sign ``payload`` with a substrate keypair; return a hex signature."""
    signature = keypair.sign(payload.signable_bytes())
    return signature.hex() if isinstance(signature, (bytes, bytearray)) else str(signature)


def verify_payload(payload: Signable, signature: str, signer_ss58: str) -> bool:
    """Verify a hex signature over ``payload`` against ``signer_ss58``.

    Returns False rather than raising on any malformed input, so a caller
    scanning many payloads can simply drop the bad ones.
    """
    if not signature or not signer_ss58:
        return False
    try:
        from bittensor_wallet import Keypair
    except ImportError:  # pragma: no cover - only in environments without the wallet lib
        from substrateinterface import Keypair  # type: ignore[no-redef]

    try:
        keypair = Keypair(ss58_address=signer_ss58)
        raw = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
        return bool(keypair.verify(payload.signable_bytes(), raw))
    except Exception:
        log.debug("signature verification failed for signer %s", signer_ss58[:12], exc_info=True)
        return False


def sign_in_place(keypair, payload) -> None:
    """Attach ``signature`` and ``signer_hotkey`` to a report or weight vector.

    Both fields are excluded from the signed bytes, so signing is idempotent:
    re-signing an already-signed payload produces the same signature.
    """
    payload.signature = sign_payload(keypair, payload)
    payload.signer_hotkey = keypair.ss58_address


def require_trusted_signature(payload, allowed_signers: set[str] | None) -> None:
    """Enforce that ``payload`` carries a valid signature from an allowed signer.

    Passing ``allowed_signers=None`` disables the check entirely. That is only
    appropriate for local development; a validator on a live network must always
    pin the operator hotkeys it trusts.

    Raises:
        SignatureError: if the payload is unsigned, the signature is invalid, or
            the signer is not on the allow-list.
    """
    if allowed_signers is None:
        log.warning(
            "signature enforcement is disabled — this is only safe for local development"
        )
        return

    signer = getattr(payload, "signer_hotkey", None)
    signature = getattr(payload, "signature", None)

    if not signature or not signer:
        raise SignatureError("payload is unsigned")
    if signer not in allowed_signers:
        raise SignatureError(
            f"signer {signer} is not in the configured operator allow-list "
            f"({len(allowed_signers)} entries)"
        )
    if not verify_payload(payload, signature, signer):
        raise SignatureError(f"signature from {signer} does not verify")
