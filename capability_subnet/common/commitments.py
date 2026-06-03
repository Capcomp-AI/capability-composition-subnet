"""On-chain commitment payloads.

A miner's entire on-chain footprint is one commitment string that says: "the
recipe with this content digest, published at this location, is my submission
for this workflow." The chain stores the digest; the recipe bytes themselves
live off-chain at an immutable, content-addressed location.

The chain's commitment field is small, so the payload is deliberately compact:
the digest is carried as unpadded base64url (43 characters) rather than hex (64),
which leaves useful room for the pointer.

Payload grammar::

    capsub1|<workflow_code>|<digest_b64url>|<uri>

Example::

    capsub1|imde|3q2-7wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|hf:alice/recipes/final.json
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass

from capability_subnet.common import constants as C
from capability_subnet.common.hashing import SHA256_PREFIX, normalise_digest

#: Magic prefix identifying a payload as belonging to this subnet and protocol
#: version. Bump the trailing digit when the grammar changes.
PAYLOAD_MAGIC = "capsub1"

#: Field separator. Chosen because it cannot occur in a base64url digest and is
#: not valid inside the URI forms accepted below.
SEPARATOR = "|"

#: Hard ceiling on the encoded payload. The chain's raw commitment field holds
#: 128 bytes; staying under it is the submitter's responsibility.
MAX_PAYLOAD_BYTES = 128

#: Short codes keep the payload compact. Every workflow the network runs needs
#: an entry here.
WORKFLOW_CODES: dict[str, str] = {
    C.DEFAULT_WORKFLOW_ID: "imde",
}
CODE_TO_WORKFLOW: dict[str, str] = {code: wf for wf, code in WORKFLOW_CODES.items()}

#: Accepted pointer schemes. Each must resolve to immutable bytes: a Hugging
#: Face repo pinned by revision, an IPFS CID, or a plain HTTPS URL whose content
#: is verified against the digest on download.
_URI_PATTERNS = (
    re.compile(r"^hf:[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+/[A-Za-z0-9._/\-]+$"),
    re.compile(r"^ipfs:[A-Za-z0-9]{46,64}$"),
    re.compile(r"^https://[A-Za-z0-9._\-]+(:\d+)?/[A-Za-z0-9._/\-~%?=&]*$"),
)


class CommitmentError(ValueError):
    """Raised when a commitment payload cannot be produced or parsed."""


@dataclass(frozen=True, slots=True)
class CommitmentPayload:
    """A decoded commitment."""

    workflow_id: str
    recipe_sha256: str
    recipe_uri: str

    def encode(self) -> str:
        return encode_commitment(self.workflow_id, self.recipe_sha256, self.recipe_uri)


def _digest_to_b64(digest: str) -> str:
    raw = bytes.fromhex(normalise_digest(digest)[len(SHA256_PREFIX) :])
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64_to_digest(token: str) -> str:
    padding = "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(token + padding)
    except Exception as exc:  # noqa: BLE001 - surfaced as a protocol error
        raise CommitmentError(f"digest is not valid base64url: {token!r}") from exc
    if len(raw) != 32:
        raise CommitmentError(f"digest decodes to {len(raw)} bytes, expected 32")
    return SHA256_PREFIX + raw.hex()


def validate_recipe_uri(uri: str) -> str:
    """Check that ``uri`` is one of the accepted immutable pointer forms."""
    text = uri.strip()
    if not text:
        raise CommitmentError("recipe_uri must not be empty")
    if SEPARATOR in text:
        raise CommitmentError(f"recipe_uri must not contain {SEPARATOR!r}")
    if any(pattern.match(text) for pattern in _URI_PATTERNS):
        return text
    raise CommitmentError(
        f"recipe_uri {uri!r} is not an accepted pointer. Use one of: "
        "hf:<owner>/<repo>/<path>, ipfs:<cid>, or https://<host>/<path>"
    )


def encode_commitment(workflow_id: str, recipe_sha256: str, recipe_uri: str) -> str:
    """Build the payload a miner writes on-chain.

    Raises:
        CommitmentError: for an unknown workflow, a malformed digest or URI, or
            a payload that would not fit in the chain's commitment field.
    """
    code = WORKFLOW_CODES.get(workflow_id)
    if code is None:
        raise CommitmentError(
            f"no commitment code registered for workflow {workflow_id!r}; "
            f"known workflows: {sorted(WORKFLOW_CODES)}"
        )

    payload = SEPARATOR.join(
        (PAYLOAD_MAGIC, code, _digest_to_b64(recipe_sha256), validate_recipe_uri(recipe_uri))
    )

    size = len(payload.encode("utf-8"))
    if size > MAX_PAYLOAD_BYTES:
        overflow = size - MAX_PAYLOAD_BYTES
        raise CommitmentError(
            f"commitment payload is {size} bytes, {overflow} over the {MAX_PAYLOAD_BYTES}-byte "
            f"limit. Shorten the recipe URI by {overflow} characters."
        )
    return payload


def decode_commitment(payload: str) -> CommitmentPayload:
    """Parse an on-chain payload.

    Raises:
        CommitmentError: if the payload does not belong to this subnet or is
            malformed. Callers scanning the chain should treat this as "not a
            submission" and skip, not as a miner failure.
    """
    text = (payload or "").strip()
    if not text:
        raise CommitmentError("empty commitment")

    parts = text.split(SEPARATOR)
    if len(parts) != 4:
        raise CommitmentError(f"expected 4 fields, found {len(parts)}")

    magic, code, digest_token, uri = parts
    if magic != PAYLOAD_MAGIC:
        raise CommitmentError(f"not a {PAYLOAD_MAGIC} commitment (magic={magic!r})")

    workflow_id = CODE_TO_WORKFLOW.get(code)
    if workflow_id is None:
        raise CommitmentError(f"unknown workflow code {code!r}")

    return CommitmentPayload(
        workflow_id=workflow_id,
        recipe_sha256=_b64_to_digest(digest_token),
        recipe_uri=validate_recipe_uri(uri),
    )


def is_subnet_commitment(payload: str) -> bool:
    """Cheap pre-filter used when scanning every commitment on a subnet."""
    return (payload or "").strip().startswith(PAYLOAD_MAGIC + SEPARATOR)
