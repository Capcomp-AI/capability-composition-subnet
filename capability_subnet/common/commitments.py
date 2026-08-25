"""On-chain commitment payloads, and why none of them is a submission.

A miner used to enter a run by writing one string to the chain: the digest of a
recipe, and a pointer to where the bytes were published — a Hugging Face file,
an IPFS object, a URL. Miners submit to the API now. The recipe travels in a
request body signed by their hotkey, with no commitment, no transaction, no fee
and nothing hosted anywhere for the subnet to fetch.

So this module no longer builds commitments and no longer accepts them.
:func:`validate_recipe_uri` refuses every pointer, and both
:func:`encode_commitment` and :func:`decode_commitment` go through it — a miner
cannot form a valid commitment, and one already sitting on the chain does not
decode into a submission.

What is left is the grammar and the workflow-code table, kept because payloads
written before the switch are still on the chain and a reader has to be able to
tell they belong to this subnet before deciding they are not submissions.

Payload grammar::

    capsub1|<workflow_code>|<digest_b64url>|<uri>
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass

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
#: Keyed by each workflow's own id, never by whichever is currently the default —
#: a commitment code is part of the on-chain wire format and must stay valid for
#: every workflow that ever ran, including ones no longer configured.
WORKFLOW_CODES: dict[str, str] = {
    "industrial_maintenance_de_v1": "imde",
    "lora_merger_logic_v1": "lmlg",
}
CODE_TO_WORKFLOW: dict[str, str] = {code: wf for wf, code in WORKFLOW_CODES.items()}

#: No pointer scheme is accepted any more, and that is the whole of the rule.
#:
#: A commitment named a place to fetch a recipe from — a Hugging Face file, an
#: IPFS object, a URL. Miners submit to the API instead: the recipe travels in
#: a signed request body, and there is no pointer for anyone to resolve. A
#: recipe hosted somewhere and named on chain is not a submission, so there is
#: nothing left for this list to hold.
_URI_PATTERNS: tuple[re.Pattern[str], ...] = ()


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
    """Refuse every pointer, because a pointer is no longer a submission.

    Kept as the single place the rule is enforced rather than deleted with the
    patterns: :func:`encode_commitment` and :func:`decode_commitment` both go
    through it, so closing it here closes the route in both directions — a
    miner cannot form a valid commitment, and a commitment already on the chain
    does not decode into one.

    Raises:
        CommitmentError: always. Callers scanning the chain should treat this
            the way they treat any malformed payload — not a submission, skip
            it — which is what they already do.
    """
    raise CommitmentError(
        f"recipe_uri {uri!r} names a place to fetch a recipe from, and the "
        "subnet no longer reads one. Submit to the API instead: the recipe "
        "travels in a request body signed by your hotkey, with no commitment, "
        "no transaction and no fee."
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
