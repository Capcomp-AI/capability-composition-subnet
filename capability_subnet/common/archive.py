"""On-chain commitments for published run archives.

A bundle's root digest is what makes the archive checkable. Writing that root
on chain is what makes it checkable *against us*: the chain timestamps the
claim in a block nobody can rewrite, so a bundle that changes after the fact
stops matching a value we can no longer edit.

Deliberately a different payload kind from the miner commitment in
:mod:`capability_subnet.common.commitments`. That one is closed — recipes go to
the API and ``validate_recipe_uri`` refuses every pointer — and reusing its
magic would make an archive commitment decode as a submission in every scanner
that already reads the chain. A distinct magic means those scanners skip these,
which is what they should do.

**What is committed is the digest, not the location.** A repository reference
would be the wrong thing to bind: repositories are mutable, and one that still
resolves after a force-push proves nothing. A digest is a statement about
content, so it holds wherever the content is fetched from — an archive mirrored
to a third site verifies against the same commitment. The location is included
when it fits because it helps somebody find the bundle, and it is advisory: a
verifier that fetched the bytes elsewhere is not worse off.
"""

from __future__ import annotations

import base64
import dataclasses

#: Distinct from PAYLOAD_MAGIC, so a submission scanner skips these.
ARCHIVE_MAGIC = "capsub-a1"
SEPARATOR = "|"

#: The Commitments pallet field this has to fit inside.
MAX_RAW_FIELD_BYTES = 128


class ArchiveCommitmentError(Exception):
    """Raised when an archive payload cannot be built or read."""


@dataclasses.dataclass(frozen=True, slots=True)
class ArchivePayload:
    """A published bundle, as the chain records it."""

    run_id: int
    root_sha256: str
    #: ``owner/repo@revision``, or empty when it did not fit. Advisory.
    location: str = ""


def _digest_to_b64(digest: str) -> str:
    raw = digest.split(":", 1)[1] if ":" in digest else digest
    try:
        packed = bytes.fromhex(raw)
    except ValueError as exc:
        raise ArchiveCommitmentError(f"{digest!r} is not a hex sha256") from exc
    if len(packed) != 32:
        raise ArchiveCommitmentError(f"{digest!r} is {len(packed)} bytes, expected 32")
    return base64.urlsafe_b64encode(packed).decode().rstrip("=")


def _b64_to_digest(encoded: str) -> str:
    padding = "=" * (-len(encoded) % 4)
    try:
        packed = base64.urlsafe_b64decode(encoded + padding)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same answer
        raise ArchiveCommitmentError(f"{encoded!r} is not base64") from exc
    if len(packed) != 32:
        raise ArchiveCommitmentError(f"digest is {len(packed)} bytes, expected 32")
    return "sha256:" + packed.hex()


def encode(run_id: int, root_sha256: str, location: str = "") -> str:
    """Build the payload to write on chain.

    The location is dropped rather than truncated when the field is too small
    for it. A truncated repository name is a pointer to nothing that looks like
    a pointer to something; the digest alone is honest and still sufficient.
    """
    if run_id < 0:
        raise ArchiveCommitmentError("run_id cannot be negative")
    if SEPARATOR in location:
        raise ArchiveCommitmentError(f"location may not contain {SEPARATOR!r}")

    stem = SEPARATOR.join((ARCHIVE_MAGIC, str(run_id), _digest_to_b64(root_sha256)))
    if not location:
        return stem

    full = stem + SEPARATOR + location
    if len(full.encode()) > MAX_RAW_FIELD_BYTES:
        return stem
    return full


def decode(payload: str) -> ArchivePayload:
    parts = (payload or "").strip().split(SEPARATOR)
    if len(parts) < 3 or parts[0] != ARCHIVE_MAGIC:
        raise ArchiveCommitmentError(f"not a {ARCHIVE_MAGIC} commitment")
    try:
        run_id = int(parts[1])
    except ValueError as exc:
        raise ArchiveCommitmentError(f"run id {parts[1]!r} is not an integer") from exc
    return ArchivePayload(
        run_id=run_id,
        root_sha256=_b64_to_digest(parts[2]),
        location=parts[3] if len(parts) > 3 else "",
    )


def looks_like_archive(payload: str) -> bool:
    return (payload or "").strip().startswith(ARCHIVE_MAGIC + SEPARATOR)


def signing_message(run_id: int, root_sha256: str) -> bytes:
    """What the operator signs over a bundle root. Mirrors ops.bundle."""
    return f"capcomp-bundle:v1:{run_id}:{root_sha256}".encode()
