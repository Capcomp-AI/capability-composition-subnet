"""On-chain commitments for published run archives.

A bundle's root digest is what makes the archive checkable. Writing that root
on chain is what makes it checkable *against us*: the chain timestamps the
claim in a block nobody can rewrite, so a bundle that changes after the fact
stops matching a value fixed in a block.

A different payload kind from the miner commitment in
:mod:`capability_subnet.common.commitments`, so that a scanner reading the
chain for submissions skips these and a reader of these skips submissions.
Both directions are asserted by tests.

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


# -- verifying a published bundle -------------------------------------------
#
# These live here, in the package a third party installs, rather than beside
# the builder in the operator's repository. A verifier that had to be obtained
# from the party being verified would be worth very little, and one implemented
# twice would eventually disagree with itself about what a digest covers.

SCHEMA_VERSION = 1

#: Files a repository host adds on its own. Not part of the record, and not
#: cause to report tampering — a check that fails on the first honest fetch
#: teaches people to ignore it.
HOST_FILES = frozenset({".gitattributes", ".gitignore", ".huggingface.yaml"})


def is_host_file(relative: str) -> bool:
    import pathlib

    return relative in HOST_FILES or any(
        part in (".cache", ".git") for part in pathlib.Path(relative).parts
    )


def canonical_bytes(document) -> bytes:
    """The one serialisation any digest here is taken over."""
    import json

    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


#: The manifest fields the root digest covers, in the order they are declared.
#: Listed explicitly so that a field added to a manifest by a newer publisher
#: changes the root rather than being silently ignored by an older verifier.
ROOT_FIELDS = (
    "schema_version",
    "submitted_run",
    "measured_in_run",
    "paid_in_run",
    "beacon",
    "reference_e2e",
    "backfilled",
    "previous_run",
    "previous_root",
    "files",
)


def manifest_root(document: dict) -> str:
    """Recompute a manifest's root from its own contents.

    Recomputed, never read from ``document["root"]``. A bundle edited together
    with its own manifest is internally consistent and passes every per-file
    check; recomputation is what catches it.
    """
    missing = [f for f in ROOT_FIELDS if f not in document]
    if missing:
        raise ArchiveCommitmentError(
            f"manifest is missing {', '.join(missing)}; it was produced by a "
            f"different schema than this verifier understands"
        )
    return sha256_bytes(canonical_bytes({f: document[f] for f in ROOT_FIELDS}))


def verify_bundle(directory, *, expected_root: str = "") -> tuple[bool, list[str]]:
    """Check a fetched bundle against itself, and optionally against a digest.

    ``expected_root`` is the root a caller obtained elsewhere — from the chain,
    or from the ``previous_root`` of the run after this one. Supplying it is
    what turns "this bundle is internally consistent" into "this bundle is the
    one that was published".
    """
    import json
    import pathlib

    directory = pathlib.Path(directory)
    problems: list[str] = []

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return False, [f"no manifest.json in {directory}"]
    document = json.loads(manifest_path.read_text())

    listed = {entry["path"]: entry for entry in document.get("files", [])}
    on_disk = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file()
        and p.name != "manifest.json"
        and not is_host_file(str(p.relative_to(directory)))
    }
    for extra in sorted(on_disk - set(listed)):
        problems.append(f"{extra} is present but not in the manifest")
    for absent in sorted(set(listed) - on_disk):
        problems.append(f"{absent} is in the manifest but not present")

    for name, entry in sorted(listed.items()):
        path = directory / name
        if not path.exists():
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            problems.append(f"{name} hashes to {actual}, manifest says {entry['sha256']}")

    try:
        recomputed = manifest_root(document)
    except ArchiveCommitmentError as exc:
        return False, [*problems, str(exc)]
    if recomputed != document.get("root"):
        problems.append(
            f"root is recorded as {document.get('root')} but recomputes to {recomputed}"
        )
    if expected_root and recomputed != expected_root:
        problems.append(
            f"root {recomputed} does not match the digest this bundle was "
            f"reached by ({expected_root})"
        )

    return not problems, problems
