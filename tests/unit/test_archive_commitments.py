"""The on-chain half of the archive.

A commitment has one job: let somebody who does not trust the operator tell
whether a published bundle is the bundle that was published. These check the
payload round-trips, fits the pallet field, and cannot be confused with a
miner submission by anything already reading the chain.
"""

from __future__ import annotations

import hashlib

import pytest

from capability_subnet.common import archive as A
from capability_subnet.common import commitments as C

ROOT = "sha256:" + hashlib.sha256(b"a run's bundle").hexdigest()
LOCATION = "capcomp/subnet103-archive@" + "a" * 40


class TestItRoundTrips:
    def test_a_payload_survives_the_chain(self):
        decoded = A.decode(A.encode(420, ROOT, LOCATION))

        assert decoded.run_id == 420
        assert decoded.root_sha256 == ROOT
        assert decoded.location == LOCATION

    def test_it_round_trips_without_a_location(self):
        decoded = A.decode(A.encode(420, ROOT))

        assert decoded.root_sha256 == ROOT
        assert decoded.location == ""


class TestItFitsTheChain:
    def test_a_full_payload_fits_the_pallet_field(self):
        """128 bytes is the whole budget, and a real repo name has to fit."""
        payload = A.encode(420, ROOT, LOCATION)

        assert len(payload.encode()) <= A.MAX_RAW_FIELD_BYTES
        assert A.decode(payload).location == LOCATION

    def test_an_oversized_location_is_dropped_not_truncated(self):
        """A truncated repo name points at nothing while looking like a pointer.

        The digest is the part that binds, so losing the location costs
        nothing that matters; a half-written one would cost a verifier time
        and then mislead them.
        """
        payload = A.encode(420, ROOT, "some-org/" + "x" * 200)

        assert len(payload.encode()) <= A.MAX_RAW_FIELD_BYTES
        decoded = A.decode(payload)
        assert decoded.location == ""
        assert decoded.root_sha256 == ROOT

    def test_the_digest_survives_at_the_longest_run_id(self):
        payload = A.encode(999999, ROOT, LOCATION)
        assert len(payload.encode()) <= A.MAX_RAW_FIELD_BYTES
        assert A.decode(payload).root_sha256 == ROOT


class TestItIsNotASubmission:
    def test_a_submission_scanner_skips_it(self):
        """The reason this has its own magic.

        Everything already reading the chain looks for the miner payload. An
        archive commitment sharing that magic would decode as a submission and
        appear in fields it has no business in.
        """
        payload = A.encode(420, ROOT, LOCATION)

        assert not C.is_subnet_commitment(payload)
        assert A.looks_like_archive(payload)
        with pytest.raises(C.CommitmentError):
            C.decode_commitment(payload)

    def test_an_archive_reader_skips_a_submission(self):
        with pytest.raises(A.ArchiveCommitmentError):
            A.decode("capsub1|imde|abc|https://example.test/r.json")


class TestItRefusesNonsense:
    @pytest.mark.parametrize(
        "digest",
        ["sha256:not-hex", "sha256:" + "ab" * 16, "", "sha256:"],
    )
    def test_a_digest_that_is_not_a_sha256_is_refused(self, digest):
        with pytest.raises(A.ArchiveCommitmentError):
            A.encode(420, digest)

    def test_a_location_carrying_the_separator_is_refused(self):
        """It would decode as a fourth field and silently change the payload."""
        with pytest.raises(A.ArchiveCommitmentError):
            A.encode(420, ROOT, "org/repo|extra")

    def test_a_truncated_payload_is_refused(self):
        with pytest.raises(A.ArchiveCommitmentError):
            A.decode("capsub-a1|420")


class TestTheSignedBytesMatchTheBuilder:
    def test_the_message_is_the_one_ops_bundle_signs(self):
        """Two repositories, one string.

        The engine signs a bundle root in ops.bundle and an auditor verifies it
        here. If the two ever disagreed about the bytes, every signature would
        fail verification for a reason nobody could see from either side.
        """
        assert A.signing_message(420, ROOT) == f"capcomp-bundle:v1:420:{ROOT}".encode()

    def test_it_is_bound_to_both_the_run_and_the_root(self):
        other_root = "sha256:" + hashlib.sha256(b"a different bundle").hexdigest()

        assert A.signing_message(420, ROOT) != A.signing_message(419, ROOT)
        assert A.signing_message(420, ROOT) != A.signing_message(420, other_root)
