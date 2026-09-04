"""A revealed commitment becomes a submission only if every check passes.

The chain accepts bytes. It does not care whether they decompress, parse, hash
to what they claim, or unseal at the right moment - so everything that made the
submission API a gate has to be done here instead, on data a miner chose and
nobody vetted.

These are written from the attacker's side rather than the happy path's: what
can be put in a commitment that should not become a submission, and does it get
refused with a reason the miner can act on.
"""

from __future__ import annotations

import json
import zlib

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common import sealed
from capability_subnet.common import timelock as T
from capability_subnet.miner.recipe import new_recipe
from capability_subnet.miner.submit import canonical_body

RUN = 900
HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


@pytest.fixture(scope="module")
def recipe():
    from capability_subnet.registry.snapshot import load_snapshot

    return new_recipe(list(load_snapshot().adapter_ids)[:2])


@pytest.fixture(scope="module")
def sealed_payload(recipe):
    """What the chain publishes: the compressed body, frame already eaten."""
    return zlib.compress(canonical_body(recipe), 9)


def _unseal(payload, *, run_id=RUN, reveal_round=None, framed=False):
    return sealed.unseal(
        payload,
        hotkey=HOTKEY,
        uid=7,
        block=1,
        reveal_round=reveal_round if reveal_round is not None else T.reveal_round_for_run(run_id),
        run_id=run_id,
        framed=framed,
    )


class TestAGoodCommitmentBecomesASubmission:
    def test_it_round_trips_to_the_same_digest_the_miner_sealed(self, recipe, sealed_payload):
        from capability_subnet.miner import commit as chain

        got = _unseal(sealed_payload)
        assert got.recipe_sha256 == chain.seal(recipe, RUN).recipe_sha256

    def test_the_digest_is_over_the_canonical_form_not_the_bytes_sealed(self, recipe):
        """Two spellings of one recipe are one submission.

        The anti-copy check compares digests, so a miner who reformats their
        own recipe must not thereby produce a second identity for it.
        """
        spaced = json.dumps(json.loads(canonical_body(recipe)), indent=4).encode()
        assert spaced != canonical_body(recipe)
        assert (
            _unseal(zlib.compress(spaced)).recipe_sha256
            == _unseal(zlib.compress(canonical_body(recipe))).recipe_sha256
        )

    def test_a_framed_payload_is_unframed_when_asked(self, sealed_payload):
        """The direct storage read keeps the frame; the metagraph eats it."""
        framed = T.frame_payload(sealed_payload)
        assert _unseal(framed, framed=True).recipe_sha256 == _unseal(sealed_payload).recipe_sha256


class TestTheRevealRoundDecidesWhichRunItIs:
    def test_a_round_from_another_run_is_not_a_submission_to_this_one(self, sealed_payload):
        with pytest.raises(sealed.NotThisRun, match="not run 900's pinned round"):
            _unseal(sealed_payload, reveal_round=T.reveal_round_for_run(RUN + 1))

    def test_a_round_that_already_passed_is_not_this_run_s(self, sealed_payload):
        """It was never sealed: the key was public when it landed."""
        with pytest.raises(sealed.NotThisRun, match="not run 900's pinned round"):
            _unseal(sealed_payload, reveal_round=T.reveal_round_for_run(RUN - 5))

    def test_a_round_off_by_one_is_rejected_like_any_other(self, sealed_payload):
        with pytest.raises(sealed.NotThisRun):
            _unseal(sealed_payload, reveal_round=T.reveal_round_for_run(RUN) + 1)

    def test_a_wrong_round_is_not_a_refusal(self, sealed_payload):
        """Belonging to another run is not a fault, and must not read as one.

        A field is built by asking each source run in turn, so every commitment
        is offered to a run it does not belong to at least once. Reporting that
        as a refusal marked every sound submission rejected once a pass, and
        the queue's status is a miner's only record of why a run was lost.
        """
        assert not issubclass(sealed.NotThisRun, sealed.SealedError)


class TestWhatElseCanBePutInACommitment:
    """The pallet stores bytes. These are the bytes worth refusing."""

    def test_random_bytes_are_refused_rather_than_crashing(self):
        with pytest.raises(sealed.SealedError, match="not a compressed recipe"):
            _unseal(b"\x00\xff" * 40)

    def test_valid_json_that_is_not_a_recipe_is_refused(self):
        with pytest.raises(sealed.SealedError, match="do not parse as a recipe"):
            _unseal(zlib.compress(json.dumps({"hello": "world"}).encode()))

    def test_a_recipe_naming_an_adapter_that_does_not_exist_is_refused(self, recipe):
        body = json.loads(canonical_body(recipe))
        body["selected_adapters"] = ["not-a-real-adapter-v1"]
        with pytest.raises(sealed.SealedError, match="do not parse as a recipe"):
            _unseal(zlib.compress(json.dumps(body).encode()))

    def test_a_zip_bomb_is_refused_rather_than_inflated(self):
        """The ciphertext is capped by the pallet; the plaintext is not.

        A reader that decompresses first and checks the size second has already
        spent the memory by the time it could object, and the ratio is the
        miner's to choose.
        """
        # Sized to what a commitment can actually carry. zlib manages about a
        # thousand to one on a run of identical bytes, so one field of
        # ciphertext buys roughly a megabyte of plaintext - past the bound, and
        # nowhere near enough to matter if it were not checked.
        bomb = zlib.compress(b"\x00" * (sealed.MAX_DECOMPRESSED_BYTES * 2), 9)
        assert len(bomb) <= C.MAX_COMMITMENT_FIELDS * C.MAX_TIMELOCK_FIELD_BYTES, (
            "a bomb this test cannot fit on chain proves nothing"
        )
        with pytest.raises(sealed.SealedError, match="expands past"):
            _unseal(bomb)

    def test_an_empty_payload_is_refused(self):
        with pytest.raises(sealed.SealedError):
            _unseal(b"")

    def test_a_truncated_frame_is_refused(self, sealed_payload):
        framed = T.frame_payload(sealed_payload)
        with pytest.raises(sealed.SealedError, match="length prefix does not describe"):
            _unseal(framed[:-5], framed=True)


class TestBuildingTheFieldFromWhatTheChainHolds:
    class _Record:
        def __init__(self, hotkey, payload, reveal_round, uid=7, block=10):
            self.hotkey, self.uid, self.reveal_round = hotkey, uid, reveal_round
            self.revealed = [(block, "0x" + payload.hex())]

    def test_a_good_commitment_is_admitted(self, sealed_payload):
        rec = self._Record(HOTKEY, sealed_payload, T.reveal_round_for_run(RUN))
        admitted, refused = sealed.field_from_commitments([rec], RUN)
        assert [r.hotkey for r in admitted] == [HOTKEY]
        assert refused == []

    def test_a_bad_one_is_refused_with_a_reason_rather_than_dropped(self):
        """A refusal nobody hears about is a run lost for no stated reason."""
        rec = self._Record(HOTKEY, b"not compressed", T.reveal_round_for_run(RUN))
        admitted, refused = sealed.field_from_commitments([rec], RUN)
        assert admitted == []
        assert len(refused) == 1 and "not a compressed recipe" in refused[0][1]

    def test_a_deregistered_hotkey_is_refused_and_said_so(self, sealed_payload):
        rec = self._Record(HOTKEY, sealed_payload, T.reveal_round_for_run(RUN))
        admitted, refused = sealed.field_from_commitments([rec], RUN, registered=set())
        assert admitted == []
        assert "no longer registered" in refused[0][1]

    def test_a_commitment_that_has_not_revealed_is_neither(self, sealed_payload):
        """Still sealed is not yet a submission, and not a refusal either."""

        class Pending:
            hotkey, uid, reveal_round, revealed = HOTKEY, 7, T.reveal_round_for_run(RUN), []

        admitted, refused = sealed.field_from_commitments([Pending()], RUN)
        assert admitted == [] and refused == []

    def test_one_run_does_not_admit_another_run_s_commitments(self, sealed_payload):
        rec = self._Record(HOTKEY, sealed_payload, T.reveal_round_for_run(RUN))
        admitted, refused = sealed.field_from_commitments([rec], RUN + 1)
        assert admitted == []
        # Skipped, not refused: it is a sound commitment to a different run.
        assert refused == []

    def test_the_settling_rule_is_not_applied_here(self, sealed_payload):
        """Deliberately the caller's job.

        Which run measures a commitment is decided on its block by the same
        helpers the API path uses. Duplicating that rule here would be a second
        copy of it to drift.
        """
        assert not hasattr(sealed, "measuring_run_for")
        assert "settling" not in sealed.unseal.__doc__.lower()


class TestTheLimitsAreTheChainsOwn:
    def test_the_decompression_bound_is_far_above_any_real_recipe(self, sealed_payload):
        assert sealed.MAX_DECOMPRESSED_BYTES > C.MAX_ONCHAIN_RECIPE_BYTES * 100
        assert len(sealed_payload) < C.MAX_TIMELOCK_FIELD_BYTES
