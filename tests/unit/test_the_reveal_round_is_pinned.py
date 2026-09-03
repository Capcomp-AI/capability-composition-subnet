"""The reveal round is derived, identically, by everyone who needs it.

A miner encrypts to a round; the engine refuses any commitment carrying a
different one. Nothing coordinates the two - they compute it from the same
constants, days apart, on different machines. So what these tests protect is
agreement: that the arithmetic is a pure function of the constants, that it
matches drand's own definition of a round, and that the one direction which
loses the subnet its embargo is the one the margin is aimed at.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common import timelock as T


class TestRoundsMatchDrand:
    """Our arithmetic and the SDK's must not diverge, ever.

    The constants are mirrored rather than imported so that an auditor can
    recompute a reveal round without a bittensor install. Mirroring is only safe
    while it stays true, which is what this pins.
    """

    def test_genesis_and_period_match_the_sdk(self):
        sdk = pytest.importorskip("bittensor.timelock")
        assert T.QUICKNET_GENESIS == sdk.QUICKNET_GENESIS
        assert T.QUICKNET_PERIOD == sdk.QUICKNET_PERIOD

    def test_round_at_matches_the_sdk_across_a_period_boundary(self):
        """Off-by-one at the boundary is the whole risk, so walk over one."""
        sdk = pytest.importorskip("bittensor.timelock")
        for offset in range(0, 40):
            when = T.QUICKNET_GENESIS + offset
            assert T.round_at(when) == sdk.round_at(when), f"diverged at +{offset}s"

    def test_round_at_never_lands_before_the_time_asked_for(self):
        """The rounding direction is a security property, not a preference.

        A round chosen for "unseal at or after T" that resolved to a round
        emitted *before* T would unseal the field early. Rounding up is what
        makes the margin a floor rather than an approximation.
        """
        for offset in range(0, 30):
            when = T.QUICKNET_GENESIS + 1_000_000 + offset
            assert T.round_time(T.round_at(when)) >= when

    def test_round_time_inverts_round_at_on_exact_rounds(self):
        for reveal_round in (1, 2, 1000, 31_881_388):
            assert T.round_at(T.round_time(reveal_round)) == reveal_round

    def test_a_time_before_genesis_is_refused(self):
        with pytest.raises(ValueError, match="precedes quicknet genesis"):
            T.round_at(T.QUICKNET_GENESIS - 1)


class TestTheRoundFollowsTheRunSchedule:
    """It rides the anchored schedule rather than carrying its own."""

    def test_each_run_unseals_one_run_length_after_the_last(self):
        step = C.DEFAULT_RUN_BLOCKS * C.BLOCK_SECONDS // T.QUICKNET_PERIOD
        for run in (420, 421, 422, 500):
            assert T.reveal_round_for_run(run + 1) - T.reveal_round_for_run(run) == step

    def test_the_round_lands_after_the_run_closes(self):
        """The margin, restated as the property it exists to hold.

        If this ever inverts, recipes unseal while the run they belong to is
        still accepting submissions, and the last miner to commit gets to read
        the field first.
        """
        from capability_subnet.common.chain import run_opens_block

        for run in (415, 421, 430):
            close = run_opens_block(run + 1, C.DEFAULT_RUN_BLOCKS)
            assert T.round_time(T.reveal_round_for_run(run)) > T.block_instant(close)

    def test_the_margin_clears_the_submission_cutoff_by_an_hour(self):
        """Nominal distance from cutoff to unseal, in seconds.

        MIN_COMMITMENT_AGE_BLOCKS closes submissions an hour before the run
        ends; the margin puts the unseal an hour after it. The gap between them
        is what a slow chain has to eat through before the two cross.
        """
        from capability_subnet.common.chain import run_opens_block

        close = run_opens_block(422, C.DEFAULT_RUN_BLOCKS)
        cutoff = close - C.MIN_COMMITMENT_AGE_BLOCKS
        gap = T.round_time(T.reveal_round_for_run(421)) - T.block_instant(cutoff)
        assert gap >= 2 * 3600

    def test_it_reads_the_anchor_rather_than_multiplying_the_run_id(self):
        """Runs before the epoch are a different length; the round must follow.

        The bug this stops is the one run_opens_block was written for - treating
        a run id as run_id * run_blocks, which is off by whatever the epoch is
        not a multiple of, for every run after it.
        """
        assert T.reveal_round_for_run(C.RUN_EPOCH_ID) != T.reveal_round_for_run(
            C.RUN_EPOCH_ID, run_blocks=C.LEGACY_RUN_BLOCKS
        )

    def test_the_anchor_instant_agrees_with_the_anchor_block(self):
        assert T.block_instant(C.RUN_EPOCH_BLOCK) == C.RUN_EPOCH_UNIX


class TestAWrongRoundIsNotASubmission:
    """Exact equality, and a refusal that says which side is stale."""

    def test_the_pinned_round_is_accepted(self):
        T.check_reveal_round(421, T.reveal_round_for_run(421))

    @pytest.mark.parametrize("off", [-1, 1, -28_800, 28_800])
    def test_any_other_round_is_refused(self, off):
        with pytest.raises(T.RevealRoundError):
            T.check_reveal_round(421, T.reveal_round_for_run(421) + off)

    def test_a_neighbouring_run_s_round_is_refused(self):
        """The likeliest real mistake: a miner who did not restart their CLI."""
        with pytest.raises(T.RevealRoundError, match="not run 421's pinned round"):
            T.check_reveal_round(421, T.reveal_round_for_run(420))

    def test_the_refusal_reports_the_drift_in_seconds(self):
        with pytest.raises(T.RevealRoundError, match=r"\+30s"):
            T.check_reveal_round(421, T.reveal_round_for_run(421) + 10)

    def test_there_is_no_tolerance_window(self):
        """Both sides use integer arithmetic on shared constants.

        A round "nearly" right is not a rounding artefact - it means one side is
        running different code, and admitting it means admitting an unseal at a
        time nobody chose.
        """
        pinned = T.reveal_round_for_run(421)
        accepted = [r for r in range(pinned - 3, pinned + 4) if _accepts(421, r)]
        assert accepted == [pinned]


def _accepts(run_id: int, reveal_round: int) -> bool:
    try:
        T.check_reveal_round(run_id, reveal_round)
    except T.RevealRoundError:
        return False
    return True


class TestThePayloadSurvivesTheReaders:
    """A sealed payload must come back whole, including through a stock SDK.

    The pallet stores exactly what was sealed. Every SDK-built reader strips a
    SCALE compact length prefix on the way out, so an unframed payload is
    silently truncated rather than rejected - it fails its digest check later
    and looks like a miner who submitted garbage. Confirmed on testnet 544: a
    zlib stream sealed as ``78 da …`` was handed back as ``da …``.
    """

    @pytest.mark.parametrize("size", [0, 1, 63, 64, 433, 16_383, 16_384, 70_000])
    def test_framing_round_trips_at_every_compact_width(self, size):
        payload = b"x" * size
        assert T.unframe_payload(T.frame_payload(payload)) == payload

    @pytest.mark.parametrize("size,width", [(63, 1), (64, 2), (16_383, 2), (16_384, 4)])
    def test_the_prefix_width_follows_the_scale_encoding(self, size, width):
        assert len(T.frame_payload(b"x" * size)) - size == width

    def test_a_stock_sdk_reader_recovers_a_framed_payload(self):
        """The regression this whole helper exists for.

        Reads through ``bittensor``'s own decoder rather than a copy of its
        logic, so that an SDK that changes its framing breaks this test instead
        of the subnet.
        """
        metagraph = pytest.importorskip("bittensor.metagraph")
        payload = bytes.fromhex("78da") + b"compressed recipe bytes" * 20

        _, unframed = metagraph._revealed_entry(("0x" + payload.hex(), 1))
        recovered = bytes.fromhex(unframed[2:]) if unframed.startswith("0x") else unframed.encode()
        assert recovered != payload, "unframed payloads used to survive; re-check the frame"

        _, framed = metagraph._revealed_entry(("0x" + T.frame_payload(payload).hex(), 1))
        recovered = bytes.fromhex(framed[2:]) if framed.startswith("0x") else framed.encode()
        assert recovered == payload

    def test_a_truncated_frame_is_refused_rather_than_guessed(self):
        framed = T.frame_payload(b"x" * 100)
        with pytest.raises(ValueError, match="declares 100 bytes"):
            T.unframe_payload(framed[:-1])

    def test_an_empty_payload_is_refused(self):
        with pytest.raises(ValueError, match="carries no compact length prefix"):
            T.unframe_payload(b"")
