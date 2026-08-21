"""Locating a block inside its evaluation run.

Derived from the block height and the run length alone. In the default
arrangement each validator evaluates for itself, so there is no central engine
to ask where a run is — anyone with a chain connection works it out.
"""

from __future__ import annotations

import pytest

from capability_subnet.common.chain import run_id_for_block, run_position


class TestRunPosition:
    def test_the_first_block_of_a_run_has_nothing_elapsed(self):
        p = run_position(7200, 7200)
        assert (p.run_id, p.opened_block, p.closes_block) == (1, 7200, 14400)
        assert p.blocks_elapsed == 0
        assert p.progress == 0.0
        assert p.blocks_remaining == 7200

    def test_the_last_block_is_still_inside_the_run(self):
        p = run_position(14399, 7200)
        assert p.run_id == 1
        assert p.blocks_remaining == 1
        assert p.progress < 1.0, "progress reaching 1.0 would name the next run"

    def test_the_midpoint_reads_as_half(self):
        assert run_position(3600, 7200).progress == pytest.approx(0.5)

    def test_it_agrees_with_the_run_id_it_is_derived_from(self):
        for block in (0, 1, 7199, 7200, 123_456, 8_847_760):
            for span in (600, 7200, 21600):
                assert run_position(block, span).run_id == run_id_for_block(block, span)

    def test_elapsed_and_remaining_account_for_the_whole_run(self):
        for block in (0, 5, 7199, 40_000, 8_847_760):
            p = run_position(block, 7200)
            assert p.blocks_elapsed + p.blocks_remaining == 7200

    def test_time_left_follows_the_chain_s_block_time(self):
        p = run_position(7200 + 600, 7200)
        assert p.seconds_remaining(12.0) == pytest.approx((7200 - 600) * 12.0)

    @pytest.mark.parametrize("block,span", [(-1, 7200), (100, 0), (100, -7200)])
    def test_nonsense_is_refused_rather_than_answered(self, block, span):
        """A position computed from a negative block or a zero-length run
        would look meaningful and mean nothing."""
        with pytest.raises(ValueError):
            run_position(block, span)


class TestACommitmentIsMeasuredOnce:
    """Which run owns a commitment, derived from the chain and nothing else."""

    W = 21_600

    def test_it_is_measured_in_the_run_after_the_one_it_was_made_in(self):
        from capability_subnet.common.chain import measured_in_run

        # committed midway through run 410
        block = 410 * self.W + 5_000
        assert measured_in_run(block, 411, self.W)

    def test_it_is_not_measured_in_the_run_it_was_made_in(self):
        """The beacon already existed, so the instances were visible."""
        from capability_subnet.common.chain import measured_in_run

        assert not measured_in_run(410 * self.W + 5_000, 410, self.W)

    def test_it_is_not_measured_again_afterwards(self):
        """Earning once per commitment is what makes resubmission the way to earn."""
        from capability_subnet.common.chain import measured_in_run

        block = 410 * self.W + 5_000
        assert not any(measured_in_run(block, w, self.W) for w in range(412, 420))

    def test_the_first_block_of_a_run_belongs_to_that_run(self):
        from capability_subnet.common.chain import measured_in_run

        assert measured_in_run(410 * self.W, 411, self.W)
        assert not measured_in_run(410 * self.W, 412, self.W)

    def test_a_commitment_made_at_the_close_is_held_to_the_run_after(self):
        """The last block of a run is too fresh to be measured by the next one.

        It is held over rather than dropped: a commitment made in the closing
        minutes has not been standing for MIN_COMMITMENT_AGE_BLOCKS when the
        next run opens, so it is measured by the one after that instead.
        """
        from capability_subnet.common import constants as C
        from capability_subnet.common.chain import measured_in_run

        closing = 411 * self.W - 1
        assert not measured_in_run(closing, 411, self.W), "too fresh for run 411"
        assert measured_in_run(closing, 412, self.W), "measured by the run after"

        # With no age requirement it would have been run 411, which is the
        # behaviour this rule deliberately changes.
        assert measured_in_run(closing, 411, self.W, min_age_blocks=0)

    def test_every_validator_agrees_without_remembering_anything(self):
        """The rule reads one chain fact, so a fresh validator selects the same set."""
        from capability_subnet.common import constants as C
        from capability_subnet.common.chain import measured_in_run

        # Every block of run 410 with room to settle before 411 opens.
        settled = [410 * self.W + n for n in (0, 1, 17, 5_000)]
        assert all(measured_in_run(b, 411, self.W) for b in settled)

        # And one without, which run 411 does not measure.
        fresh = 411 * self.W - C.MIN_COMMITMENT_AGE_BLOCKS + 1
        assert not measured_in_run(fresh, 411, self.W)

    def test_a_nonsense_run_length_is_refused(self):
        import pytest

        from capability_subnet.common.chain import measured_in_run

        with pytest.raises(ValueError):
            measured_in_run(1000, 1, 0)


class TestTheSettlingDeadlineIsVisible:
    """A miner should learn the deadline before missing it, not after.

    The chain accepts a commitment at any point in a run, so nothing stops a
    late one being made — it is simply measured a run later than intended. The
    only warning available is the arithmetic, and it is only useful if the
    tooling does it.
    """

    W = 21600

    def test_the_deadline_is_an_hour_before_the_run_closes(self):
        from capability_subnet.common import constants as C
        from capability_subnet.common.chain import run_position

        p = run_position(411 * self.W + 10, self.W)
        assert p.closes_block - p.settles_by_block == C.MIN_COMMITMENT_AGE_BLOCKS

    def test_early_in_the_run_there_is_time_to_change_your_mind(self):
        from capability_subnet.common.chain import run_position

        p = run_position(411 * self.W + 10, self.W)
        assert not p.in_settling_window
        assert p.blocks_until_settling_window > 0

    def test_inside_the_window_it_says_so(self):
        from capability_subnet.common import constants as C
        from capability_subnet.common.chain import run_position

        late = 412 * self.W - C.MIN_COMMITMENT_AGE_BLOCKS + 1
        p = run_position(late, self.W)
        assert p.in_settling_window
        assert p.blocks_until_settling_window == 0

    def test_the_boundary_block_is_still_in_time(self):
        """Standing for exactly the required age counts, here as elsewhere."""
        from capability_subnet.common import constants as C
        from capability_subnet.common.chain import measured_in_run, run_position

        exactly = 412 * self.W - C.MIN_COMMITMENT_AGE_BLOCKS
        p = run_position(exactly, self.W)

        assert not p.in_settling_window
        # And the position agrees with the rule it is describing.
        assert measured_in_run(exactly, p.run_id + 1, self.W)


class TestOneDefinitionOfWhichRunMeasuresACommitment:
    """`measuring_run_for` is the only place that decides this.

    The console files a mirrored row under the run that will score it, and it
    used to re-derive that as commit+1. That is the first half of the rule; a
    commitment made inside the closing window is measured a run later, so the
    copy disagreed with the protocol exactly at the boundary — where a wrong
    answer looks like an ordinary row.
    """

    W = 21600

    def test_it_agrees_with_the_predicate_at_every_point_in_a_run(self):
        from capability_subnet.common.chain import measured_in_run, measuring_run_for

        start = 411 * self.W
        for offset in (0, 1, 5_000, self.W // 2, self.W - 301, self.W - 300, self.W - 1):
            block = start + offset
            run = measuring_run_for(block, self.W)
            assert measured_in_run(block, run, self.W), f"disagreed at +{offset}"
            # And it is the *only* run that measures it.
            others = [
                r for r in range(410, 416)
                if r != run and measured_in_run(block, r, self.W)
            ]
            assert not others, f"+{offset} is measured by {others} as well as {run}"

    def test_a_settled_commitment_is_measured_by_the_next_run(self):
        from capability_subnet.common import constants as C
        from capability_subnet.common.chain import measuring_run_for

        settled = 411 * self.W + self.W - C.MIN_COMMITMENT_AGE_BLOCKS
        assert measuring_run_for(settled, self.W) == 412

    def test_a_commitment_at_the_close_is_measured_a_run_later(self):
        from capability_subnet.common.chain import measuring_run_for

        assert measuring_run_for(412 * self.W - 1, self.W) == 413
