"""Locating a block inside its evaluation window.

Derived from the block height and the window length alone. In the default
arrangement each validator evaluates for itself, so there is no central engine
to ask where a window is — anyone with a chain connection works it out.
"""

from __future__ import annotations

import pytest

from capability_subnet.common.chain import window_id_for_block, window_position


class TestWindowPosition:
    def test_the_first_block_of_a_window_has_nothing_elapsed(self):
        p = window_position(7200, 7200)
        assert (p.window_id, p.opened_block, p.closes_block) == (1, 7200, 14400)
        assert p.blocks_elapsed == 0
        assert p.progress == 0.0
        assert p.blocks_remaining == 7200

    def test_the_last_block_is_still_inside_the_window(self):
        p = window_position(14399, 7200)
        assert p.window_id == 1
        assert p.blocks_remaining == 1
        assert p.progress < 1.0, "progress reaching 1.0 would name the next window"

    def test_the_midpoint_reads_as_half(self):
        assert window_position(3600, 7200).progress == pytest.approx(0.5)

    def test_it_agrees_with_the_window_id_it_is_derived_from(self):
        for block in (0, 1, 7199, 7200, 123_456, 8_847_760):
            for span in (600, 7200, 21600):
                assert window_position(block, span).window_id == window_id_for_block(block, span)

    def test_elapsed_and_remaining_account_for_the_whole_window(self):
        for block in (0, 5, 7199, 40_000, 8_847_760):
            p = window_position(block, 7200)
            assert p.blocks_elapsed + p.blocks_remaining == 7200

    def test_time_left_follows_the_chain_s_block_time(self):
        p = window_position(7200 + 600, 7200)
        assert p.seconds_remaining(12.0) == pytest.approx((7200 - 600) * 12.0)

    @pytest.mark.parametrize("block,span", [(-1, 7200), (100, 0), (100, -7200)])
    def test_nonsense_is_refused_rather_than_answered(self, block, span):
        """A position computed from a negative block or a zero-length window
        would look meaningful and mean nothing."""
        with pytest.raises(ValueError):
            window_position(block, span)


class TestACommitmentIsMeasuredOnce:
    """Which window owns a commitment, derived from the chain and nothing else."""

    W = 21_600

    def test_it_is_measured_in_the_window_after_the_one_it_was_made_in(self):
        from capability_subnet.common.chain import measured_in_window

        # committed midway through window 410
        block = 410 * self.W + 5_000
        assert measured_in_window(block, 411, self.W)

    def test_it_is_not_measured_in_the_window_it_was_made_in(self):
        """The beacon already existed, so the instances were visible."""
        from capability_subnet.common.chain import measured_in_window

        assert not measured_in_window(410 * self.W + 5_000, 410, self.W)

    def test_it_is_not_measured_again_afterwards(self):
        """Earning once per commitment is what makes resubmission the way to earn."""
        from capability_subnet.common.chain import measured_in_window

        block = 410 * self.W + 5_000
        assert not any(measured_in_window(block, w, self.W) for w in range(412, 420))

    def test_the_first_block_of_a_window_belongs_to_that_window(self):
        from capability_subnet.common.chain import measured_in_window

        assert measured_in_window(410 * self.W, 411, self.W)
        assert not measured_in_window(410 * self.W, 412, self.W)

    def test_the_last_block_of_a_window_belongs_to_that_window(self):
        from capability_subnet.common.chain import measured_in_window

        assert measured_in_window(411 * self.W - 1, 411, self.W)

    def test_every_validator_agrees_without_remembering_anything(self):
        """The rule reads one chain fact, so a fresh validator selects the same set."""
        from capability_subnet.common.chain import measured_in_window

        blocks = [410 * self.W + n for n in (0, 1, 17, 5_000, self.W - 1)]
        assert all(measured_in_window(b, 411, self.W) for b in blocks)

    def test_a_nonsense_window_length_is_refused(self):
        import pytest

        from capability_subnet.common.chain import measured_in_window

        with pytest.raises(ValueError):
            measured_in_window(1000, 1, 0)
