"""A second commitment replaces the first. Sometimes that is the point.

A hotkey holds exactly one commitment and the pallet keeps no history. Writing
another replaces it, and a sealed commitment replaced before it opens never
opens, is never measured and is never reported.

Whether that is a loss depends entirely on which run the held one was for:

* the run being committed to — ordinary resubmission, and the newest is what
  gets measured. There is no submission limit and this must stay allowed.
* any other run — the miner meant to enter both, and the pallet cannot hold
  two. This is the one that costs a field.

Thirteen of run 424's sixty-nine entrants were lost this way in one window,
eight of them in the hour *after* the next run opened, where every check built
on distance-to-close said yes.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.chain import run_opens_block
from capability_subnet.common.timelock import reveal_round_for_run
from capability_subnet.miner.commit import (
    CommitError,
    check_not_closing,
    check_not_overwriting,
)


class TestReplacingIsSometimesThepoint:
    def test_a_commitment_for_the_same_run_is_ordinary_resubmission(self):
        """The subnet advertises no submission limit. Refusing here would
        contradict the contract and stop a miner improving a recipe."""
        check_not_overwriting(reveal_round_for_run(500), 500)

    def test_holding_nothing_is_allowed(self):
        check_not_overwriting(None, 500)
        check_not_overwriting(0, 500)

    def test_a_commitment_for_another_run_is_refused(self):
        with pytest.raises(CommitError) as caught:
            check_not_overwriting(reveal_round_for_run(499), 500)

        assert "never opens" in str(caught.value)
        assert "Nothing was committed" in str(caught.value)

    def test_the_refusal_offers_the_override(self):
        """--run is how a miner says they meant it. The guard must not be
        un-overridable, only un-missable."""
        with pytest.raises(CommitError) as caught:
            check_not_overwriting(reveal_round_for_run(499), 500)

        assert "--run 500" in str(caught.value)

    def test_a_future_run_is_refused_too(self):
        """Sealed ahead, committing behind. Direction does not matter: the
        pallet holds one, and one of the two is lost either way."""
        with pytest.raises(CommitError):
            check_not_overwriting(reveal_round_for_run(501), 500)


class TestTheHourAfterARunOpens:
    """check_not_closing measures distance to this run's close, so it went
    quiet the instant the boundary passed - through the whole hour the previous
    run's commitments were still sealed."""

    def test_just_after_a_run_opens_is_refused(self):
        block = run_opens_block(500, C.DEFAULT_RUN_BLOCKS) + 10

        with pytest.raises(CommitError) as caught:
            check_not_closing(block)

        assert "REPLACES" in str(caught.value)

    def test_the_protocol_reveal_alone_is_not_enough(self):
        """Clear of the reveal by ten blocks is still refused: the advisory
        holds an hour past it, so a miner is not racing the reveal."""
        block = run_opens_block(500, C.DEFAULT_RUN_BLOCKS) + C.REVEAL_MARGIN_BLOCKS + 10

        with pytest.raises(CommitError):
            check_not_closing(block)

    def test_it_clears_after_the_hold(self):
        block = run_opens_block(500, C.DEFAULT_RUN_BLOCKS) + C.COMMIT_HOLD_AFTER_OPEN_BLOCKS

        check_not_closing(block)

    def test_the_middle_of_a_run_is_untouched(self):
        block = run_opens_block(500, C.DEFAULT_RUN_BLOCKS) + 3000

        check_not_closing(block)

    def test_the_hour_before_a_close_still_refuses(self):
        """The half that already worked."""
        block = run_opens_block(501, C.DEFAULT_RUN_BLOCKS) - 100

        with pytest.raises(CommitError):
            check_not_closing(block)


class TestTheAdvisedSpan:
    """Three spans, and they are not the same number. Worth writing down.

    * the protocol's own danger: MIN_COMMITMENT_AGE_BLOCKS before a run closes,
      then REVEAL_MARGIN_BLOCKS while the closing run is still sealed - 2 hours.
    * the submissions API's advice: 1 hour before, 2 after - 3 hours.
    * this CLI: COMMIT_CUTOFF_BLOCKS is already 2 hours before a close, so with
      a 2 hour hold after an open it declines across 4.

    The CLI is the most conservative because it is the one holding the miner's
    keys at the moment of the extrinsic, and its refusal is overridable with
    --run. Nothing here is a gate: set_commitment is an extrinsic against a
    pallet this subnet does not own.
    """

    def test_the_protocol_danger_is_two_hours(self):
        span = C.MIN_COMMITMENT_AGE_BLOCKS + C.REVEAL_MARGIN_BLOCKS
        assert span * C.BLOCK_SECONDS / 3600 == 2.0

    def test_this_cli_declines_across_four(self):
        span = C.COMMIT_CUTOFF_BLOCKS + C.COMMIT_HOLD_AFTER_OPEN_BLOCKS
        assert span * C.BLOCK_SECONDS / 3600 == 4.0

    def test_every_advisory_covers_the_real_danger(self):
        """Advice shorter than the danger would be worse than none."""
        assert C.COMMIT_CUTOFF_BLOCKS >= C.MIN_COMMITMENT_AGE_BLOCKS
        assert C.COMMIT_HOLD_AFTER_OPEN_BLOCKS >= C.REVEAL_MARGIN_BLOCKS
