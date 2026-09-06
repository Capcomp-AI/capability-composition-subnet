"""A commitment near a run's close destroys the one already held.

A hotkey holds exactly one commitment and the pallet keeps no history. Inside
the settling window a new commitment joins the *next* run - correctly, and the
chain takes it without complaint - while overwriting whatever was sealed for
the run now closing. That one never opens, so it is never measured and nothing
reports it: the miner sees two successful commits and one scored run.

Fourteen of run 423's fifty-five submissions went that way, eleven of them from
commits made in a sixteen-block span just past the cutoff.

So ``capcomp commit`` stops twice as far out as the protocol does. The extra
hour is a client-side refusal and nothing a validator computes depends on it -
``MIN_COMMITMENT_AGE_BLOCKS`` decides which run a commitment joins and is left
alone, because every validator has to agree on that one.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.chain import run_opens_block
from capability_subnet.miner.commit import CommitError, check_not_closing, run_for_commit


def _close(run: int = 424) -> int:
    return run_opens_block(run + 1, C.DEFAULT_RUN_BLOCKS)


class TestTheCommandStopsBeforeTheProtocolDoes:
    def test_a_commitment_well_inside_the_run_is_allowed(self):
        check_not_closing(_close() - C.COMMIT_CUTOFF_BLOCKS - 1)

    def test_one_block_past_the_cutoff_is_refused(self):
        with pytest.raises(CommitError, match="stops 300 blocks out"):
            check_not_closing(_close() - C.COMMIT_CUTOFF_BLOCKS)

    def test_the_cutoff_is_the_protocol_s_own(self):
        """They were two apart, and the command stopped an hour early to buy a
        margin against an extrinsic landing late. They are now the same, so
        every surface names one window - one hour before a close, two after the
        next opens. The margin that bought is covered instead by the engine
        keeping every sealed ciphertext, which recovers a commitment lost that
        way rather than trying to avoid losing it.
        """
        assert C.COMMIT_CUTOFF_BLOCKS == C.MIN_COMMITMENT_AGE_BLOCKS

    def test_the_protocol_rule_is_untouched(self):
        """Consensus reads MIN_COMMITMENT_AGE_BLOCKS, so it must not move.

        Changing it would have validators on different versions select
        different fields from the same chain.
        """
        assert C.MIN_COMMITMENT_AGE_BLOCKS == 300


class TestTheRefusalSaysWhatIsAtStake:
    def test_it_names_the_destruction_not_just_the_run(self):
        """The run it would join is the lesser half.

        A miner reading "this joins run N+1" reasonably concludes their earlier
        commitment still stands. It does not.
        """
        with pytest.raises(CommitError) as caught:
            check_not_closing(_close() - 10)
        message = str(caught.value)

        assert "DESTROYS" in message
        assert "never measured" in message
        assert "holds exactly one" in message

    def test_it_gives_the_wait_in_minutes_not_only_blocks(self):
        with pytest.raises(CommitError) as caught:
            check_not_closing(_close() - 300)
        assert "minutes" in str(caught.value)

    def test_past_the_protocol_cutoff_it_offers_the_next_run(self):
        with pytest.raises(CommitError) as caught:
            check_not_closing(_close(424) - 10)
        assert "--run 425" in str(caught.value)

    def test_before_it_the_override_is_this_run_not_the_next(self):
        """The advice has to differ, because the safe move differs.

        At exactly the cutoff a commitment still joins run 424, so telling a
        miner to wait for 425 would cost them the run they can still enter.
        With the two cutoffs equal this band is one block wide, which is
        precisely where an off-by-one would hide.
        """
        with pytest.raises(CommitError) as caught:
            check_not_closing(_close(424) - C.MIN_COMMITMENT_AGE_BLOCKS)
        message = str(caught.value)

        assert "--run 424 to commit to run 424 anyway" in message
        assert "--run 425" not in message

    def test_inside_the_protocol_window_it_says_the_run_changes(self):
        """Past 300 blocks the commitment genuinely joins the next run."""
        with pytest.raises(CommitError) as caught:
            check_not_closing(_close(424) - 100)
        assert "joins run 425, not run 424" in str(caught.value)

    def test_at_the_cutoff_it_says_the_run_still_holds(self):
        """Standing exactly the settling window counts as settled, so the run
        does not change here - the refusal is about how little room is left,
        and must not claim otherwise."""
        with pytest.raises(CommitError) as caught:
            check_not_closing(_close(424) - C.MIN_COMMITMENT_AGE_BLOCKS)
        message = str(caught.value)

        assert "would still join run 424" in message
        # It may name run 425 - it explains what crossing the cutoff would do -
        # but it must not assert the commitment has already moved there.
        assert "joins run 425, not run 424" not in message


class TestItAgreesWithTheRunItWouldJoin:
    @pytest.mark.parametrize("left", [1, 50, 299])
    def test_a_refused_block_inside_the_protocol_window_joins_the_next_run(self, left):
        block = _close(424) - left
        assert run_for_commit(block) == 425
        with pytest.raises(CommitError):
            check_not_closing(block)

    def test_the_boundary_block_is_refused_but_still_joins_this_run(self):
        """Refused by the command, admissible to the protocol. Both true.

        300 is the boundary and it belongs on this side: standing exactly the
        settling window counts as settled.
        """
        block = _close(424) - C.MIN_COMMITMENT_AGE_BLOCKS
        assert run_for_commit(block) == 424
        with pytest.raises(CommitError):
            check_not_closing(block)

    @pytest.mark.parametrize("left", [301, 450, 600, 1200])
    def test_further_out_than_the_cutoff_is_allowed(self, left):
        """The hour the command used to decline and no longer does. It joins
        this run and replaces only this run's commitment, which is ordinary
        resubmission."""
        block = _close(424) - left
        assert run_for_commit(block) == 424
        check_not_closing(block)
