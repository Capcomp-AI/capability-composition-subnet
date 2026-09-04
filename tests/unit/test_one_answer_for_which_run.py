"""The same commitment must give the same run, before and after it opens.

Nothing asserted this, and it was false. `block` is the commit block while a
commitment is sealed and the block the *reveal* landed at once the pallet opens
it - one shared value for a whole field, a run later - so every caller that
derived a run from it changed its answer at the reveal. Run 423's entire field
was filed under 424 the moment it opened.

Two rules also disagreed on the same input before any of that. `run_id_for_block`
gives the run the clock was in; the settling rule gives the run actually joined.
They differ for everything committed in a run's closing window, and the console
used one for `submitted_run` and the other for `measured_in_run` in the same
function - so those columns came out two runs apart for exactly those miners.

The invariant is one line: whatever a caller knows about a commitment, it gets
the same run. These are the cases that broke it.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.chain import (
    RunUnknown,
    run_for_commitment,
    run_id_for_block,
    run_opens_block,
)

RUN_BLOCKS = C.DEFAULT_RUN_BLOCKS


def _settled(run: int) -> int:
    """A commit block comfortably inside ``run``, past nothing."""
    return run_opens_block(run, RUN_BLOCKS) + 100


def _in_settling_window(run: int) -> int:
    """A commit block in ``run``'s closing window, so it joins ``run + 1``."""
    return run_opens_block(run + 1, RUN_BLOCKS) - 10


def _reveal_block(run: int) -> int:
    """Where ``run``'s commitments open: its close plus the margin."""
    return run_opens_block(run + 1, RUN_BLOCKS) + C.REVEAL_MARGIN_BLOCKS


class TestTheAnswerSurvivesTheReveal:
    """The failure that cost run 423 its whole field."""

    @pytest.mark.parametrize("run", [412, 423, 424, 500])
    def test_a_settled_commitment_reads_the_same_either_side(self, run):
        sealed = run_for_commitment(commit_block=_settled(run), run_blocks=RUN_BLOCKS)
        opened = run_for_commitment(revealed_at_block=_reveal_block(run), run_blocks=RUN_BLOCKS)

        assert sealed == opened == run

    @pytest.mark.parametrize("run", [423, 424])
    def test_a_late_commitment_reads_the_same_either_side(self, run):
        """It joins run + 1, and still does after it opens."""
        joined = run + 1
        sealed = run_for_commitment(commit_block=_in_settling_window(run), run_blocks=RUN_BLOCKS)
        opened = run_for_commitment(revealed_at_block=_reveal_block(joined), run_blocks=RUN_BLOCKS)

        assert sealed == opened == joined

    def test_the_reveal_block_alone_never_answers_with_its_own_run(self):
        """The specific wrong answer: 8995364 is in run 424 and means 423."""
        reveal = _reveal_block(423)
        assert run_id_for_block(reveal, RUN_BLOCKS) == 424
        assert run_for_commitment(revealed_at_block=reveal, run_blocks=RUN_BLOCKS) == 423


class TestEveryCallerAgrees:
    def test_a_stated_run_beats_a_block_that_would_mislead(self):
        """Admission records the run; the block it carries is the reveal's."""
        assert (
            run_for_commitment(
                stated_run=423, revealed_at_block=_reveal_block(423), run_blocks=RUN_BLOCKS
            )
            == 423
        )

    def test_a_stated_run_is_taken_even_against_a_commit_block(self):
        assert run_for_commitment(stated_run=423, commit_block=_settled(419)) == 423

    def test_the_clock_rule_and_this_one_differ_only_in_the_window(self):
        """Which is exactly where the console's two columns disagreed."""
        inside = _in_settling_window(423)
        assert run_id_for_block(inside, RUN_BLOCKS) == 423
        assert run_for_commitment(commit_block=inside, run_blocks=RUN_BLOCKS) == 424

        outside = _settled(423)
        assert run_id_for_block(outside, RUN_BLOCKS) == run_for_commitment(
            commit_block=outside, run_blocks=RUN_BLOCKS
        )


class TestItRefusesRatherThanGuesses:
    def test_nothing_known_raises(self):
        with pytest.raises(RunUnknown):
            run_for_commitment()

    def test_zero_is_not_a_block(self):
        with pytest.raises(RunUnknown):
            run_for_commitment(commit_block=0, revealed_at_block=0)

    def test_a_run_of_zero_is_still_a_stated_run(self):
        """Falsy but present. `if stated_run` would have lost run 0."""
        assert run_for_commitment(stated_run=0) == 0
