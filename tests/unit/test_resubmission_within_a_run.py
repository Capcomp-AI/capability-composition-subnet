"""A miner may replace its recipe freely; the last one standing is measured.

This is a promise made to miners, so it is pinned rather than left to follow
from arithmetic. Three properties, and the last two are the ones that cost
someone a run if misunderstood:

* replacing a recipe during a run is allowed and costs nothing — the chain keeps
  one commitment per hotkey, so earlier attempts are simply overwritten;
* whatever stands when the run ends is measured once, in the run after, provided
  it has been standing for MIN_COMMITMENT_AGE_BLOCKS by then;
* replacing it during *that* run destroys the queued measurement, because the
  commitment block moves forward with it.
"""

from __future__ import annotations

from capability_subnet.common import constants as C
from capability_subnet.common.chain import measured_in_run, run_id_for_block

RUN_ID = 411
RUN_BLOCKS = C.DEFAULT_RUN_BLOCKS
START = RUN_ID * RUN_BLOCKS
#: Late enough to be the miner's final word, early enough to have settled.
SETTLED = START + RUN_BLOCKS - C.MIN_COMMITMENT_AGE_BLOCKS - 10


class TestOnlyTheLastSubmissionIsMeasured:
    def test_every_attempt_in_a_run_maps_to_the_same_measurement(self):
        """Submitting three times buys one measurement, not three.

        The cap is structural. There is nothing counting attempts — the chain
        holds one commitment per hotkey — so however many times a miner commits
        inside a run, exactly one recipe stands at the end of it and exactly one
        measurement follows.
        """
        attempts = [START + 10, START + RUN_BLOCKS // 2, SETTLED]

        measured_in = {run_id_for_block(block, RUN_BLOCKS) + 1 for block in attempts}
        assert measured_in == {RUN_ID + 1}, "attempts in one run must share one measurement"

    def test_the_final_attempt_is_measured_in_the_next_run(self):
        assert not measured_in_run(SETTLED, RUN_ID, RUN_BLOCKS), "not the run it was made in"
        assert measured_in_run(SETTLED, RUN_ID + 1, RUN_BLOCKS), "measured in the run after"
        assert not measured_in_run(SETTLED, RUN_ID + 2, RUN_BLOCKS), "and never again"


class TestSubmittingAtTheCloseCostsARun:
    """The enforceable form of a rate limit.

    Attempts cannot be counted — nothing records that an earlier commitment
    existed — but the age of the one that stands can be, and every replacement
    restarts it.
    """

    def test_a_commitment_made_at_the_close_waits_a_further_run(self):
        late = START + RUN_BLOCKS - 10

        assert not measured_in_run(late, RUN_ID + 1, RUN_BLOCKS), "too fresh for the next run"
        assert measured_in_run(late, RUN_ID + 2, RUN_BLOCKS), "held over, not discarded"

    def test_it_is_the_age_rule_doing_this_and_not_the_run_boundary(self):
        """Without the age requirement the same block lands in the next run."""
        late = START + RUN_BLOCKS - 10

        assert measured_in_run(late, RUN_ID + 1, RUN_BLOCKS, min_age_blocks=0)

    def test_settling_for_the_full_hour_is_enough(self):
        exactly = START + RUN_BLOCKS - C.MIN_COMMITMENT_AGE_BLOCKS

        assert measured_in_run(exactly, RUN_ID + 1, RUN_BLOCKS)


class TestReplacingAQueuedSubmissionCostsARun:
    def test_committing_during_the_judging_run_forfeits_it(self):
        """The trap worth warning miners about.

        A recipe committed in run 411 is measured in run 412. Committing again
        *during* run 412 moves the commitment block into 412, so it is measured
        in 413 instead and 412 measures nothing from this hotkey. The miner did
        not fail a gate or lose a comparison; they withdrew the submission that
        was about to be judged.
        """
        assert measured_in_run(SETTLED, RUN_ID + 1, RUN_BLOCKS)

        replaced = (RUN_ID + 1) * RUN_BLOCKS + 100
        assert not measured_in_run(replaced, RUN_ID + 1, RUN_BLOCKS), (
            "the queued measurement survived a replacement, which it must not"
        )
        assert measured_in_run(replaced, RUN_ID + 2, RUN_BLOCKS)

    def test_leaving_a_queued_submission_alone_earns_in_the_next_run(self):
        """The other half of the same rule, so the guidance is unambiguous."""
        assert measured_in_run(SETTLED, RUN_ID + 1, RUN_BLOCKS)
