"""A miner may replace its recipe freely; the last one standing is measured.

This is a promise made to miners, so it is pinned rather than left to follow
from arithmetic. Three properties, and the third is the one that costs someone
a run if it is misunderstood:

* replacing a recipe during a run is allowed and costs nothing — the chain keeps
  one commitment per hotkey, so earlier attempts are simply overwritten;
* whatever stands when the run ends is measured once, in the run after;
* replacing it during *that* run destroys the queued measurement, because the
  commitment block moves forward with it.
"""

from __future__ import annotations

from capability_subnet.common import constants as C
from capability_subnet.common.chain import measured_in_window, window_id_for_block

RUN = 411
WINDOW = C.DEFAULT_WINDOW_BLOCKS
START = RUN * WINDOW


class TestOnlyTheLastSubmissionIsMeasured:
    def test_every_attempt_in_a_run_maps_to_the_same_measurement(self):
        """Submitting three times buys one measurement, not three.

        The cap is structural. There is nothing counting attempts — the chain
        holds one commitment per hotkey — so however many times a miner
        commits inside a run, exactly one recipe stands at the end of it and
        exactly one measurement follows.
        """
        attempts = [START + 10, START + WINDOW // 2, START + WINDOW - 50]

        measured_in = {
            window_id_for_block(block, WINDOW) + 1 for block in attempts
        }
        assert measured_in == {RUN + 1}, "attempts in one run must share one measurement"

    def test_the_final_attempt_is_measured_in_the_next_run(self):
        final = START + WINDOW - 50

        assert not measured_in_window(final, RUN, WINDOW), "not in the run it was made in"
        assert measured_in_window(final, RUN + 1, WINDOW), "measured in the run after"
        assert not measured_in_window(final, RUN + 2, WINDOW), "and never again"


class TestReplacingAQueuedSubmissionCostsARun:
    def test_committing_during_the_judging_run_forfeits_it(self):
        """The trap worth warning miners about.

        A recipe committed in run 411 is measured in run 412. Committing again
        *during* run 412 moves the commitment block into 412, so it is measured
        in 413 instead and 412 measures nothing from this hotkey. The miner did
        not fail a gate or lose a comparison; they withdrew the submission that
        was about to be judged.
        """
        queued = START + 100
        assert measured_in_window(queued, RUN + 1, WINDOW)

        replaced = (RUN + 1) * WINDOW + 100
        assert not measured_in_window(replaced, RUN + 1, WINDOW), (
            "the queued measurement survived a replacement, which it must not"
        )
        assert measured_in_window(replaced, RUN + 2, WINDOW)

    def test_leaving_a_queued_submission_alone_earns_in_the_next_run(self):
        """The other half of the same rule, so the guidance is unambiguous."""
        queued = START + 100
        assert measured_in_window(queued, RUN + 1, WINDOW)
