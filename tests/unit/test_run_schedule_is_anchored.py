"""The deployed run schedule: one run a day, opening at noon Eastern.

Not the arithmetic — that is `test_run_position.py`, which states its own
schedule so it can say "run 411 starts at 411 * W". This is the schedule that
is actually shipped, checked against real block heights and real dates, because
the constants are the only thing standing between "runs open at noon" and
"runs open at 04:26 on a rotating three-day cycle", which is what unanchored
arithmetic produced and nobody chose.

The wall-clock claims are checked at finney's measured block time, 12.0029 s
over the 201,600 blocks to 22 August 2026. A boundary is allowed to drift
within the hour it is supposed to land in; anything worse means the epoch needs
re-anchoring, which is a one-line change and the point of pinning it here.
"""

from __future__ import annotations

import datetime
import zoneinfo

from capability_subnet.common import constants as C
from capability_subnet.common.chain import (
    measuring_run_for,
    run_id_for_block,
    run_opens_block,
    run_position,
    weighting_run_for,
)

ET = zoneinfo.ZoneInfo("America/New_York")
W = C.DEFAULT_RUN_BLOCKS

#: A block whose timestamp was read from the chain, and finney's measured rate.
REFERENCE_BLOCK = 8_900_085
REFERENCE_TIME = datetime.datetime(2026, 8, 22, 11, 23, 12, tzinfo=datetime.timezone.utc)
SECONDS_PER_BLOCK = 12.0029


def when(block: int) -> datetime.datetime:
    """Approximately when the chain reaches a block, in Eastern time."""
    offset = (block - REFERENCE_BLOCK) * SECONDS_PER_BLOCK
    return (REFERENCE_TIME + datetime.timedelta(seconds=offset)).astimezone(ET)


class TestARunIsADay:
    def test_the_length_is_twenty_four_hours(self):
        assert W * 12 == 24 * 60 * 60

    def test_consecutive_runs_are_one_day_apart(self):
        for run_id in range(C.RUN_EPOCH_ID, C.RUN_EPOCH_ID + 14):
            span = run_opens_block(run_id + 1, W) - run_opens_block(run_id, W)
            assert span == W


class TestBoundariesLandAtNoonEastern:
    """The whole reason the epoch exists."""

    #: How far a boundary may sit from noon before the epoch needs re-anchoring.
    #: Blocks are not a clock: finney runs about 12.0029 s rather than 12, so a
    #: day's boundary slides ~21 s and a month's ~10 min. Asserting noon to the
    #: minute would fail on drift that costs nobody anything; asserting nothing
    #: would let it slide into the evening unnoticed.
    TOLERANCE = datetime.timedelta(minutes=30)

    def drift(self, run_id: int) -> datetime.timedelta:
        opens = when(run_opens_block(run_id, W))
        noon = opens.replace(hour=12, minute=0, second=0, microsecond=0)
        return abs(opens - noon)

    def test_the_epoch_opens_run_412_at_noon_on_23_august(self):
        opens = when(run_opens_block(C.RUN_EPOCH_ID, W))

        assert (opens.year, opens.month, opens.day) == (2026, 8, 23)
        assert self.drift(C.RUN_EPOCH_ID) <= self.TOLERANCE, (
            f"run 412 opens at {opens:%I:%M:%S %p %Z}, not noon"
        )

    def test_the_next_fortnight_all_open_near_noon(self):
        """Block drift is real but slow: about 21 seconds a day.

        Checked over a fortnight rather than one boundary, because a schedule
        that is right on the first day and wrong on the tenth is the failure
        this is here to catch.
        """
        for offset in range(14):
            run_id = C.RUN_EPOCH_ID + offset
            opens = when(run_opens_block(run_id, W))
            assert self.drift(run_id) <= self.TOLERANCE, (
                f"run {run_id} opens {opens:%a %b %d %I:%M %p %Z}, too far from noon"
            )

    def test_each_run_opens_on_the_next_calendar_day(self):
        days = [when(run_opens_block(C.RUN_EPOCH_ID + n, W)).date() for n in range(7)]
        expected = [datetime.date(2026, 8, 23) + datetime.timedelta(days=n) for n in range(7)]

        assert days == expected


class TestHistoryIsFrozenRatherThanRenumbered:
    """Runs that have closed keep the ids they were published under.

    Re-deriving them at today's length would move every stored run, report and
    console row to a number it was never filed under — run 411 would become
    run 1233. Run 411 instead runs long: it opened under the old length and
    closes at the epoch.
    """

    def test_the_block_before_the_epoch_is_still_run_411(self):
        assert run_id_for_block(C.RUN_EPOCH_BLOCK - 1, W) == C.RUN_EPOCH_ID - 1

    def test_the_epoch_block_itself_is_run_412(self):
        assert run_id_for_block(C.RUN_EPOCH_BLOCK, W) == C.RUN_EPOCH_ID

    def test_old_runs_keep_the_boundaries_they_ran_at(self):
        assert run_opens_block(410, W) == 410 * C.LEGACY_RUN_BLOCKS
        assert run_opens_block(411, W) == 411 * C.LEGACY_RUN_BLOCKS

    def test_run_411_runs_long_and_closes_at_the_epoch(self):
        position = run_position(C.RUN_EPOCH_BLOCK - 1, W)

        assert position.run_id == 411
        assert position.closes_block == C.RUN_EPOCH_BLOCK
        assert position.opened_block == 411 * C.LEGACY_RUN_BLOCKS
        assert 0.0 <= position.progress < 1.0

    def test_no_run_id_is_claimed_by_two_blocks_ranges(self):
        """The join must not leave a gap or an overlap.

        Every block from well before the epoch to well after it maps to a run
        whose own opening block is the one this block belongs to.
        """
        for block in range(C.RUN_EPOCH_BLOCK - 40_000, C.RUN_EPOCH_BLOCK + 40_000, 617):
            run_id = run_id_for_block(block, W)
            assert run_opens_block(run_id, W) <= block < run_opens_block(run_id + 1, W)


class TestTheThreeRunPipeline:
    """commit in N, measured in N+1, paid in N+2.

    Stated against the deployed schedule because it is the promise a miner
    plans around, and every part of it is derived from the commitment block
    alone — no validator keeps a record that could disagree.
    """

    def test_a_settled_commitment_walks_the_pipeline_one_run_at_a_time(self):
        committed = C.RUN_EPOCH_BLOCK + 50
        run = run_id_for_block(committed, W)

        assert run == 412
        assert measuring_run_for(committed, W) == run + 1
        assert weighting_run_for(committed, W) == run + 2

    def test_it_holds_for_every_hour_of_a_run(self):
        opens = run_opens_block(500, W)
        settles_by = run_opens_block(501, W) - C.MIN_COMMITMENT_AGE_BLOCKS

        for block in range(opens, settles_by + 1, 300):
            assert measuring_run_for(block, W) == 501, f"block {block}"
            assert weighting_run_for(block, W) == 502, f"block {block}"

    def test_committing_in_the_last_hour_moves_the_whole_pipeline_a_run(self):
        """The settling rule, and it carries through to the payment run."""
        late = run_opens_block(501, W) - C.MIN_COMMITMENT_AGE_BLOCKS + 1

        assert run_id_for_block(late, W) == 500
        assert measuring_run_for(late, W) == 502
        assert weighting_run_for(late, W) == 503

    def test_the_settling_hour_is_an_hour(self):
        position = run_position(run_opens_block(500, W), W)
        blocks = position.closes_block - position.settles_by_block

        assert blocks == C.MIN_COMMITMENT_AGE_BLOCKS
        assert blocks * 12 == 60 * 60
