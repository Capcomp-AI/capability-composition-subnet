"""How a run's emission is split.

Two rules decide everything, and the second is the one with teeth:

* ``BURN_SHARE`` of the run burns; the rest is the miner share.
* The run pays nothing unless its *leader* takes the throne, which means
  exceeding the reigning champion's grade by ``CHAMPION_DETHRONE_MARGIN``. Once
  the leader clears it the field behind them is paid by rank; if the leader
  cannot, nobody places and the whole miner share burns.

Everything here is arithmetic over a list of grades. No chain, no GPU.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.scoring.weight_vector import (
    champion_ladder,
    dethrone_threshold,
    rank_shares,
)

#: Eleven candidates, descending, spaced widely enough that the dethrone
#: margin is never what separates two of them.
FIELD = [
    (7, "5A", 0.50),
    (9, "5B", 0.48),
    (3, "5C", 0.46),
    (1, "5D", 0.44),
    (5, "5E", 0.42),
    (2, "5F", 0.40),
    (4, "5G", 0.38),
    (6, "5H", 0.36),
    (8, "5I", 0.34),
    (10, "5J", 0.32),
    (11, "5K", 0.30),
]


def paid(vector) -> dict[int, float]:
    return {e.uid: e.weight for e in vector.entries if e.role != "burn"}


def burned(vector) -> float:
    return sum(e.weight for e in vector.entries if e.role == "burn")


class TestTheSplit:
    def test_the_burn_takes_four_fifths(self):
        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)

        assert burned(vector) == pytest.approx(C.BURN_SHARE, abs=1e-9)

    def test_the_ladder_matches_the_published_shares(self):
        """Rank shares are of the *miner pool*, not of the run."""
        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)
        pool = 1.0 - C.BURN_SHARE
        weights = paid(vector)

        for index, share in enumerate(C.RANK_SHARES):
            uid = FIELD[index][0]
            assert weights[uid] == pytest.approx(share * pool, abs=1e-9), f"rank {index + 1}"

    def test_the_tail_is_split_by_grade_not_evenly(self):
        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)
        weights = paid(vector)
        tail = [FIELD[i] for i in range(len(C.RANK_SHARES), C.PAID_RANKS)]

        assert len(tail) == 5
        shares = [weights[uid] for uid, _, _ in tail]
        assert shares == sorted(shares, reverse=True), "an even split would order them arbitrarily"
        grades = [grade for _, _, grade in tail]
        for share, grade in zip(shares, grades, strict=True):
            expected = C.TAIL_SHARE * (grade / sum(grades)) * (1.0 - C.BURN_SHARE)
            assert share == pytest.approx(expected, abs=1e-9)

    def test_nothing_below_the_last_paid_rank_earns(self):
        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)

        assert len(paid(vector)) == C.PAID_RANKS
        assert FIELD[C.PAID_RANKS][0] not in paid(vector)

    def test_the_whole_vector_sums_to_one(self):
        for count in range(0, len(FIELD) + 1):
            vector = champion_ladder(FIELD[:count], run_id=412, block=1, champion_grade=None)
            total = sum(e.weight for e in vector.entries)
            assert total == pytest.approx(1.0, abs=1e-9), f"{count} candidates"


class TestTheRunPaysOnlyIfItsLeaderTakesTheThrone:
    def test_a_field_that_cannot_clear_the_bar_pays_nothing(self):
        champion = FIELD[0][2]
        vector = champion_ladder(FIELD, run_id=413, block=1, champion_grade=champion)

        assert paid(vector) == {}
        assert burned(vector) == pytest.approx(1.0, abs=1e-9)
        assert vector.champion_hotkey is None

    def test_matching_the_margin_exactly_is_not_enough(self):
        """A tie is not an improvement, and neither is the margin itself."""
        champion = 0.40
        exactly = [(7, "5A", champion + C.CHAMPION_DETHRONE_MARGIN)]

        assert paid(champion_ladder(exactly, run_id=413, block=1, champion_grade=champion)) == {}

    def test_clearing_the_margin_by_the_smallest_amount_is_enough(self):
        champion = 0.40
        over = [(7, "5A", champion + C.CHAMPION_DETHRONE_MARGIN + 1e-9)]

        vector = champion_ladder(over, run_id=413, block=1, champion_grade=champion)
        assert vector.champion_hotkey == "5A"

    def test_a_winner_alone_in_the_field_takes_the_first_share_and_the_rest_burns(self):
        vector = champion_ladder([(7, "5A", 0.60)], run_id=413, block=1, champion_grade=0.50)
        pool = 1.0 - C.BURN_SHARE

        assert paid(vector) == {7: pytest.approx(C.RANK_SHARES[0] * pool, abs=1e-9)}
        assert burned(vector) == pytest.approx(1.0 - C.RANK_SHARES[0] * pool, abs=1e-9)

    def test_the_field_behind_the_winner_is_paid_whether_or_not_it_cleared(self):
        """The bar is on the leader. Once it is taken, the run pays by rank."""
        vector = champion_ladder(
            [(7, "5A", 0.60), (9, "5B", 0.55), (3, "5C", 0.10)],
            run_id=413,
            block=1,
            champion_grade=0.50,
        )

        assert set(paid(vector)) == {7, 9, 3}

    def test_a_strong_field_behind_a_leader_who_did_not_win_is_paid_nothing(self):
        """The converse, and the one that costs someone a run if misread.

        Second place cannot inherit the run. If the best candidate did not beat
        what the network already has, the run bought nothing.

        The leader's grade has to sit inside CHAMPION_DETHRONE_MARGIN of the
        throne, so it moves whenever the margin does: 0.501 was inside at 0.002
        and at 0.001, and outside at 0.0005 — at which point this stopped
        testing the case it names. The tripwire below is what catches that.
        """
        inside = 0.50 + C.CHAMPION_DETHRONE_MARGIN / 2
        vector = champion_ladder(
            [(7, "5A", inside), (9, "5B", 0.500)], run_id=413, block=1, champion_grade=0.50
        )

        assert paid(vector) == {}
        assert burned(vector) == pytest.approx(1.0, abs=1e-9)

    def test_the_margin_is_the_one_the_protocol_sets(self):
        """A tripwire on a policy number, and one this repo did not have.

        The engine pins it; here nothing did, so the margin could shrink under
        these cases and they would pass while testing nothing. It has done that
        twice.
        """
        assert C.CHAMPION_DETHRONE_MARGIN == 0.0005, (
            "the dethrone margin moved; the grades in this file need rechecking"
        )

    def test_an_empty_throne_pays_the_field_as_it_stands(self):
        """How the first throne is filled: no incumbent, so no bar."""
        assert dethrone_threshold(None) == 0.0

        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)
        assert vector.champion_hotkey == "5A"


class TestItRefusesRatherThanPayingTheWrongMiner:
    def test_an_unsorted_field_raises(self):
        with pytest.raises(ValueError, match="descending grade"):
            champion_ladder(
                [(1, "5A", 0.30), (2, "5B", 0.50)], run_id=412, block=1, champion_grade=None
            )

    def test_a_reference_in_the_field_raises(self):
        """The reference is the bar, not a competitor."""
        from capability_subnet.scoring.references import BASE_MODEL

        with pytest.raises(ValueError, match="permanent reference"):
            champion_ladder([(1, BASE_MODEL, 0.9)], run_id=412, block=1, champion_grade=None)

    def test_a_burn_share_outside_the_unit_interval_raises(self):
        with pytest.raises(ValueError, match="burn_share"):
            champion_ladder(FIELD, run_id=412, block=1, champion_grade=None, burn_share=1.5)


class TestTheSharesAreWhatWasPublished:
    def test_they_account_for_the_whole_pool(self):
        assert sum(C.RANK_SHARES) + C.TAIL_SHARE == pytest.approx(1.0, abs=1e-12)

    def test_the_ladder_descends(self):
        assert list(C.RANK_SHARES) == sorted(C.RANK_SHARES, reverse=True)
        assert C.RANK_SHARES[-1] >= C.TAIL_SHARE

    def test_rank_shares_stops_at_the_last_paid_rank(self):
        assert sum(rank_shares(C.PAID_RANKS)) == pytest.approx(1.0, abs=1e-12)
        assert sum(rank_shares(C.PAID_RANKS + 5)) == pytest.approx(1.0, abs=1e-12)
