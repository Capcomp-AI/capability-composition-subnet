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
    def test_only_the_throne_is_paid(self):
        """PAID_RANKS is one. Placing is published, not paid."""
        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)

        assert list(paid(vector)) == [FIELD[0][0]]

    def test_the_winner_takes_its_share_of_the_pool(self):
        """Rank shares are of the *miner pool*, not of the run."""
        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)
        pool = 1.0 - C.BURN_SHARE

        assert paid(vector)[FIELD[0][0]] == pytest.approx(C.RANK_SHARES[0] * pool, abs=1e-9)

    def test_what_the_places_would_have_taken_burns(self):
        """It does not go to the winner.

        The winner's share is RANK_SHARES[0] of the pool whether or not anyone
        placed behind it, so the rest of the pool is emission the network did
        not spend rather than a bigger prize.
        """
        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)
        pool = 1.0 - C.BURN_SHARE

        assert burned(vector) == pytest.approx(1.0 - C.RANK_SHARES[0] * pool, abs=1e-9)
        assert burned(vector) > C.BURN_SHARE, "the unpaid places must burn, not vanish"

    def test_nothing_below_the_last_paid_rank_earns(self):
        vector = champion_ladder(FIELD, run_id=412, block=1, champion_grade=None)

        assert len(paid(vector)) == C.PAID_RANKS
        assert FIELD[C.PAID_RANKS][0] not in paid(vector)

    def test_the_rank_cap_is_the_one_the_protocol_sets(self):
        """A tripwire on a policy number. Every case in this class is written
        for a one-rank ladder; restoring places has to fail here first."""
        assert C.PAID_RANKS == 1, "the rank cap moved; these cases need rewriting"

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

    def test_the_field_behind_the_winner_is_paid_nothing(self):
        """The bar is on the leader, and so is the whole prize.

        uid 9 here is a good candidate by any reading — it clears the throne on
        its own, and under a ladder that paid places it would have taken second.
        It is paid nothing. That is the rule now: the number of eligible miners
        behind the winner does not change what any of them receive.
        """
        vector = champion_ladder(
            [(7, "5A", 0.60), (9, "5B", 0.55), (3, "5C", 0.10)],
            run_id=413,
            block=1,
            champion_grade=0.50,
        )

        assert set(paid(vector)) == {7}

    def test_nine_eligible_behind_the_winner_are_still_paid_nothing(self):
        """The case the shape of the old ladder makes tempting to assume.

        Ten qualify, every one of them clears the throne, and nine of them
        would have shared the tail under RANK_SHARES. One is paid.
        """
        # From uid 1: uid 0 is BURN_UID, and a champion sitting on it is merged
        # into the burn entry by _normalise, which would read here as the winner
        # taking the whole run.
        field = [(uid, f"5{uid}", 0.60 - uid / 1000) for uid in range(1, 11)]
        vector = champion_ladder(field, run_id=413, block=1, champion_grade=0.50)

        assert set(paid(vector)) == {1}
        pool = 1.0 - C.BURN_SHARE
        assert paid(vector)[1] == pytest.approx(C.RANK_SHARES[0] * pool, abs=1e-9)
        assert burned(vector) == pytest.approx(1.0 - C.RANK_SHARES[0] * pool, abs=1e-9)

    def test_a_strong_field_behind_a_leader_who_did_not_win_is_paid_nothing(self):
        """The converse, and the one that costs someone a run if misread.

        Second place cannot inherit the run. If the best candidate did not beat
        what the network already has, the run bought nothing.
        """
        vector = champion_ladder(
            [(7, "5A", 0.501), (9, "5B", 0.500)], run_id=413, block=1, champion_grade=0.50
        )

        assert paid(vector) == {}
        assert burned(vector) == pytest.approx(1.0, abs=1e-9)

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
        """The paid shares no longer sum to the whole pool, and must not.

        With one rank paid the winner takes RANK_SHARES[0] and the remainder is
        emission the network did not spend. Asking for more ranks than are paid
        must not conjure the named shares back.
        """
        payable = sum(C.RANK_SHARES[: C.PAID_RANKS])
        assert sum(rank_shares(C.PAID_RANKS)) == pytest.approx(payable, abs=1e-12)
        assert sum(rank_shares(C.PAID_RANKS + 5)) == pytest.approx(payable, abs=1e-12)
        assert all(s == 0.0 for s in rank_shares(C.PAID_RANKS + 5)[C.PAID_RANKS :])
