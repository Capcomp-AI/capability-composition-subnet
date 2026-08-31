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


class TestTheThroneIsARecordAndNotACondition:
    """``dethrone_threshold`` still decides who is recorded as champion.

    It stopped deciding who is paid. Keeping the two apart is the point of the
    rule, so both halves are asserted: the threshold still behaves exactly as it
    did, and the ladder no longer consults it before paying.
    """

    def test_an_empty_throne_has_a_threshold_of_zero(self):
        """How the first throne is filled: the field is ranked as it stands."""
        assert dethrone_threshold(None) == 0.0

    def test_a_held_throne_adds_the_margin(self):
        assert dethrone_threshold(0.40) == pytest.approx(0.40 + C.CHAMPION_DETHRONE_MARGIN)

    def test_the_threshold_does_not_change_what_is_paid(self):
        """The same field, against an empty throne and against an unbeatable
        one, is paid identically."""
        field = [(7, "5A", 0.30), (9, "5B", 0.29)]
        empty = champion_ladder(field, run_id=413, block=1, champion_grade=None)
        held = champion_ladder(field, run_id=413, block=1, champion_grade=0.99)

        assert paid(empty) == paid(held)


class TestTheRunPaysWhatClearedItsGates:
    """Payment follows the hard gates, not the incumbent.

    It used to follow the incumbent: a run paid nothing unless its leader
    exceeded the reigning grade by ``CHAMPION_DETHRONE_MARGIN``. That made
    emission depend on a quantity no candidate in the run could see or affect —
    a grade earned on a different draw — so a field that cleared every absolute
    bar could be paid nothing because an earlier run happened to be strong. Run
    415 burned entirely with five qualified packages in it.

    The entry gate is the bar now, and it is absolute: completion over the
    strongest permanent reference by ``DEFAULT_END_TO_END_MARGIN``. Everything
    reaching ``champion_ladder`` has already cleared it.
    """

    def test_a_field_below_the_incumbent_is_still_paid(self):
        """The case that motivated the change."""
        champion = FIELD[0][2]
        vector = champion_ladder(FIELD, run_id=413, block=1, champion_grade=champion)

        assert paid(vector) != {}

    def test_being_paid_the_champion_share_is_not_holding_the_throne(self):
        """Rank one always carries the "champion" role — it takes the
        champion's share — and holds the throne only if it beat the reigning
        grade. Conflating the two records every run's leader as having taken a
        throne it did not take, and the next run is then measured against a bar
        nobody actually cleared."""
        champion = FIELD[0][2]
        vector = champion_ladder(FIELD, run_id=413, block=1, champion_grade=champion)

        assert any(e.role == "champion" for e in vector.entries)
        assert vector.champion_hotkey is None

    def test_matching_the_margin_exactly_does_not_take_the_throne_but_is_paid(self):
        """A tie is not an improvement, and neither is the margin itself.

        The throne is a record of who leads the network; it still needs beating
        outright. Being paid for this run is a separate question.
        """
        champion = 0.40
        exactly = [(7, "5A", champion + C.CHAMPION_DETHRONE_MARGIN)]
        vector = champion_ladder(exactly, run_id=413, block=1, champion_grade=champion)

        assert paid(vector) == {7: pytest.approx(C.RANK_SHARES[0] * (1.0 - C.BURN_SHARE))}

    def test_an_empty_field_still_pays_nobody(self):
        """The one case that must keep burning.

        Nothing cleared the gates, so nothing was bought. This is the whole of
        what a total burn now means, and it is a statement about the packages
        rather than about the throne.
        """
        vector = champion_ladder([], run_id=413, block=1, champion_grade=0.50)

        assert paid(vector) == {}
        assert burned(vector) == pytest.approx(1.0, abs=1e-9)
        assert vector.champion_hotkey is None

    def test_a_winner_alone_in_the_field_takes_the_first_share_and_the_rest_burns(self):
        vector = champion_ladder([(7, "5A", 0.60)], run_id=413, block=1, champion_grade=0.50)
        pool = 1.0 - C.BURN_SHARE

        assert paid(vector) == {7: pytest.approx(C.RANK_SHARES[0] * pool, abs=1e-9)}
        assert burned(vector) == pytest.approx(1.0 - C.RANK_SHARES[0] * pool, abs=1e-9)

    def test_the_whole_field_is_paid_by_rank(self):
        vector = champion_ladder(
            [(7, "5A", 0.60), (9, "5B", 0.55), (3, "5C", 0.10)],
            run_id=413,
            block=1,
            champion_grade=0.50,
        )

        assert set(paid(vector)) == {7, 9, 3}

    def test_a_strong_field_behind_a_leader_who_did_not_win_is_paid(self):
        """The converse of the old rule, and the point of the change.

        Second place still cannot inherit the leader's share — the ladder pays
        position — but the field is no longer unpaid because the run failed to
        beat a package measured on another draw.
        """
        field = [(7, "5A", 0.30), (9, "5B", 0.29), (3, "5C", 0.28)]
        vector = champion_ladder(field, run_id=413, block=1, champion_grade=0.90)
        pool = 1.0 - C.BURN_SHARE

        assert set(paid(vector)) == {7, 9, 3}
        assert paid(vector)[7] == pytest.approx(C.RANK_SHARES[0] * pool, abs=1e-9)


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


def _graded_report(uid, hotkey, e2e, qs, *, passed=True):
    """A minimal signed-shape report carrying enough score to grade."""
    from capability_subnet.common.schemas import (
        CandidateScores,
        EvaluationReport,
        GateVerdict,
    )

    return EvaluationReport(
        run_id=419,
        evaluated_at_block=1,
        miner_hotkey=hotkey,
        miner_uid=uid,
        candidate_id=hotkey,
        base_revision="rev",
        source_snapshot_sha256="sha256:" + "a" * 64,
        evaluator_image_digest="sha256:" + "b" * 64,
        hard_gates=[
            GateVerdict(name="artifact_size", passed=True),
            GateVerdict(name="baseline", passed=passed),
        ],
        scores=CandidateScores(end_to_end=e2e, qualified_score=qs),
        strongest_reference_id="reference:base_model",
        strongest_reference_score=0.10,
        verdict="held" if passed else "terminated",
    )


def test_vector_from_reports_pays_gate_clearers_by_grade_and_drops_the_rest():
    """A validator derives the same ladder from reports that it would from its
    own measurements: gate-clearers ranked by grade, everyone else absent."""
    from capability_subnet.scoring.weight_vector import vector_from_reports

    reports = [
        _graded_report(1, "5A", 0.30, 0.30),
        _graded_report(2, "5B", 0.20, 0.20),
        _graded_report(3, "5C", 0.15, 0.15),
        # Best scores of the field, but it failed a hard gate: it earns nothing.
        _graded_report(4, "5D", 0.40, 0.40, passed=False),
    ]
    vector = vector_from_reports(reports, run_id=420, block=1, champion_grade=None, burn_uid=0)

    paid = {e.uid: e.weight for e in vector.entries if e.role != "burn" and e.weight > 0}
    assert 4 not in paid
    assert set(paid) == {1, 2, 3}
    assert paid[1] > paid[2] > paid[3]
    assert next(e for e in vector.entries if e.role == "champion").uid == 1
