"""The dethrone rule and the weight vector.

Together these decide who holds the throne and who gets paid. The tests below
are written around the properties that make the mechanism defensible rather than
around its implementation: a challenger that abandons a capability cannot win, a
copy of the champion cannot win, a reference on the throne earns nothing, and an
unfilled share burns instead of being handed to whoever came closest.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import ChampionRecord
from capability_subnet.scoring import references as ref
from capability_subnet.scoring.comparator import (
    ComparatorConfig,
    compare,
    compare_axis,
    decisive_loss,
    strongest_reference,
)
from capability_subnet.scoring.weight_vector import (
    apply_validator_burn,
    graded_top3,
    winner_take_all,
)
from capability_subnet.testing import make_results

AXES = ("stage_a", "stage_b", "stage_c")
CONFIG = ComparatorConfig(min_axis_samples=5, min_dominant_axes=1)


def rows(stage_scores: dict[str, float], *, success_rate: float, count: int = 40):
    return make_results(stage_scores, count=count, success_rate=success_rate)


class TestAxisComparison:
    def test_a_clear_improvement_is_dominant(self):
        verdict = compare_axis(
            "stage_a",
            rows({"stage_a": 0.9}, success_rate=1.0),
            rows({"stage_a": 0.6}, success_rate=1.0),
            CONFIG,
        )
        assert verdict.verdict == "dominant"

    def test_a_tie_is_not_worse_but_not_dominant(self):
        verdict = compare_axis(
            "stage_a",
            rows({"stage_a": 0.7}, success_rate=1.0),
            rows({"stage_a": 0.7}, success_rate=1.0),
            CONFIG,
        )
        assert verdict.verdict == "not_worse"

    def test_a_small_regression_stays_inside_the_tolerance(self):
        verdict = compare_axis(
            "stage_a",
            rows({"stage_a": 0.995}, success_rate=1.0),
            rows({"stage_a": 1.0}, success_rate=1.0),
            CONFIG,
        )
        assert verdict.verdict == "not_worse"

    def test_a_real_regression_is_worse(self):
        verdict = compare_axis(
            "stage_a",
            rows({"stage_a": 0.4}, success_rate=1.0),
            rows({"stage_a": 0.9}, success_rate=1.0),
            CONFIG,
        )
        assert verdict.verdict == "worse"

    def test_too_few_paired_samples_counts_as_worse(self):
        # Absence of evidence that a capability was kept is not evidence that it
        # was kept.
        verdict = compare_axis(
            "stage_a",
            rows({"stage_a": 1.0}, success_rate=1.0, count=3),
            rows({"stage_a": 0.0}, success_rate=1.0, count=3),
            CONFIG,
        )
        assert verdict.verdict == "worse"
        assert verdict.paired_samples == 3


class TestDethroneRule:
    def _compare(self, challenger, champion, reference=None, config=CONFIG):
        return compare(
            challenger,
            champion,
            reference if reference is not None else champion,
            axes=AXES,
            reference_id="incumbent",
            config=config,
            bootstrap_seed=7,
        )

    def test_a_genuine_improvement_takes_the_throne(self):
        challenger = rows(dict.fromkeys(AXES, 1.0), success_rate=0.95)
        champion = rows(dict.fromkeys(AXES, 0.7), success_rate=0.60)

        outcome = self._compare(challenger, champion)
        assert outcome.dethrones, outcome.reason

    def test_abandoning_one_capability_blocks_a_dethrone(self):
        # Better on two axes and far better end-to-end, but it gave up stage_c.
        challenger = rows({"stage_a": 1.0, "stage_b": 1.0, "stage_c": 0.2}, success_rate=0.95)
        champion = rows({"stage_a": 0.7, "stage_b": 0.7, "stage_c": 0.9}, success_rate=0.60)

        outcome = self._compare(challenger, champion)
        assert not outcome.dethrones
        assert "regressed on stage_c" in outcome.reason

    def test_an_exact_copy_of_the_champion_cannot_win(self):
        # The defender advantage: reproducing the champion's scores is not a
        # margin, so copying is worthless even when it is undetectable.
        identical = dict.fromkeys(AXES, 0.8)
        outcome = self._compare(
            rows(identical, success_rate=0.75), rows(identical, success_rate=0.75)
        )
        assert not outcome.dethrones

    def test_a_win_too_small_to_clear_the_margin_is_refused(self):
        challenger = rows(dict.fromkeys(AXES, 0.85), success_rate=0.71)
        champion = rows(dict.fromkeys(AXES, 0.80), success_rate=0.70)

        outcome = self._compare(challenger, champion)
        assert not outcome.dethrones
        assert "margin" in outcome.reason

    def test_the_challenger_must_beat_the_strongest_reference_not_the_incumbent(self):
        challenger = rows(dict.fromkeys(AXES, 1.0), success_rate=0.75)
        weak_champion = rows(dict.fromkeys(AXES, 0.5), success_rate=0.30)
        strong_reference = rows(dict.fromkeys(AXES, 0.95), success_rate=0.90)

        outcome = compare(
            challenger,
            weak_champion,
            strong_reference,
            axes=AXES,
            reference_id="reference:equal_ties_svd_merge",
            config=CONFIG,
            bootstrap_seed=7,
        )
        assert not outcome.dethrones
        assert "equal_ties_svd_merge" in outcome.reason

    def test_strict_pareto_requires_every_axis(self):
        challenger = rows({"stage_a": 1.0, "stage_b": 1.0, "stage_c": 0.8}, success_rate=0.95)
        champion = rows({"stage_a": 0.6, "stage_b": 0.6, "stage_c": 0.8}, success_rate=0.55)

        partial = self._compare(challenger, champion)
        strict = self._compare(
            challenger,
            champion,
            config=ComparatorConfig(min_axis_samples=5, strict_pareto=True),
        )

        assert partial.dethrones
        assert not strict.dethrones

    def test_the_outcome_records_what_it_was_based_on(self):
        outcome = self._compare(
            rows(dict.fromkeys(AXES, 1.0), success_rate=0.95),
            rows(dict.fromkeys(AXES, 0.6), success_rate=0.50),
        )
        assert len(outcome.per_axis_verdicts) == len(AXES)
        assert outcome.paired is not None
        assert outcome.paired.paired_instances > 0


class TestDecisiveLoss:
    def test_a_measured_loss_terminates_the_challenger(self):
        outcome = compare(
            rows(dict.fromkeys(AXES, 0.3), success_rate=0.2),
            rows(dict.fromkeys(AXES, 0.9), success_rate=0.9),
            rows(dict.fromkeys(AXES, 0.9), success_rate=0.9),
            axes=AXES,
            reference_id="incumbent",
            config=CONFIG,
        )
        assert decisive_loss(outcome)

    def test_an_unmeasured_axis_is_not_a_decisive_loss(self):
        # The engine failed to gather evidence. Spending the hotkey's one shot on
        # that would punish a miner for the engine's bad night.
        outcome = compare(
            rows(dict.fromkeys(AXES, 0.9), success_rate=0.9),
            [],
            [],
            axes=AXES,
            reference_id="incumbent",
            config=CONFIG,
        )
        assert not outcome.dethrones
        assert not decisive_loss(outcome)


class TestStrongestReference:
    def test_picks_the_maximum(self):
        name, value = strongest_reference({"a": 0.4, "b": 0.9, "c": 0.7})
        assert (name, value) == ("b", 0.9)

    def test_ties_resolve_stably(self):
        assert strongest_reference({"z": 0.9, "a": 0.9})[0] == "a"

    def test_no_references_is_handled(self):
        assert strongest_reference({}) == ("", 0.0)

    def test_single_adapter_references_collapse_to_the_best_one(self):
        collapsed = ref.collapse_single_adapters(
            {
                f"{ref.BEST_SINGLE}:alpha": 0.3,
                f"{ref.BEST_SINGLE}:beta": 0.6,
                ref.BASE_MODEL: 0.2,
            }
        )
        assert collapsed[ref.BEST_SINGLE] == 0.6
        assert ref.BASE_MODEL in collapsed
        assert f"{ref.BEST_SINGLE}:alpha" not in collapsed


class TestWeightVector:
    def _champion(self, **overrides) -> ChampionRecord:
        payload = {"candidate_id": "5Miner", "hotkey": "5Miner", "uid": 7}
        payload.update(overrides)
        return ChampionRecord(**payload)

    def test_the_champion_takes_the_whole_share(self):
        vector = winner_take_all(self._champion(), window_id=1, block=100)
        assert vector.entries == [vector.entries[0]]
        assert vector.entries[0].uid == 7
        assert vector.entries[0].weight == pytest.approx(1.0)

    def test_an_empty_throne_burns(self):
        vector = winner_take_all(None, window_id=1, block=100)
        assert vector.entries[0].uid == C.BURN_UID
        assert vector.entries[0].role == "burn"

    def test_a_reference_on_the_throne_earns_nothing(self):
        # A reference holding the throne means no miner has beaten an
        # off-the-shelf merge yet. Paying for that would be paying the operator.
        champion = self._champion(
            candidate_id=ref.EQUAL_TIES, hotkey="5Operator", uid=3, is_reference=True
        )
        vector = winner_take_all(champion, window_id=1, block=100)

        assert vector.entries[0].role == "burn"
        assert vector.champion_hotkey is None

    def test_the_burn_valve_splits_the_share(self):
        vector = winner_take_all(
            self._champion(), window_id=1, block=100, burn_percentage=0.25
        )
        by_uid = {entry.uid: entry.weight for entry in vector.entries}
        assert by_uid[7] == pytest.approx(0.75)
        assert by_uid[C.BURN_UID] == pytest.approx(0.25)

    def test_a_champion_on_the_burn_uid_does_not_produce_a_duplicate(self):
        # The chain rejects a repeated UID, so the two entries must merge.
        vector = winner_take_all(
            self._champion(uid=C.BURN_UID),
            window_id=1,
            block=100,
            burn_percentage=0.3,
        )
        uids = [entry.uid for entry in vector.entries]
        assert len(uids) == len(set(uids))
        assert sum(entry.weight for entry in vector.entries) == pytest.approx(1.0)

    def test_graded_mode_uses_the_published_split(self):
        vector = graded_top3(
            [(1, "5A"), (2, "5B"), (3, "5C")], window_id=1, block=100
        )
        by_uid = {entry.uid: entry.weight for entry in vector.entries}
        assert by_uid[1] == pytest.approx(0.60)
        assert by_uid[2] == pytest.approx(0.25)
        assert by_uid[3] == pytest.approx(0.15)

    def test_unfilled_graded_ranks_burn_rather_than_promoting_anyone(self):
        vector = graded_top3([(1, "5A")], window_id=1, block=100)
        by_uid = {entry.uid: entry.weight for entry in vector.entries}

        assert by_uid[1] == pytest.approx(0.60)
        assert by_uid[C.BURN_UID] == pytest.approx(0.40)

    def test_graded_mode_with_nobody_qualified_burns_everything(self):
        vector = graded_top3([], window_id=1, block=100)
        assert vector.entries[0].uid == C.BURN_UID
        assert vector.entries[0].weight == pytest.approx(1.0)

    def test_every_vector_sums_to_one(self):
        for vector in (
            winner_take_all(self._champion(), window_id=1, block=1),
            winner_take_all(None, window_id=1, block=1),
            winner_take_all(
                self._champion(), window_id=1, block=1, burn_percentage=0.37
            ),
            graded_top3([(1, "a"), (2, "b")], window_id=1, block=1),
        ):
            assert sum(entry.weight for entry in vector.entries) == pytest.approx(1.0)

    def test_a_validator_may_burn_more_but_the_vector_stays_normalised(self):
        published = winner_take_all(self._champion(), window_id=1, block=1)
        adjusted = apply_validator_burn(published, 0.5, C.BURN_UID)

        by_uid = {entry.uid: entry.weight for entry in adjusted.entries}
        assert by_uid[7] == pytest.approx(0.5)
        assert by_uid[C.BURN_UID] == pytest.approx(0.5)
        assert sum(by_uid.values()) == pytest.approx(1.0)

    def test_zero_extra_burn_leaves_the_vector_untouched(self):
        published = winner_take_all(self._champion(), window_id=1, block=1)
        assert apply_validator_burn(published, 0.0, C.BURN_UID) is published

    def test_uid_and_weight_lists_stay_aligned(self):
        vector = graded_top3(
            [(4, "5A"), (9, "5B")], window_id=1, block=1, burn_percentage=0.1
        )
        uids, weights = vector.as_uid_weight_lists()
        assert len(uids) == len(weights)
        assert sum(weights) == pytest.approx(1.0)
