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
from capability_subnet.scoring import references as ref
from capability_subnet.scoring.comparator import (
    ComparatorConfig,
    compare,
    compare_axis,
    decisive_loss,
    strongest_reference,
)
from capability_subnet.scoring.weight_vector import apply_validator_burn, champion_ladder
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

    def test_the_incumbent_is_excluded_from_the_absolute_bar(self):
        """The bar is the permanent reference, not the last winner.

        Replaces a test that folded the per-adapter references into a best-
        single-adapter entry. There are no per-adapter references any more, so
        the only rule left in this helper is the one below — and it is the one
        that matters: counting the incumbent would make every new champion clear
        the previous one by a further fixed margin, a staircase that completion,
        being bounded by one, cannot climb far.
        """
        scores = {ref.BASE_MODEL: 0.2, ref.INCUMBENT: 0.6}

        bar = ref.bar_scores(scores, include_incumbent=False)
        assert bar == {ref.BASE_MODEL: 0.2}

        # Excluded from the bar, not hidden — the report still shows it.
        assert ref.INCUMBENT in ref.bar_scores(scores)


class TestWeightVector:
    """What survives a vector being built: normalisation, and a validator's
    right to burn more than the engine asked for.

    The split itself is checked in test_champion_ladder.py.
    """

    def _vector(self, **overrides):
        payload = {
            "ranked": [(7, "5A", 0.50), (9, "5B", 0.40)],
            "run_id": 1,
            "block": 100,
            "champion_grade": None,
        }
        payload.update(overrides)
        ranked = payload.pop("ranked")
        return champion_ladder(ranked, **payload)

    def test_every_vector_sums_to_one(self):
        for vector in (
            self._vector(),
            self._vector(ranked=[]),
            self._vector(champion_grade=0.99),
            self._vector(burn_share=0.37),
            self._vector(burn_share=0.0),
            self._vector(burn_share=1.0),
        ):
            assert sum(entry.weight for entry in vector.entries) == pytest.approx(1.0)

    def test_a_validator_may_burn_more_and_gets_exactly_what_it_asked_for(self):
        """Half again on top of what the vector already burns.

        Scaling only the miners and leaving the existing burn at full weight
        over-burns: the sum comes back to one after normalisation, so it looks
        right, and the miners are quietly paid less than the half they are owed.
        """
        published = self._vector(ranked=[(7, "5A", 0.50)], burn_share=0.0)
        before = {entry.uid: entry.weight for entry in published.entries}
        adjusted = apply_validator_burn(published, 0.5, C.BURN_UID)

        by_uid = {entry.uid: entry.weight for entry in adjusted.entries}
        assert by_uid[7] == pytest.approx(before[7] * 0.5)
        assert by_uid[C.BURN_UID] == pytest.approx(before[C.BURN_UID] * 0.5 + 0.5)
        assert sum(by_uid.values()) == pytest.approx(1.0)

    def test_burning_more_on_top_of_the_standard_burn_is_exact(self):
        published = self._vector(ranked=[(7, "5A", 0.50)])
        paid_before = next(e.weight for e in published.entries if e.role != "burn")
        adjusted = apply_validator_burn(published, 0.5, C.BURN_UID)
        paid_after = next(e.weight for e in adjusted.entries if e.role != "burn")

        assert paid_after == pytest.approx(paid_before * 0.5)

    def test_zero_extra_burn_leaves_the_vector_untouched(self):
        published = self._vector()
        assert apply_validator_burn(published, 0.0, C.BURN_UID) is published

    def test_uid_and_weight_lists_stay_aligned(self):
        uids, weights = self._vector().as_uid_weight_lists()
        assert len(uids) == len(weights)
        assert sum(weights) == pytest.approx(1.0)

    def test_a_repeated_uid_is_merged_rather_than_submitted_twice(self):
        """The chain rejects a vector naming one UID twice."""
        vector = self._vector(ranked=[(7, "5A", 0.50)], burn_share=0.8)

        uids, _ = vector.as_uid_weight_lists()
        assert len(uids) == len(set(uids))
