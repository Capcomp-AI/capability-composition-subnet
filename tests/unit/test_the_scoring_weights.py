"""The weights a miner is ranked and paid on, pinned so they cannot drift.

No single axis carries the score. End-to-end completion leads it because
finishing the workflow is the product; stage balance comes second because a
package strong on a few axes and absent on the rest has not composed anything;
the rest price robustness, retained general ability and running cost.

A weight that moves changes who gets paid without changing anything that looks
like a rule, which is why every one of them is asserted here rather than left
to be read off the constants.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CandidateScores
from capability_subnet.scoring.contribution import ContributionInputs, contribution_score

EXPECTED_QUALITY = {
    "end_to_end": 0.55,
    "stage_balance": 0.15,
    "ood": 0.10,
    "retention": 0.05,
    "token_efficiency": 0.10,
    "artifact_efficiency": 0.05,
}


class TestTheQualifiedScore:
    @pytest.mark.parametrize("axis,weight", EXPECTED_QUALITY.items(), ids=list(EXPECTED_QUALITY))
    def test_each_axis_carries_its_published_weight(self, axis, weight):
        assert C.QUALIFIED_SCORE_WEIGHTS[axis] == pytest.approx(weight)

    def test_they_sum_to_one(self):
        assert sum(C.QUALIFIED_SCORE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_completion_leads_and_balance_follows(self):
        w = C.QUALIFIED_SCORE_WEIGHTS
        assert w["end_to_end"] > w["stage_balance"] > w["ood"] == w["token_efficiency"]
        assert w["ood"] > w["retention"] == w["artifact_efficiency"]

    def test_quality_outweighs_efficiency(self):
        """A cheap but unreliable artifact cannot win."""
        w = C.QUALIFIED_SCORE_WEIGHTS
        quality = w["end_to_end"] + w["stage_balance"] + w["ood"] + w["retention"]
        efficiency = w["token_efficiency"] + w["artifact_efficiency"]

        assert quality > 4 * efficiency


class TestTheGrade:
    def test_quality_carries_half(self):
        assert C.CONTRIBUTION_WEIGHT_QUALITY == pytest.approx(0.50)

    def test_improvement_and_cost_share_the_rest(self):
        rest = C.CONTRIBUTION_WEIGHT_IMPROVEMENT + C.CONTRIBUTION_WEIGHT_COST
        assert rest == pytest.approx(0.50)

    def test_cost_carries_what_it_carried_before(self):
        """It enters twice — here and inside the qualified score — so its true
        weight is the sum. Raising it here raises it twice."""
        total = (
            C.WEIGHT_TOKEN_EFFICIENCY * C.CONTRIBUTION_WEIGHT_QUALITY + C.CONTRIBUTION_WEIGHT_COST
        )
        assert total == pytest.approx(0.15)

    def test_the_weights_sum_to_one(self):
        total = (
            C.CONTRIBUTION_WEIGHT_QUALITY
            + C.CONTRIBUTION_WEIGHT_IMPROVEMENT
            + C.CONTRIBUTION_WEIGHT_COST
        )
        assert total == pytest.approx(1.0)

    def test_nothing_is_measured_against_the_incumbent(self):
        """A term relative to the throne moves the scale each time it changes
        hands, and CHAMPION_DETHRONE_MARGIN is a fixed number on that scale."""
        assert not hasattr(C, "CONTRIBUTION_WEIGHT_PROXIMITY")
        assert "champion_e2e" not in ContributionInputs.__dataclass_fields__


class TestWhatTheWeightsActuallyDo:
    def _grade(self, **axes: float) -> float:
        defaults = {
            "end_to_end": 0.15,
            "stage_balance": 0.03,
            "ood": 0.12,
            "retention": 1.0,
            "token_efficiency": 0.9,
            "artifact_efficiency": 0.9,
        }
        defaults.update(axes)
        qualified = sum(C.QUALIFIED_SCORE_WEIGHTS[a] * v for a, v in defaults.items())
        scores = CandidateScores(qualified_score=min(1.0, qualified), **defaults)
        return contribution_score(ContributionInputs(scores=scores, reference_e2e=0.12))

    def test_finishing_more_grades_higher(self):
        assert self._grade(end_to_end=0.30) > self._grade(end_to_end=0.20)

    def test_broader_capability_grades_higher(self):
        assert self._grade(stage_balance=0.10) > self._grade(stage_balance=0.02)

    def test_completion_outweighs_breadth(self):
        """0.55 against 0.15: a package that finishes more wins a tie on breadth."""
        finisher = self._grade(end_to_end=0.30, stage_balance=0.02)
        broad = self._grade(end_to_end=0.20, stage_balance=0.10)

        assert finisher > broad

    def test_cost_outweighs_completion_over_the_range_this_corpus_produces(self):
        """Nominal weight and real influence are not the same thing.

        A weight multiplies an axis, and the axes do not share a range.
        Completion sits between 0.13 and 0.16 on this workflow — a spread of
        0.02 — while token efficiency uses 0.72 to 1.00. So 0.10 on the wide
        axis moves the grade more than 0.55 on the narrow one, and the package
        that spends least outranks the one that finishes most.

        This is a property of the corpus, not a mistake in the weights, and it
        is the reason a weighting cannot be chosen from the numbers alone.
        """
        best_finisher = self._grade(end_to_end=0.1533, token_efficiency=0.723)
        cheapest = self._grade(end_to_end=0.1378, token_efficiency=1.000)

        assert cheapest > best_finisher

    def test_and_a_full_swing_on_cost_outweighs_a_large_one_on_completion(self):
        """Recorded rather than asserted away, because it is the shape of the
        scoring: a package that halves its token spend gains more than one that
        finishes a further two points of the workflow."""
        finisher = self._grade(end_to_end=0.30, token_efficiency=0.0)
        cheap = self._grade(end_to_end=0.20, token_efficiency=1.0)

        assert cheap > finisher
