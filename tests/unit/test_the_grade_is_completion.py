"""What the score a miner is ranked and paid on actually measures.

Nine tenths of the qualified score is end-to-end completion, and nine tenths of
the grade is the qualified score — so the number that orders the field and sets
the weights is, to within a couple of percent, how much of the workflow the
package finished. The subnet buys completed work, and the scoring says so.

The remaining slivers are not decoration. Every one of them is a hard gate as
well as a scored term, so the floor is enforced whatever the weight carries;
the weight only decides what *exceeding* a floor is worth, which is now very
little. These pin that arrangement, because a weight that drifts changes who
gets paid without changing anything that looks like a rule.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CandidateScores
from capability_subnet.scoring.contribution import ContributionInputs, contribution_score

NON_E2E = ("stage_balance", "ood", "retention", "token_efficiency", "artifact_efficiency")


class TestTheQualifiedScoreIsMostlyCompletion:
    def test_completion_takes_nine_tenths(self):
        assert C.WEIGHT_END_TO_END == pytest.approx(0.90)

    def test_everything_else_shares_the_last_tenth(self):
        rest = sum(C.QUALIFIED_SCORE_WEIGHTS[name] for name in NON_E2E)

        assert rest == pytest.approx(0.10)

    def test_the_weights_still_sum_to_one(self):
        assert sum(C.QUALIFIED_SCORE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_the_minor_axes_keep_their_relative_emphasis(self):
        """Their share shrank; their ordering against each other did not."""
        w = C.QUALIFIED_SCORE_WEIGHTS
        assert w["stage_balance"] > w["ood"] == w["token_efficiency"]
        assert w["ood"] > w["retention"] == w["artifact_efficiency"]


class TestTheGradeIsMostlyTheQualifiedScore:
    def test_quality_takes_nine_tenths(self):
        assert C.CONTRIBUTION_WEIGHT_QUALITY == pytest.approx(0.90)

    def test_improvement_and_cost_share_the_last_tenth(self):
        rest = C.CONTRIBUTION_WEIGHT_IMPROVEMENT + C.CONTRIBUTION_WEIGHT_COST

        assert rest == pytest.approx(0.10)

    def test_they_keep_their_three_to_one_split(self):
        assert C.CONTRIBUTION_WEIGHT_IMPROVEMENT == pytest.approx(3 * C.CONTRIBUTION_WEIGHT_COST)

    def test_the_weights_still_sum_to_one(self):
        total = (
            C.CONTRIBUTION_WEIGHT_QUALITY
            + C.CONTRIBUTION_WEIGHT_IMPROVEMENT
            + C.CONTRIBUTION_WEIGHT_COST
        )
        assert total == pytest.approx(1.0)


class TestSoTheGradeTracksCompletion:
    """The consequence, stated as a number rather than left implied."""

    def test_completion_is_four_fifths_of_the_grade(self):
        share = C.CONTRIBUTION_WEIGHT_QUALITY * C.WEIGHT_END_TO_END

        assert share == pytest.approx(0.81)

    def _grade(self, end_to_end: float, **rest: float) -> float:
        scores = CandidateScores(
            end_to_end=end_to_end,
            stage_balance=rest.get("stage_balance", 0.5),
            ood=rest.get("ood", 0.5),
            retention=rest.get("retention", 1.0),
            token_efficiency=rest.get("token_efficiency", 0.5),
            artifact_efficiency=rest.get("artifact_efficiency", 0.5),
            qualified_score=rest.get("qualified_score", 0.0),
        )
        # The qualified score is computed upstream from the same weights; here
        # it is supplied directly so the grade's own weighting is what is under
        # test rather than the aggregation feeding it.
        return contribution_score(
            ContributionInputs(scores=scores, reference_e2e=rest.get("reference", 0.10))
        )

    def test_a_better_finisher_grades_higher(self):
        assert self._grade(0.30, qualified_score=0.30) > self._grade(0.20, qualified_score=0.20)

    def test_cost_cannot_outrank_completion(self):
        """The whole point of 90/10: cheap and unreliable does not win."""
        finisher = self._grade(0.30, qualified_score=0.30, token_efficiency=0.0)
        cheap = self._grade(0.20, qualified_score=0.20, token_efficiency=1.0)

        assert finisher > cheap

    def test_cost_still_separates_two_equal_finishers(self):
        """It is the only term left that can, which is why it is not zero."""
        cheap = self._grade(0.25, qualified_score=0.25, token_efficiency=1.0)
        dear = self._grade(0.25, qualified_score=0.25, token_efficiency=0.0)

        assert cheap > dear
