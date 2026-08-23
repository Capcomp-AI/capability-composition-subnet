"""What the score a miner is ranked and paid on actually measures.

Breadth. ``stage_balance`` is the geometric mean across the twelve capability
axes, and it carries 95% of the qualified score, which in turn carries 95% of
the grade — so the number that orders the field and sets the weights is, to
within a few percent, how evenly capable the package is across all twelve.
A subnet buying *composition* is buying breadth, and the scoring says so.

The remaining slivers are not decoration. Every one of them is a hard gate as
well as a scored term, so the floor is enforced whatever the weight carries;
the weight only decides what *exceeding* a floor is worth. These pin the
arrangement, because a weight that drifts changes who gets paid without
changing anything that looks like a rule.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CandidateScores
from capability_subnet.scoring.contribution import ContributionInputs, contribution_score

MINOR = ("end_to_end", "ood", "retention", "token_efficiency", "artifact_efficiency")


class TestTheQualifiedScoreIsMostlyBreadth:
    def test_stage_balance_takes_ninety_five_percent(self):
        assert C.WEIGHT_STAGE_BALANCE == pytest.approx(0.95)

    def test_everything_else_shares_the_last_twentieth(self):
        assert sum(C.QUALIFIED_SCORE_WEIGHTS[name] for name in MINOR) == pytest.approx(0.05)

    def test_the_weights_still_sum_to_one(self):
        assert sum(C.QUALIFIED_SCORE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_completion_leads_the_minor_axes(self):
        """It is small but not the smallest: of what is left, it is most of it."""
        w = C.QUALIFIED_SCORE_WEIGHTS
        assert w["end_to_end"] > w["ood"] > w["retention"]
        assert w["retention"] == pytest.approx(w["artifact_efficiency"])


class TestTheGradeIsMostlyTheQualifiedScore:
    def test_quality_takes_ninety_five_percent(self):
        assert C.CONTRIBUTION_WEIGHT_QUALITY == pytest.approx(0.95)

    def test_improvement_and_cost_share_the_last_twentieth(self):
        rest = C.CONTRIBUTION_WEIGHT_IMPROVEMENT + C.CONTRIBUTION_WEIGHT_COST
        assert rest == pytest.approx(0.05)

    def test_they_keep_their_three_to_one_split(self):
        assert C.CONTRIBUTION_WEIGHT_IMPROVEMENT == pytest.approx(3 * C.CONTRIBUTION_WEIGHT_COST)

    def test_the_weights_still_sum_to_one(self):
        total = (
            C.CONTRIBUTION_WEIGHT_QUALITY
            + C.CONTRIBUTION_WEIGHT_IMPROVEMENT
            + C.CONTRIBUTION_WEIGHT_COST
        )
        assert total == pytest.approx(1.0)


class TestTokenSpendIsCountedOnce:
    """It enters twice — inside the qualified score and again as the cost term.

    Halving its share of the qualified score is what keeps the double count
    from making running cost the loudest thing in a grade meant to measure
    breadth.
    """

    def test_its_share_inside_quality_matches_the_smallest_axes(self):
        w = C.QUALIFIED_SCORE_WEIGHTS
        assert w["token_efficiency"] == pytest.approx(w["retention"])

    def test_its_total_influence_on_the_grade_is_bounded(self):
        total = (
            C.WEIGHT_TOKEN_EFFICIENCY * C.CONTRIBUTION_WEIGHT_QUALITY + C.CONTRIBUTION_WEIGHT_COST
        )
        assert total < 0.02, f"token spend carries {total:.4f} of the grade"


class TestSoTheGradeTracksBreadth:
    def test_breadth_is_most_of_the_grade(self):
        share = C.CONTRIBUTION_WEIGHT_QUALITY * C.WEIGHT_STAGE_BALANCE
        assert share == pytest.approx(0.9025)

    def _grade(self, stage_balance: float, **rest: float) -> float:
        scores = CandidateScores(
            end_to_end=rest.get("end_to_end", 0.15),
            stage_balance=stage_balance,
            ood=rest.get("ood", 0.12),
            retention=rest.get("retention", 1.0),
            token_efficiency=rest.get("token_efficiency", 0.9),
            artifact_efficiency=rest.get("artifact_efficiency", 0.9),
            qualified_score=rest.get("qualified_score", 0.0),
        )
        return contribution_score(
            ContributionInputs(scores=scores, reference_e2e=rest.get("reference", 0.12))
        )

    def test_a_broader_package_grades_higher(self):
        broad = self._grade(0.05, qualified_score=0.05)
        narrow = self._grade(0.02, qualified_score=0.02)

        assert broad > narrow

    def test_breadth_outweighs_cheapness(self):
        broad = self._grade(0.05, qualified_score=0.05, token_efficiency=0.0)
        cheap = self._grade(0.02, qualified_score=0.02, token_efficiency=1.0)

        assert broad > cheap

    def test_cost_still_separates_two_equally_broad_packages(self):
        cheap = self._grade(0.03, qualified_score=0.03, token_efficiency=1.0)
        dear = self._grade(0.03, qualified_score=0.03, token_efficiency=0.0)

        assert cheap > dear
