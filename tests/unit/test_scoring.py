"""Score aggregation, hard gates and the paired statistics.

The numbers here decide who gets paid, so each component is checked against a
hand-computable case rather than against whatever the pipeline happens to
produce.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CandidateScores
from capability_subnet.scoring import gates
from capability_subnet.scoring.aggregate import (
    EfficiencyInputs,
    aggregate_scores,
    artifact_efficiency,
    end_to_end_completion,
    percentile,
    stage_balance,
    valid_rows,
)
from capability_subnet.scoring.bootstrap import (
    outcome_map,
    paired_bootstrap,
    paired_differences,
)
from capability_subnet.scoring.sampler import common_instance_ids, draw_run
from capability_subnet.scoring.weight_vector import champion_ladder
from capability_subnet.testing import make_results

STAGES = ("stage_a", "stage_b", "stage_c")
FULL = dict.fromkeys(STAGES, 1.0)


class TestAggregation:
    def test_completion_counts_only_scored_rows(self):
        rows = make_results(FULL, count=10, success_rate=0.6)
        rows[0].error = "sandbox failure"

        # Nine rows survive, five of which succeeded (the failed one was a
        # success). Infrastructure trouble lowers the sample count, never the
        # score.
        assert len(valid_rows(rows)) == 9
        assert end_to_end_completion(rows) == pytest.approx(5 / 9)

    def test_completion_of_an_empty_set_is_zero_not_an_error(self):
        assert end_to_end_completion([]) == 0.0

    def test_stage_balance_punishes_a_single_abandoned_capability(self):
        even = make_results(dict.fromkeys(STAGES, 0.8))
        lopsided = make_results({"stage_a": 1.0, "stage_b": 1.0, "stage_c": 0.4})

        # The arithmetic means are 0.80 and 0.80 respectively, but the geometric
        # mean separates them — which is the whole point of using one.
        assert stage_balance(even, STAGES) > stage_balance(lopsided, STAGES)

    def test_stage_balance_of_a_perfect_package_is_one(self):
        assert stage_balance(make_results(FULL), STAGES) == pytest.approx(1.0)

    def test_a_zero_stage_does_not_erase_the_remaining_signal(self):
        one_zero = make_results({"stage_a": 1.0, "stage_b": 1.0, "stage_c": 0.0})
        two_zeros = make_results({"stage_a": 1.0, "stage_b": 0.0, "stage_c": 0.0})

        assert stage_balance(one_zero, STAGES) > stage_balance(two_zeros, STAGES) > 0.0

    # Retention itself is no longer derived from workflow completion — it is a
    # held-out probe, covered in test_incentive_and_retention.py. What remains
    # testable here is that the aggregate simply carries the value it is given.
    def test_the_aggregate_carries_the_retention_it_is_given(self):
        scores = aggregate_scores(
            make_results(FULL),
            [],
            STAGES,
            retention=0.5,
            efficiency=EfficiencyInputs(artifact_bytes=0),
        )
        assert scores.retention == 0.5

    def test_artifact_efficiency_rewards_headroom_below_the_size_limit(self):
        """Size alone. Peak memory used to carry half of this and could not: it
        tracked the operator's reservation, so it was the same constant for
        every candidate and halved the one term that actually varies."""
        assert artifact_efficiency(0) == pytest.approx(1.0)
        assert artifact_efficiency(C.MAX_ARTIFACT_BYTES) == 0.0
        assert artifact_efficiency(C.MAX_ARTIFACT_BYTES // 2) == pytest.approx(0.5)

    def test_percentile_uses_nearest_rank(self):
        # An interpolated percentile would report a duration that never occurred,
        # and a reported percentile is a promise about an observed run.
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        assert percentile(values, 0.95) == 100.0
        assert percentile(values, 0.5) == 3.0
        assert percentile([], 0.95) == 0.0

    def test_qualified_score_uses_the_published_weights(self):
        scores = aggregate_scores(
            make_results(FULL, count=20, success_rate=1.0, seconds=5.0),
            make_results(FULL, count=10, success_rate=1.0, prefix="ood"),
            STAGES,
            retention=1.0,
            efficiency=EfficiencyInputs(artifact_bytes=0),
        )
        # A perfect package on every component scores exactly 1.
        assert scores.qualified_score == pytest.approx(1.0)
        assert scores.end_to_end == 1.0
        assert scores.valid_samples == 20

    def test_quality_dominates_efficiency(self):
        efficient_but_wrong = aggregate_scores(
            make_results(dict.fromkeys(STAGES, 0.0), count=20, success_rate=0.0),
            [],
            STAGES,
            retention=1.0,
            efficiency=EfficiencyInputs(artifact_bytes=0),
        )
        costly_but_right = aggregate_scores(
            make_results(FULL, count=20, success_rate=1.0, seconds=30.0),
            [],
            STAGES,
            retention=1.0,
            efficiency=EfficiencyInputs(
                artifact_bytes=C.MAX_ARTIFACT_BYTES,
            ),
        )
        assert costly_but_right.qualified_score > efficient_but_wrong.qualified_score


class TestGates:
    def test_artifact_size_gate(self):
        assert gates.gate_artifact_size(C.MAX_ARTIFACT_BYTES).passed
        assert not gates.gate_artifact_size(C.MAX_ARTIFACT_BYTES + 1).passed

    def test_safety_gate_admits_no_tolerance(self):
        clean = make_results(FULL, count=20, unsafe=0)
        one_incident = make_results(FULL, count=20, unsafe=1)

        assert gates.gate_safety(clean).passed
        assert not gates.gate_safety(one_incident).passed

    def test_agent_limit_gate_uses_the_worst_instance(self):
        rows = make_results(FULL, count=5)
        rows[3].turns_used = C.MAX_AGENT_TURNS + 1
        assert not gates.gate_agent_limits(rows).passed

    def test_stage_floor_gate(self):
        # Floors now, not pass thresholds the gate halves for you: what counts as
        # "this axis is gone" is the workflow's to declare.
        floors = dict.fromkeys(STAGES, 0.5)

        healthy = CandidateScores(per_stage_means=dict.fromkeys(STAGES, 0.9))
        abandoned = CandidateScores(
            per_stage_means={"stage_a": 0.9, "stage_b": 0.9, "stage_c": 0.1}
        )

        assert gates.gate_stage_floors(healthy, floors).passed
        assert not gates.gate_stage_floors(abandoned, floors).passed

    def test_base_retention_gate(self):
        assert gates.gate_base_retention(C.BASE_RETENTION_FLOOR).passed
        assert not gates.gate_base_retention(C.BASE_RETENTION_FLOOR - 0.01).passed

    def test_baseline_gate_requires_the_full_margin(self):
        margin = C.DEFAULT_END_TO_END_MARGIN
        assert gates.gate_beats_strongest_reference(0.70, "ref", 0.70 - margin, margin).passed
        assert not gates.gate_beats_strongest_reference(
            0.70, "ref", 0.70 - margin + 0.001, margin
        ).passed

    def test_beating_the_reference_by_nothing_is_not_enough(self):
        assert not gates.gate_beats_strongest_reference(0.8, "ref", 0.8, 0.03).passed

    def test_sample_sufficiency_counts_only_scored_rows(self):
        rows = make_results(FULL, count=25)
        for row in rows[:10]:
            row.error = "sandbox failure"

        assert not gates.gate_sample_sufficiency(rows, 20).passed
        assert gates.gate_sample_sufficiency(rows, 15).passed

    def test_summarise_names_every_failure(self):
        verdicts = [
            gates.gate_artifact_size(C.MAX_ARTIFACT_BYTES + 1),
            gates.gate_sample_sufficiency([], minimum=C.DEFAULT_MIN_AXIS_SAMPLES),
        ]
        passed, detail = gates.summarise(verdicts)

        assert passed is False
        assert "artifact_size" in detail and "sample_sufficiency" in detail


class TestPairedBootstrap:
    def test_a_clear_improvement_is_significant(self):
        result = paired_bootstrap([0.4] * 60, seed=1)
        assert result.significant
        assert result.lower_confidence_bound > 0

    def test_no_difference_is_not_significant(self):
        result = paired_bootstrap([0.0] * 60, seed=1)
        assert not result.significant

    def test_a_noisy_wash_is_not_significant(self):
        # Equal numbers of wins and losses: the point estimate is zero and the
        # lower bound must not clear it.
        differences = [1.0, -1.0] * 40
        assert not paired_bootstrap(differences, seed=1).significant

    def test_a_narrow_win_against_high_variance_is_not_significant(self):
        differences = ([1.0] * 26) + ([-1.0] * 24)
        result = paired_bootstrap(differences, seed=1)
        assert result.difference > 0
        assert not result.significant

    def test_one_pair_carries_no_information(self):
        # A single lucky instance must never dethrone a champion.
        result = paired_bootstrap([1.0], seed=1)
        assert result.difference == 1.0
        assert result.lower_confidence_bound == 0.0
        assert not result.significant

    def test_empty_input_is_handled(self):
        assert paired_bootstrap([], seed=1).paired_count == 0

    def test_the_verdict_is_reproducible(self):
        differences = [1.0, 0.0, 1.0, -1.0] * 20
        first = paired_bootstrap(differences, seed=99)
        again = paired_bootstrap(differences, seed=99)
        assert first.lower_confidence_bound == again.lower_confidence_bound

    def test_pairing_uses_only_shared_instances(self):
        challenger = {"a": 1.0, "b": 1.0, "c": 1.0}
        reference = {"b": 0.0, "c": 1.0, "d": 0.0}
        assert paired_differences(challenger, reference) == [1.0, 0.0]

    def test_outcome_map_drops_unscored_rows(self):
        rows = make_results(FULL, count=5, success_rate=1.0)
        rows[2].error = "sandbox failure"
        assert len(outcome_map(rows)) == 4


class TestSampler:
    def test_a_run_draw_is_reproducible(self):
        first = draw_run(7, root=12345, hidden_count=20, ood_count=5)
        again = draw_run(7, root=12345, hidden_count=20, ood_count=5)
        assert first.hidden_seeds == again.hidden_seeds

    def test_different_runs_draw_different_instances(self):
        first = draw_run(7, root=12345, hidden_count=40, ood_count=5)
        second = draw_run(8, root=12345, hidden_count=40, ood_count=5)
        assert set(first.hidden_seeds).isdisjoint(second.hidden_seeds)

    def test_a_different_root_gives_a_different_draw(self):
        first = draw_run(7, root=1, hidden_count=40, ood_count=5)
        second = draw_run(7, root=2, hidden_count=40, ood_count=5)
        assert set(first.hidden_seeds).isdisjoint(second.hidden_seeds)

    def test_hidden_and_ood_draws_do_not_overlap(self):
        sample = draw_run(3, root=999, hidden_count=50, ood_count=50)
        assert set(sample.hidden_seeds).isdisjoint(sample.ood_seeds)

    def test_seeds_are_distinct(self):
        sample = draw_run(1, root=42, hidden_count=100, ood_count=30)
        assert len(set(sample.hidden_seeds)) == 100

    def test_common_instances_exclude_anything_either_side_failed(self):
        left = make_results(FULL, count=5)
        right = make_results(FULL, count=5)
        left[1].error = "sandbox failure"
        right[3].error = "sandbox failure"

        shared = common_instance_ids(left, right)
        assert len(shared) == 3
        assert "hidden-000001" not in shared and "hidden-000003" not in shared


class TestAnEmptyThroneIsFilledRatherThanAssumed:
    """The first throne, and what a run pays before anybody holds one.

    Two failures are held apart here. Treating an empty throne as a champion
    with grade zero pays the leader of any field at all. Treating a missing
    grade as an empty throne pays a field that cleared nothing, every run,
    forever.
    """

    FIELD = [(7, "5A", 0.50), (9, "5B", 0.40), (3, "5C", 0.30)]

    def test_an_empty_throne_ranks_the_field_as_it_stands(self):
        vector = champion_ladder(self.FIELD, run_id=1, block=1, champion_grade=None)

        assert vector.champion_hotkey == "5A"
        paid = {e.uid: e.weight for e in vector.entries if e.role != "burn"}
        assert set(paid) == {7, 9, 3}

    def test_the_leader_of_an_empty_throne_takes_the_same_share_as_a_champion(self):
        """Filling the throne is not a lesser prize than holding it."""
        first = champion_ladder(self.FIELD, run_id=1, block=1, champion_grade=None)
        defended = champion_ladder(
            [(7, "5A", 0.90), (9, "5B", 0.80), (3, "5C", 0.70)],
            run_id=2,
            block=1,
            champion_grade=0.50,
        )

        lead = lambda v: next(e.weight for e in v.entries if e.role == "champion")  # noqa: E731
        assert lead(first) == pytest.approx(lead(defended))

    def test_a_field_that_clears_nothing_pays_nobody(self):
        """Empty means nothing cleared the hard gates, which is the only way a
        run burns entirely now. A field that merely trails the incumbent is
        paid — the throne no longer decides who earns."""
        vector = champion_ladder([], run_id=2, block=1, champion_grade=0.50)

        assert vector.champion_hotkey is None
        assert [e.role for e in vector.entries] == ["burn"]
        assert vector.entries[0].weight == pytest.approx(1.0)

    def test_a_field_that_trails_the_incumbent_is_paid(self):
        vector = champion_ladder(self.FIELD, run_id=2, block=1, champion_grade=0.90)

        assert vector.champion_hotkey == "5A"
        assert {e.uid for e in vector.entries if e.role != "burn"} == {7, 9, 3}
