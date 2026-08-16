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
    latency_efficiency,
    percentile,
    stage_balance,
    valid_rows,
)
from capability_subnet.scoring.bootstrap import (
    outcome_map,
    paired_bootstrap,
    paired_differences,
)
from capability_subnet.scoring.sampler import common_instance_ids, draw_window
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
            efficiency=EfficiencyInputs(artifact_bytes=0, peak_vram_gb=0.0),
        )
        assert scores.retention == 0.5

    def test_latency_efficiency_is_relative_and_capped(self):
        assert latency_efficiency(10.0, 20.0) == 1.0  # faster than the reference
        assert latency_efficiency(20.0, 10.0) == pytest.approx(0.5)
        assert latency_efficiency(0.0, 10.0) == 0.0

    def test_artifact_efficiency_rewards_headroom_below_the_gate(self):
        assert artifact_efficiency(0, 0.0) == pytest.approx(1.0)
        assert artifact_efficiency(C.MAX_ARTIFACT_BYTES, C.MAX_PEAK_VRAM_GB) == 0.0
        assert artifact_efficiency(
            C.MAX_ARTIFACT_BYTES // 2, C.MAX_PEAK_VRAM_GB / 2
        ) == pytest.approx(0.5)

    def test_percentile_uses_nearest_rank(self):
        # An interpolated percentile would report a duration that never occurred,
        # and the latency gate is a promise about an observed run.
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
            efficiency=EfficiencyInputs(artifact_bytes=0, peak_vram_gb=0.0, reference_seconds=5.0),
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
            efficiency=EfficiencyInputs(
                artifact_bytes=0, peak_vram_gb=0.0, reference_seconds=100.0
            ),
        )
        costly_but_right = aggregate_scores(
            make_results(FULL, count=20, success_rate=1.0, seconds=30.0),
            [],
            STAGES,
            retention=1.0,
            efficiency=EfficiencyInputs(
                artifact_bytes=C.MAX_ARTIFACT_BYTES,
                peak_vram_gb=C.MAX_PEAK_VRAM_GB,
                reference_seconds=1.0,
            ),
        )
        assert costly_but_right.qualified_score > efficient_but_wrong.qualified_score


class TestGates:
    def test_artifact_size_gate(self):
        assert gates.gate_artifact_size(C.MAX_ARTIFACT_BYTES).passed
        assert not gates.gate_artifact_size(C.MAX_ARTIFACT_BYTES + 1).passed

    def test_latency_gate_uses_the_worst_realistic_case(self):
        fast = make_results(FULL, count=20, seconds=5.0)
        slow = make_results(FULL, count=20, seconds=45.0)

        assert gates.gate_latency(fast).passed
        assert not gates.gate_latency(slow).passed

    def test_latency_gate_fails_closed_with_no_measurements(self):
        assert not gates.gate_latency([]).passed

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
            gates.gate_base_retention(0.5),
        ]
        passed, detail = gates.summarise(verdicts)

        assert passed is False
        assert "artifact_size" in detail and "base_retention" in detail


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
    def test_a_window_draw_is_reproducible(self):
        first = draw_window(7, root=12345, hidden_count=20, ood_count=5)
        again = draw_window(7, root=12345, hidden_count=20, ood_count=5)
        assert first.hidden_seeds == again.hidden_seeds

    def test_different_windows_draw_different_instances(self):
        first = draw_window(7, root=12345, hidden_count=40, ood_count=5)
        second = draw_window(8, root=12345, hidden_count=40, ood_count=5)
        assert set(first.hidden_seeds).isdisjoint(second.hidden_seeds)

    def test_a_different_root_gives_a_different_draw(self):
        first = draw_window(7, root=1, hidden_count=40, ood_count=5)
        second = draw_window(7, root=2, hidden_count=40, ood_count=5)
        assert set(first.hidden_seeds).isdisjoint(second.hidden_seeds)

    def test_hidden_and_ood_draws_do_not_overlap(self):
        sample = draw_window(3, root=999, hidden_count=50, ood_count=50)
        assert set(sample.hidden_seeds).isdisjoint(sample.ood_seeds)

    def test_seeds_are_distinct(self):
        sample = draw_window(1, root=42, hidden_count=100, ood_count=30)
        assert len(set(sample.hidden_seeds)) == 100

    def test_common_instances_exclude_anything_either_side_failed(self):
        left = make_results(FULL, count=5)
        right = make_results(FULL, count=5)
        left[1].error = "sandbox failure"
        right[3].error = "sandbox failure"

        shared = common_instance_ids(left, right)
        assert len(shared) == 3
        assert "hidden-000001" not in shared and "hidden-000003" not in shared


class TestAnEmptyThroneDoesNotPayTheRunnersUp:
    """A leaderless window pays less than a crowned one, and pays something.

    Half of it burns, and the best measured package leads what remains on the
    same terms a champion leads the whole — so its best miner takes roughly half
    what a champion would. Two failures are being held apart here.

    Handing the leader's share to the runners-up intact pays *more* in exactly
    the windows where the field was weakest. Observed on the testnet arena: with
    no champion a single contributor took 0.80 of the window instead of 0.36.

    Burning all of it pays nothing in every window before the first crown, which
    is the state a launch begins in and can hold for as long as the queue takes
    to work through.
    """

    @staticmethod
    def _champion():
        from capability_subnet.common.schemas import ChampionRecord

        return ChampionRecord(
            candidate_id="5Champ", hotkey="5Champ", uid=7, recipe_sha256="sha256:" + "0" * 64
        )

    @staticmethod
    def _vector(champion):
        from capability_subnet.scoring.weight_vector import graded_contribution

        return graded_contribution(
            workflow_id="w",
            window_id=1,
            block=10,
            spec_version=1,
            champion=champion,
            contributors=[(1, "5A", 1.0)],
            tail=[],
        )

    def test_with_no_champion_half_burns_and_the_leader_takes_the_rest(self):
        vector = self._vector(None)
        by_role = {e.role: e.weight for e in vector.entries}

        leaderless = 1.0 - C.NO_CHAMPION_BURN_SHARE
        leader = leaderless * C.CHAMPION_BASE_SHARE
        # One contributor, so there is nobody to share the runner-up pool with
        # and it burns alongside the half.
        assert by_role.get("contributor", 0) == pytest.approx(leader, abs=1e-6)
        assert by_role.get("burn", 0) == pytest.approx(1.0 - leader, abs=1e-6)

    def test_a_leaderless_window_pays_its_best_miner_less_than_a_crown(self):
        """The throne has to stay worth taking."""
        leaderless = {e.role: e.weight for e in self._vector(None).entries}
        crowned = {e.role: e.weight for e in self._vector(self._champion()).entries}
        assert leaderless.get("contributor", 0) < crowned.get("champion", 0)

    def test_the_runners_up_share_what_the_leader_does_not_take(self):
        from capability_subnet.scoring.weight_vector import graded_contribution

        vector = graded_contribution(
            workflow_id="w",
            window_id=1,
            block=10,
            spec_version=1,
            champion=None,
            contributors=[(1, "5A", 1.0), (2, "5B", 0.5), (3, "5C", 0.25)],
            tail=[],
        )
        weights = sorted(
            (e.weight for e in vector.entries if e.role == "contributor"), reverse=True
        )
        assert len(weights) == 3
        leaderless = 1.0 - C.NO_CHAMPION_BURN_SHARE
        assert weights[0] == pytest.approx(leaderless * C.CHAMPION_BASE_SHARE, abs=1e-6)
        assert sum(weights[1:]) == pytest.approx(
            leaderless * (1.0 - C.CHAMPION_BASE_SHARE), abs=1e-6
        )
        assert sum(e.weight for e in vector.entries) == pytest.approx(1.0, abs=1e-9)

    def test_a_real_champion_is_still_paid_its_share(self):
        from capability_subnet.common.schemas import ChampionRecord

        champion = ChampionRecord(
            candidate_id="5Champ", hotkey="5Champ", uid=7, recipe_sha256="sha256:" + "0" * 64
        )
        vector = self._vector(champion)
        by_role = {e.role: e.weight for e in vector.entries}
        assert by_role.get("champion", 0) == pytest.approx(C.CHAMPION_BASE_SHARE, abs=1e-6)
        assert by_role.get("burn", 0) == pytest.approx(0.0, abs=1e-6)
