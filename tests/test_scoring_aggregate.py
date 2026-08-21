def _single_axis_row(stage: str, score: float, seed: int = 1):
    from capability_subnet.common.schemas import InstanceResult, StageResult

    return InstanceResult(
        instance_id=f"i{seed}",
        instance_seed=seed,
        split="hidden",
        turns_used=1,
        stages={stage: StageResult(stage=stage, score=score, passed=score >= 1.0)},
    )


def test_an_axis_is_averaged_over_the_rows_that_scored_it():
    """Not over every row.

    A workflow whose instances each exercise one axis put about a twelfth of the
    draw on each. Dividing by the whole draw made every mean about twelve times
    too small, which put every axis under any absolute floor and made the arena
    unable to crown anyone.
    """
    from capability_subnet.scoring.aggregate import per_stage_means

    rows = [_single_axis_row("cipher", 1.0, 1), _single_axis_row("zebra_puzzle", 1.0, 2)]
    means = per_stage_means(rows, ("cipher", "zebra_puzzle"))

    assert means["cipher"] == 1.0
    assert means["zebra_puzzle"] == 1.0


def test_an_unsampled_axis_is_absent_rather_than_zero():
    from capability_subnet.scoring.aggregate import per_stage_means, per_stage_sample_counts

    rows = [_single_axis_row("cipher", 1.0)]
    stages = ("cipher", "never_drawn")

    assert "never_drawn" not in per_stage_means(rows, stages)
    assert per_stage_sample_counts(rows, stages)["never_drawn"] == 0


def test_stage_floors_ignores_an_axis_the_draw_never_sampled():
    from capability_subnet.common.schemas import CandidateScores
    from capability_subnet.scoring.gates import gate_stage_floors

    scores = CandidateScores(per_stage_means={"cipher": 1.0}, per_stage_samples={"cipher": 6})
    verdict = gate_stage_floors(scores, {"cipher": 1.0, "never_drawn": 1.0})

    assert verdict.passed, verdict.detail


def test_stage_floors_still_fails_an_axis_that_was_measured_and_poor():
    from capability_subnet.common.schemas import CandidateScores
    from capability_subnet.scoring.gates import gate_stage_floors

    scores = CandidateScores(per_stage_means={"cipher": 0.2}, per_stage_samples={"cipher": 6})

    assert not gate_stage_floors(scores, {"cipher": 1.0}).passed


class TestTheFloorGateSeparatesBrokenFromMerelyHard:
    """Measured on the real arena, not invented.

    The unmerged base model answers 18.8% of instances and is under 0.5 on nine
    axes of eleven; the equal-weight linear merge of twelve adapters emits
    repeated fragments and whitespace. A floor derived as half the pass threshold
    terminated both, so nothing could ever be crowned and every run burned its
    whole emission.
    """

    #: reference:base_model, run 5150 — coherent, and hard problems are hard.
    BASE = {
        "arrow_maze": 0.000,
        "boolean_expressions": 0.333,
        "cipher": 0.200,
        "code_execution": 0.214,
        "format_compliance": 0.771,
        "object_counting": 0.000,
        "time_sequence": 0.000,
        "web_of_lies": 0.000,
        "word_sorting": 0.600,
        "word_sorting_mistake": 0.200,
        "zebra_puzzle": 0.000,
    }
    COUNTS = {
        "arrow_maze": 8,
        "boolean_expressions": 3,
        "cipher": 5,
        "code_execution": 14,
        "format_compliance": 48,
        "object_counting": 1,
        "time_sequence": 1,
        "web_of_lies": 2,
        "word_sorting": 5,
        "word_sorting_mistake": 5,
        "zebra_puzzle": 4,
    }

    @staticmethod
    def _verdict(means):
        from capability_subnet.common.schemas import CandidateScores
        from capability_subnet.scoring.gates import gate_stage_floors
        from capability_subnet.workflows import get_workflow

        floors = get_workflow("lora_merger_logic_v1").stage_floors
        scores = CandidateScores(
            per_stage_means=means,
            per_stage_samples=TestTheFloorGateSeparatesBrokenFromMerelyHard.COUNTS,
        )
        return gate_stage_floors(scores, floors, min_samples=3)

    def test_a_working_package_survives_hard_problems(self):
        assert self._verdict(self.BASE).passed, "the strongest package must not be terminated"

    def test_a_package_that_stopped_producing_usable_output_is_terminated(self):
        broken = dict.fromkeys(self.BASE, 0.0)
        verdict = self._verdict(broken)
        assert not verdict.passed
        assert "format_compliance" in verdict.detail

    def test_an_axis_with_too_few_samples_cannot_terminate_a_candidate(self):
        """A mean over one instance is 0.0 or 1.0 and nothing between."""
        from capability_subnet.common.schemas import CandidateScores
        from capability_subnet.scoring.gates import gate_stage_floors

        scores = CandidateScores(
            per_stage_means={"format_compliance": 0.0},
            per_stage_samples={"format_compliance": 1},
        )
        assert gate_stage_floors(scores, {"format_compliance": 0.5}, min_samples=3).passed
