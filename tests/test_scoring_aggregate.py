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
