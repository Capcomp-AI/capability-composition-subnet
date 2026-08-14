"""A validator measuring a candidate on its own hardware.

Reconstruction here is real — the tiny pool goes through the same merge engine a
validator would run on an 8B — while serving is stubbed, because what is being
tested is that a validator can carry the evaluation itself: reconstruct any of
the seven methods, run exactly the instances it was assigned, and produce the
per-instance record its peers compare it against.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import InstanceResult, StageResult
from capability_subnet.sandbox.model_client import ModelReply
from capability_subnet.validator.assignment import assign
from capability_subnet.validator.evaluator import (
    core_results,
    evaluate_candidate,
)

BEACON = "0x" + "ab" * 32

ALL_METHODS = (
    C.MERGE_LINEAR,
    C.MERGE_SVD,
    C.MERGE_CAT_SVD,
    C.MERGE_TIES_SVD,
    C.MERGE_DARE_TIES_SVD,
    C.MERGE_DARE_LINEAR_SVD,
    C.MERGE_MAGNITUDE_PRUNE_SVD,
)

# Taken from the contract rather than restated: density and sign election apply
# to different sets of methods, and a test that guessed would pass for the wrong
# reason the moment either set changed.
NEEDS_DENSITY = C.DENSITY_METHODS
NEEDS_SIGN = {C.MERGE_TIES_SVD, C.MERGE_DARE_TIES_SVD}


@dataclass
class _Outcome:
    result: InstanceResult


@dataclass
class _Instance:
    """The part of an instance the orchestrator and the runner read."""

    seed: int
    instance_id: str
    split: str = "hidden"


class _StubClient:
    """Stands in for a served endpoint.

    Only the retention probe reaches it — instances go through the stubbed
    runner — so a fixed well-formed reply is enough.
    """

    def complete(self, messages, tools, **k):
        return ModelReply(content="ok", tool_calls=[], input_tokens=5, output_tokens=5)


@pytest.fixture
def stub_workflow(workflow):
    """The real workflow with serving replaced by a deterministic verdict."""

    import copy

    flow = copy.copy(workflow)

    def generate_instance(seed, *, split="hidden"):
        return _Instance(seed=seed, instance_id=f"i-{seed}", split=split)

    def run_instance(instance, client, *, config=None):
        seed = instance.seed
        return _Outcome(
            InstanceResult(
                instance_id=f"i-{seed}",
                instance_seed=seed,
                # deterministic, seed-dependent, so two runs agree and the
                # per-instance record is meaningful
                end_to_end_success=(seed % 3 == 0),
                final_state_correct=(seed % 3 == 0),
                turns_used=1,
                output_tokens=10,
                input_tokens=10,
                wall_seconds=0.5,
                stages={s: StageResult(stage=s, score=1.0, passed=True) for s in flow.stages},
            )
        )

    object.__setattr__(flow, "generate_instance", generate_instance)
    object.__setattr__(flow, "run_instance", run_instance)
    return flow


def _evaluate(recipe, pool_dir, snapshot, stub_workflow, tmp_path, seeds=tuple(range(1, 41))):
    assignment = assign(seeds, hotkey="5AAA", beacon=BEACON)
    return (
        evaluate_candidate(
            recipe,
            _StubClient(),
            assignment=assignment,
            pool_dir=pool_dir,
            artifact_dir=tmp_path / "artifact",
            candidate_id="5AAA",
            snapshot=snapshot,
            workflow=stub_workflow,
        ),
        assignment,
    )


class TestAValidatorCanCarryTheEvaluation:
    def test_it_measures_exactly_what_it_was_assigned(
        self, recipe_factory, tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path
    ):
        result, assignment = _evaluate(
            recipe_factory(), tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path
        )
        assert result.usable, result.error
        assert set(result.per_instance) == set(assignment.seeds)

    def test_it_reconstructs_for_itself(
        self, recipe_factory, tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path
    ):
        """No operator hands it an artifact — it builds one."""
        result, _ = _evaluate(
            recipe_factory(), tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path
        )
        assert result.artifact_sha256.startswith("sha256:")
        assert result.artifact_bytes > 0

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_every_merge_method_is_supported(
        self, method, recipe_factory, tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path
    ):
        """All seven, not the reproducible subset. Restricting to methods that
        happen to be bitwise portable would delete the only family that has
        ever beaten the base model in this arena."""
        recipe = recipe_factory(
            combination_type=method,
            density=0.5 if method in NEEDS_DENSITY else None,
            sign_method="total" if method in NEEDS_SIGN else None,
        )
        result, _ = _evaluate(recipe, tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path)
        assert result.usable, f"{method}: {result.error}"
        assert result.artifact_sha256.startswith("sha256:")

    def test_two_runs_on_one_host_agree(
        self, recipe_factory, tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path
    ):
        recipe = recipe_factory()
        first, _ = _evaluate(recipe, tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path)
        second, _ = _evaluate(recipe, tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path)
        assert first.per_instance == second.per_instance
        assert first.artifact_sha256 == second.artifact_sha256


class TestFailureIsRecordedRatherThanRaised:
    def test_a_broken_reconstruction_does_not_take_the_validator_down(
        self, recipe_factory, tiny_snapshot, stub_workflow, tmp_path
    ):
        """One bad candidate must not stop a validator paying everyone else."""
        result, _ = _evaluate(
            recipe_factory(),
            tmp_path / "no-such-pool",
            tiny_snapshot,
            stub_workflow,
            tmp_path,
        )
        assert not result.usable
        assert "reconstruction failed" in result.error
        assert result.per_instance == {}

    def test_an_unusable_candidate_still_reports_its_recipe(
        self, recipe_factory, tiny_snapshot, stub_workflow, tmp_path
    ):
        recipe = recipe_factory()
        result, _ = _evaluate(recipe, tmp_path / "missing", tiny_snapshot, stub_workflow, tmp_path)
        assert result.recipe_sha256 == recipe.digest()


class TestOnlyTheCoreIsComparable:
    def test_the_tail_is_excluded_from_peer_comparison(
        self, recipe_factory, tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path
    ):
        """A peer holds no measurement of this validator's tail, so those
        seeds are not evidence about either of them."""
        result, assignment = _evaluate(
            recipe_factory(), tiny_pool_dir, tiny_snapshot, stub_workflow, tmp_path
        )
        shared = core_results(result, assignment)
        assert set(shared) == set(assignment.core)
        assert set(shared).isdisjoint(assignment.tail)
