"""The logic workflow, tested on the properties the engine depends on.

Not a test that it scores well — that is a measurement, and it lives in
``experiments/``. These pin the contract the engine assumes of any workflow, on
the second implementation of it, which is the first time those assumptions are
load-bearing rather than descriptive.
"""

from __future__ import annotations

import pytest

from capability_subnet.workflows import get_workflow

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def logic():
    return get_workflow("lora_merger_logic_v1")


class TestItSatisfiesTheWorkflowContract:
    def test_it_is_discoverable_alongside_the_builtin(self):
        from capability_subnet.workflows import available_workflows

        assert "lora_merger_logic_v1" in available_workflows()

    def test_it_declares_every_field_the_engine_reads(self, logic):
        assert logic.workflow_id == "lora_merger_logic_v1"
        assert logic.critical_axes and logic.stages
        assert set(logic.stage_thresholds) == set(logic.stages)
        assert callable(logic.generate_instance)
        assert callable(logic.run_instance)
        assert logic.publicly_verifiable

    def test_it_owns_how_a_package_is_asked(self, logic):
        """The V1 workflow drives a twelve-turn tool loop; this one asks once.
        Hardcoding either into the evaluator would have made the other
        unpluggable, which is what the entry-point mechanism promises."""
        builtin = get_workflow("industrial_maintenance_de_v1")
        assert logic.run_instance is not builtin.run_instance
        assert logic.tool_schemas == []


class TestInstancesAreReproducibleFromTheirSeed:
    """The property replay rests on: an auditor holding a published seed
    regenerates exactly the problem the candidate faced."""

    def test_the_same_seed_yields_the_same_problem(self, logic):
        a, b = logic.generate_instance(4242), logic.generate_instance(4242)
        assert (a.task, a.question, a.answer) == (b.task, b.question, b.answer)

    def test_different_seeds_yield_different_problems(self, logic):
        questions = {logic.generate_instance(s).question for s in range(100, 120)}
        assert len(questions) > 15

    def test_the_draw_is_spread_across_families(self, logic):
        """Stratified, so a window cannot be dominated by whichever family the
        corpus happens to be densest in — or a miner happened to tune on."""
        families = {logic.generate_instance(s).task for s in range(500, 600)}
        assert len(families) >= 5

    def test_every_instance_carries_an_exact_answer(self, logic):
        for seed in range(200, 240):
            assert logic.generate_instance(seed).answer.strip()


class TestScoringIsExactAndUnjudged:
    def test_the_expected_answer_scores_correct_in_any_requested_wrapper(self, logic):
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import answered_correctly

        instance = logic.generate_instance(4242)
        for wrap in (
            lambda a: f"reasoning\n```python\n{a}\n```",
            lambda a: f"```json\n{a}\n```",
            lambda a: f"The answer is \\boxed{{{a}}}",
            lambda a: f"[[{a}]]",
            lambda a: a,
        ):
            assert answered_correctly(wrap(instance.answer), instance.answer)

    def test_a_wrong_answer_scores_wrong(self, logic):
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import answered_correctly

        instance = logic.generate_instance(4242)
        assert not answered_correctly("```python\nWRONG\n```", instance.answer)
        assert not answered_correctly(None, instance.answer)

    def test_format_compliance_is_scored_apart_from_correctness(self, logic):
        """A package that is right but can no longer follow an output
        instruction has failed differently from one that is wrong — and losing
        compliance is the commonest way an over-aggressive merge goes bad."""
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import (
            answered_correctly,
            used_requested_format,
        )

        instance = logic.generate_instance(4242)
        bare = instance.answer
        assert answered_correctly(bare, instance.answer)
        assert not used_requested_format(bare)
        assert used_requested_format(f"```python\n{bare}\n```")


class TestTheRunnerSeparatesOutageFromFailure:
    """Scoring an unreachable endpoint as a wrong answer would charge a miner
    for the operator's outage."""

    @staticmethod
    def _client(reply=None, error=None):
        from capability_subnet.sandbox.model_client import ModelReply

        class _Client:
            def complete(self, messages, tools, *, seed, max_tokens):
                assert tools == [], "this workflow offers no tools"
                return ModelReply(content=reply, tool_calls=[], error=error, output_tokens=7)

        return _Client()

    def test_a_correct_answer_is_a_completed_instance(self, logic):
        instance = logic.generate_instance(4242)
        outcome = logic.run_instance(instance, self._client(reply=instance.answer))
        assert outcome.result.end_to_end_success
        assert outcome.result.error is None
        assert outcome.result.turns_used == 1

    def test_a_wrong_answer_is_a_scored_failure_not_an_error(self, logic):
        outcome = logic.run_instance(logic.generate_instance(4242), self._client(reply="nope"))
        assert not outcome.result.end_to_end_success
        assert outcome.result.error is None
        assert outcome.result.is_valid_sample

    def test_an_unreachable_endpoint_is_excluded_from_scoring(self, logic):
        outcome = logic.run_instance(
            logic.generate_instance(4242), self._client(error="endpoint unavailable")
        )
        assert outcome.result.error
        assert not outcome.result.is_valid_sample


class TestTheContractStatesItsOwnLimits:
    def test_it_publishes_that_the_corpus_is_visible(self, logic):
        """A contract that published only the favourable properties would be
        marketing. A miner needs the weakness to decide whether to compete."""
        note = logic.build_contract()["corpus"]["note"]
        assert "public" in note and "not the items themselves" in note

    def test_it_publishes_that_difficulty_labels_overstate_the_regime(self, logic):
        note = logic.build_contract()["harness"]["note"]
        assert "pass@16" in note and "lower absolute scores" in note

    def test_it_publishes_that_no_model_judges_a_result(self, logic):
        assert "No language model judges" in logic.build_contract()["scoring"]["note"]
