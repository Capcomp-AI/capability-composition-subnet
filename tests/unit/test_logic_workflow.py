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


@pytest.fixture(scope="module")
def exact_seed(logic):
    """A seed that draws an exact-match logic item.

    The two corpora are scored differently, so a test about one of them has to
    say which it wants. Searching rather than hardcoding: the split is a fixed
    fraction of the draw, not a fixed set of seeds, so a hardcoded seed would
    silently change meaning if the fraction moved.
    """
    return next(s for s in range(200, 400) if not logic.generate_instance(s).is_code)


@pytest.fixture(scope="module")
def code_seed(logic):
    """A seed that draws an execution-scored programming problem."""
    return next(s for s in range(200, 400) if logic.generate_instance(s).is_code)


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

    def test_every_axis_a_draw_can_produce_is_declared(self, logic):
        """An axis the scorer emits but the workflow never declared would get no
        threshold, and a stage nothing is ever drawn for would sit permanently
        unpopulated — either way the aggregate is wrong."""
        from capability_subnet.workflows.lora_merger_logic_v1 import sources as S

        assert set(logic.stages) == {*S.LOGIC_FAMILIES, S.CODE_FAMILY, "format_compliance"}
        drawn = {logic.generate_instance(s).task for s in range(600)}
        assert drawn <= set(logic.stages)
        assert S.CODE_FAMILY in drawn

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
        assert (a.task, a.question, a.answer, a.cases) == (b.task, b.question, b.answer, b.cases)

    def test_different_seeds_yield_different_problems(self, logic):
        questions = {logic.generate_instance(s).question for s in range(100, 120)}
        assert len(questions) > 15

    def test_the_draw_is_spread_across_families(self, logic):
        """Stratified, so a window cannot be dominated by whichever family the
        corpus happens to be densest in — or a miner happened to tune on."""
        families = {logic.generate_instance(s).task for s in range(500, 600)}
        assert len(families) >= 5

    def test_every_instance_carries_something_to_score_against(self, logic):
        """Either an exact answer or test cases, never neither. An instance with
        no ground truth would be scored wrong for every package."""
        for seed in range(200, 260):
            instance = logic.generate_instance(seed)
            assert instance.answer.strip() or instance.cases

    def test_both_corpora_are_drawn_from(self, logic):
        """A window that reached only one corpus would quietly drop an axis and
        the execution guarantee with it."""
        sources = [logic.generate_instance(s).source for s in range(400)]
        share = sources.count("code") / len(sources)
        assert set(sources) == {"logic", "code"}
        assert 0.15 < share < 0.35, f"code share {share:.2f} strays from the declared fraction"

    def test_a_code_instance_carries_its_cases_rather_than_refetching_them(self, logic, code_seed):
        """Replay scores against exactly the cases the candidate faced, so they
        travel with the instance instead of being looked up at scoring time."""
        instance = logic.generate_instance(code_seed)
        assert instance.is_code and instance.cases
        assert all(case.expected_stdout for case in instance.cases)
        assert instance.task == "code_execution"


class TestScoringIsExactAndUnjudged:
    def test_the_expected_answer_scores_correct_in_any_requested_wrapper(self, logic, exact_seed):
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import answered_correctly

        instance = logic.generate_instance(exact_seed)
        for wrap in (
            lambda a: f"reasoning\n```python\n{a}\n```",
            lambda a: f"```json\n{a}\n```",
            lambda a: f"The answer is \\boxed{{{a}}}",
            lambda a: f"[[{a}]]",
            lambda a: a,
        ):
            assert answered_correctly(wrap(instance.answer), instance.answer)

    def test_a_wrong_answer_scores_wrong(self, logic, exact_seed):
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import answered_correctly

        instance = logic.generate_instance(exact_seed)
        assert not answered_correctly("```python\nWRONG\n```", instance.answer)
        assert not answered_correctly(None, instance.answer)

    def test_format_compliance_is_scored_apart_from_correctness(self):
        """A package that is right but can no longer follow an output
        instruction has failed differently from one that is wrong — and losing
        compliance is the commonest way an over-aggressive merge goes bad.

        Asserted on a controlled answer rather than a drawn one, because some
        families answer with a nested list that is itself bracket-wrapped; see
        the following test."""
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import (
            answered_correctly,
            used_requested_format,
        )

        assert answered_correctly("42", "42")
        assert not used_requested_format("42")
        for wrapped in ("```python\n42\n```", "\\boxed{42}", "[[42]]"):
            assert used_requested_format(wrapped)
            assert answered_correctly(wrapped, "42")

    def test_compliance_carries_little_information_for_list_answers(self, logic):
        """A limitation worth pinning rather than discovering later.

        Families whose answer is a nested list — ``arrow_maze``, the word-sorting
        pair — have answers of the form ``[[...]]``, which is also the wrapper
        their prompt requests. Any package that states such an answer at all
        scores compliant, so on those families the axis is close to redundant with
        correctness rather than independent of it. It stays informative on the
        scalar families and on the code corpus, which is why it remains an axis.
        """
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import used_requested_format

        bare_compliant = {
            logic.generate_instance(s).task
            for s in range(400)
            if not logic.generate_instance(s).is_code
            and used_requested_format(logic.generate_instance(s).answer)
        }
        assert bare_compliant, "expected the list-answer families to show this overlap"
        assert bare_compliant < set(logic.stages), "but it must not be every family"


class TestCodeInstancesAreScoredByRunningTheProgram:
    """The stronger of the two regimes: exact match asks whether the answer looks
    right, execution asks whether the code works. A memorised problem does not
    help unless the program actually runs."""

    @staticmethod
    def _trace(reply):
        from capability_subnet.workflows.lora_merger_logic_v1.runner import LogicTrace

        trace = LogicTrace()
        trace.reply = reply
        return trace

    @staticmethod
    def _cases(*pairs):
        from capability_subnet.workflows.lora_merger_logic_v1.dataset import TestCase

        return tuple(TestCase(stdin=i, expected_stdout=o) for i, o in pairs)

    def test_a_working_program_passes_every_case(self):
        from capability_subnet.sandbox.python_runner import PythonRunner
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import run_test_cases

        passed, total, _ = run_test_cases(
            "print(int(input()) * 2)",
            self._cases(("21\n", "42\n"), ("5\n", "10\n")),
            PythonRunner(),
        )
        assert (passed, total) == (2, 2)

    def test_a_program_wrong_on_one_case_is_partially_credited(self):
        """Every case runs even after a failure: right on the sample and wrong on
        an edge case is different information from not running at all."""
        from capability_subnet.sandbox.python_runner import PythonRunner
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import run_test_cases

        passed, total, detail = run_test_cases(
            "n = int(input())\nprint(42 if n == 21 else 0)",
            self._cases(("21\n", "42\n"), ("5\n", "10\n")),
            PythonRunner(),
        )
        assert (passed, total) == (1, 2)
        assert "case 2" in detail

    def test_a_crashing_program_is_contained_and_scored_zero(self):
        from capability_subnet.sandbox.python_runner import PythonRunner
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import run_test_cases

        passed, total, detail = run_test_cases(
            "raise SystemError('boom')", self._cases(("1\n", "1\n")), PythonRunner()
        )
        assert (passed, total) == (0, 1)
        assert detail

    def test_no_program_is_a_scored_failure_not_an_execution(self):
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import run_test_cases

        class _Forbidden:
            def run_program(self, *_a, **_k):
                raise AssertionError("nothing should be executed when no program was submitted")

        passed, total, detail = run_test_cases("", self._cases(("1\n", "1\n")), _Forbidden())
        assert (passed, total) == (0, 1)
        assert "no program" in detail

    def test_an_exhausted_program_stops_the_remaining_cases(self):
        """A cost bound, not a scoring choice.

        The worst problem in the corpus carries hundreds of cases and the ceiling
        is twenty seconds each, so a hanging program could hold a validator for
        hours against a fifteen-second budget. Passing needs every case, so
        stopping cannot change the verdict.
        """
        from capability_subnet.sandbox.python_runner import EXHAUSTED
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import run_test_cases

        class _Hangs:
            def __init__(self):
                self.calls = 0

            def run_program(self, source, stdin):
                self.calls += 1
                return None, f"{EXHAUSTED}: exceeded the time limit"

        runner = _Hangs()
        passed, total, detail = run_test_cases(
            "while True: pass", self._cases(*[("", "")] * 20), runner
        )
        assert (passed, total) == (0, 20)
        assert runner.calls == 1, "a program that exhausts its ceiling is not retried 20 times"
        assert "stopped after 1 case" in detail

    def test_a_crashing_program_still_runs_every_case(self):
        """The complement: a crash is cheap and the next case may differ, so the
        partial-credit argument still applies to it."""
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import run_test_cases

        class _Crashes:
            def __init__(self):
                self.calls = 0

            def run_program(self, source, stdin):
                self.calls += 1
                return None, "exited 1: ValueError"

        runner = _Crashes()
        run_test_cases("raise ValueError", self._cases(*[("", "")] * 6), runner)
        assert runner.calls == 6

    def test_no_problem_carries_more_cases_than_the_cap(self):
        """Applied at load time so the retained cases travel with the instance and
        a replay scores the same prefix."""
        from capability_subnet.workflows.lora_merger_logic_v1 import dataset
        from capability_subnet.workflows.lora_merger_logic_v1 import sources as S

        counts = [len(item.cases) for item in dataset.load_code()]
        assert counts and max(counts) <= S.MAX_CASES_PER_PROBLEM
        assert min(counts) >= 1, "a code item with no cases could never be scored"

    def test_the_program_is_taken_from_the_last_fenced_block(self):
        """Replies commonly restate the prompt's own example before answering."""
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import extract_program

        reply = "the example was:\n```python\nWRONG\n```\nmine:\n```python\nRIGHT\n```"
        assert extract_program(reply) == "RIGHT"
        assert extract_program("bare code") == "bare code"
        assert extract_program(None) == ""

    def test_only_a_solved_instance_counts_as_success(self, logic, code_seed):
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import score_instance

        instance = logic.generate_instance(code_seed)
        n = len(instance.cases)

        class _Runner:
            def __init__(self, wrong_after):
                self.calls = 0
                self.wrong_after = wrong_after

            def run_program(self, source, stdin):
                self.calls += 1
                if self.calls > self.wrong_after:
                    return "definitely-not-it", ""
                return instance.cases[self.calls - 1].expected_stdout, ""

        solved = score_instance(instance, self._trace("```python\npass\n```"), runner=_Runner(n))
        assert solved[instance.task].passed
        assert solved["format_compliance"].passed

        if n > 1:
            partial = score_instance(
                instance, self._trace("```python\npass\n```"), runner=_Runner(n - 1)
            )
            assert not partial[instance.task].passed

    def test_compliance_means_the_same_thing_on_both_corpora(self, logic, code_seed):
        """`bool(program)` would score every non-empty reply compliant, because
        extraction falls back to the whole reply — the axis would then mean one
        thing on logic items and another on code items."""
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import score_instance

        instance = logic.generate_instance(code_seed)

        class _Runner:
            def run_program(self, source, stdin):
                return "", ""

        unfenced = score_instance(instance, self._trace("here is my code: pass"), runner=_Runner())
        assert not unfenced["format_compliance"].passed


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

    def test_a_correct_answer_is_a_completed_instance(self, logic, exact_seed):
        instance = logic.generate_instance(exact_seed)
        outcome = logic.run_instance(instance, self._client(reply=instance.answer))
        assert outcome.result.end_to_end_success
        assert outcome.result.error is None
        assert outcome.result.turns_used == 1

    def test_a_wrong_answer_is_a_scored_failure_not_an_error(self, logic, exact_seed):
        outcome = logic.run_instance(
            logic.generate_instance(exact_seed), self._client(reply="nope")
        )
        assert not outcome.result.end_to_end_success
        assert outcome.result.error is None
        assert outcome.result.is_valid_sample

    def test_an_unreachable_endpoint_is_excluded_from_scoring(self, logic, exact_seed):
        outcome = logic.run_instance(
            logic.generate_instance(exact_seed), self._client(error="endpoint unavailable")
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
