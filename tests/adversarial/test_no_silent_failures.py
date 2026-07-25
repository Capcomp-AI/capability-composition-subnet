"""Silent failures and bypass routes.

Every test here corresponds to a defect that was found by audit rather than by a
failing test — which is exactly the point. A silent failure produces no symptom,
so nothing catches it except deliberately asserting the absence.

Two families:

* **Silent failures** — a component that fails without saying so, or degrades to
  a value that reads as success.
* **Bypasses** — a way for an actor to obtain an outcome the protocol says they
  should not get.
"""

from __future__ import annotations

import logging

import pytest

from capability_subnet.backend.evaluation import EvaluationOutput
from capability_subnet.backend.monitor.fetch import FetchError, LocalRecipeSource
from capability_subnet.backend.scorer import gates
from capability_subnet.common.hashing import sha256_bytes
from capability_subnet.common.schemas import EvaluationReport, GateVerdict
from capability_subnet.sandbox.limits import ExecutionLimits


class TestEmptyGateListsAreNotSuccess:
    """``all([])`` is True. Every place that checks gates must not inherit that."""

    def test_all_passed_rejects_an_empty_list(self):
        assert gates.all_passed([]) is False
        assert gates.all_passed([GateVerdict(name="x", passed=True)]) is True
        assert gates.all_passed([GateVerdict(name="x", passed=False)]) is False

    def test_an_ungated_evaluation_has_not_passed(self):
        # Reachable whenever evaluate() returns before the gates run.
        assert EvaluationOutput(candidate_id="c").gates_passed is False

    def test_an_ungated_report_has_not_passed(self):
        # This one decides graded-mode eligibility: a report counted as qualified
        # without a single gate verdict would be paid.
        report = EvaluationReport(
            window_id=1,
            evaluated_at_block=1,
            miner_hotkey="5A",
            candidate_id="5A",
            base_revision="r",
            source_snapshot_sha256="s",
            evaluator_image_digest="d",
            spec_version=1,
        )
        assert report.gates_passed is False

    def test_summarise_says_so_rather_than_claiming_success(self):
        passed, detail = gates.summarise([])
        assert passed is False
        assert "no gates" in detail


class TestUnmeasuredResourcesDoNotPass:
    """An unreadable measurement must not read as a comfortable one."""

    def test_an_unmeasured_peak_fails_where_a_gpu_is_expected(self):
        verdict = gates.gate_peak_vram(None, require_measurement=True)
        assert verdict.passed is False
        assert "could not be measured" in verdict.detail

    def test_an_unmeasured_peak_is_tolerated_only_when_explicitly_allowed(self):
        assert gates.gate_peak_vram(None, require_measurement=False).passed is True

    def test_a_real_measurement_is_still_judged_on_its_value(self):
        from capability_subnet.common import constants as C

        assert gates.gate_peak_vram(C.MAX_PEAK_VRAM_GB - 1).passed is True
        assert gates.gate_peak_vram(C.MAX_PEAK_VRAM_GB + 1).passed is False

    def test_the_reader_reports_absence_rather_than_zero(self):
        # Zero would sail through the gate. On a host with no CUDA device the
        # reader must say "unmeasured", not "used nothing".
        import torch

        from capability_subnet.backend.executor.serving import read_peak_vram_gb

        if torch.cuda.is_available():  # pragma: no cover - depends on host
            pytest.skip("this host has a GPU, so the counter is readable")
        assert read_peak_vram_gb() is None


class TestCommitOrderIsNeverGuessed:
    """Commit order assigns challenger roles, so a wrong block changes who runs."""

    def test_an_unreadable_block_is_skipped_rather_than_defaulted(self, caplog):
        from capability_subnet.common.chain import read_commitments
        from capability_subnet.common.commitments import encode_commitment

        payload = encode_commitment(
            "industrial_maintenance_de_v1", sha256_bytes(b"r"), "https://x.test/r.json"
        )

        class _Subtensor:
            def get_all_commitments(self, netuid):
                return {"5Broken": payload}

            def get_commitment_metadata(self, netuid, hotkey):
                raise RuntimeError("storage unavailable")

        with caplog.at_level(logging.WARNING):
            results = read_commitments(_Subtensor(), 1)

        # Not queued with a guessed block — which at zero would have sorted it to
        # the *front* of the queue, ahead of every legitimate commitment.
        assert results == []
        assert any("could not be read" in record.message for record in caplog.records)

    def test_a_malformed_block_value_is_also_skipped(self):
        from capability_subnet.common.chain import read_commitments
        from capability_subnet.common.commitments import encode_commitment

        payload = encode_commitment(
            "industrial_maintenance_de_v1", sha256_bytes(b"r"), "https://x.test/r.json"
        )

        class _Subtensor:
            def get_all_commitments(self, netuid):
                return {"5Odd": payload}

            def get_commitment_metadata(self, netuid, hotkey):
                return {"block": "not-a-number"}

        assert read_commitments(_Subtensor(), 1) == []

    def test_a_readable_block_is_queued_in_order(self):
        from capability_subnet.common.chain import read_commitments
        from capability_subnet.common.commitments import encode_commitment

        payloads = {
            "5Late": encode_commitment(
                "industrial_maintenance_de_v1", sha256_bytes(b"a"), "https://x.test/a.json"
            ),
            "5Early": encode_commitment(
                "industrial_maintenance_de_v1", sha256_bytes(b"b"), "https://x.test/b.json"
            ),
        }
        blocks = {"5Late": 900, "5Early": 100}

        class _Subtensor:
            def get_all_commitments(self, netuid):
                return payloads

            def get_commitment_metadata(self, netuid, hotkey):
                return {"block": blocks[hotkey]}

        results = read_commitments(_Subtensor(), 1)
        assert [c.hotkey for c in results] == ["5Early", "5Late"]


class TestAdmittedRecipesArePersisted:
    """A queue entry whose recipe cannot be loaded blocks the queue forever."""

    def test_admission_stores_the_verified_bytes(self, tiny_snapshot, store, tmp_path):
        from capability_subnet.backend.monitor.admission import admit_new_commitments
        from tests.adversarial.test_attacks import commitment, fetcher_for
        from tests.conftest import build_recipe

        recipe = build_recipe(tiny_snapshot)
        raw = recipe.canonical_bytes()
        recipe_store = LocalRecipeSource(tmp_path / "recipes")

        results = admit_new_commitments(
            [commitment("5Miner", recipe.digest(), 100)],
            snapshot=tiny_snapshot,
            store=store,
            registered_hotkeys={"5Miner"},
            current_block=200,
            fetcher=fetcher_for(raw),
            recipe_store=recipe_store,
        )

        assert results[0].admitted
        assert recipe_store.has(recipe.digest())

        # And it round-trips, so the engine can re-measure a champion later.
        assert recipe_store.fetch("local", recipe.digest()).raw == raw

    def test_a_recipe_that_cannot_be_stored_is_not_queued(self, tiny_snapshot, store, tmp_path):
        from capability_subnet.backend.monitor.admission import admit_new_commitments
        from tests.adversarial.test_attacks import commitment, fetcher_for
        from tests.conftest import build_recipe

        recipe = build_recipe(tiny_snapshot)

        class _BrokenStore:
            def store(self, raw, digest):
                raise OSError("disk full")

        results = admit_new_commitments(
            [commitment("5Miner", recipe.digest(), 100)],
            snapshot=tiny_snapshot,
            store=store,
            registered_hotkeys={"5Miner"},
            current_block=200,
            fetcher=fetcher_for(recipe.canonical_bytes()),
            recipe_store=_BrokenStore(),
        )

        # Not admitted, so it is retried on the next chain read rather than
        # sitting at the head of the queue as an entry nobody can evaluate.
        assert not results[0].admitted
        assert store.get_queue_entry("5Miner") is None

    def test_the_store_refuses_bytes_that_do_not_match_their_digest(self, tmp_path):
        recipe_store = LocalRecipeSource(tmp_path / "recipes")
        with pytest.raises(FetchError, match="refusing to store"):
            recipe_store.store(b"some other bytes", sha256_bytes(b"the real recipe"))

    def test_corruption_on_disk_is_refused_rather_than_scored(self, tmp_path):
        recipe_store = LocalRecipeSource(tmp_path / "recipes")
        digest = sha256_bytes(b"original")
        recipe_store.store(b"original", digest)

        path = recipe_store._path_for(digest)
        path.write_bytes(b"tampered")

        with pytest.raises(FetchError, match="does not hash"):
            recipe_store.fetch("local", digest)


class TestAgentLimitsCannotBeEvaded:
    """The turn budget is a scored quantity, not a formality."""

    def _run(self, instance, calls_per_turn: int, limits: ExecutionLimits):
        import json

        from capability_subnet.common.trace import ExecutionTrace
        from capability_subnet.sandbox.agent_runner import run_agent_loop
        from capability_subnet.sandbox.db_tool import open_database
        from capability_subnet.sandbox.model_client import ModelReply
        from capability_subnet.sandbox.python_runner import PythonRunner
        from capability_subnet.sandbox.tools import ToolBox

        class _Batching:
            """Emits many tool calls per reply, which is the evasion."""

            def complete(self, messages, tools, *, seed, max_tokens):
                return ModelReply(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"c{index}",
                            "type": "function",
                            "function": {
                                "name": "read_manual",
                                "arguments": json.dumps({"section_or_query": "3.1"}),
                            },
                        }
                        for index in range(calls_per_turn)
                    ],
                    output_tokens=10,
                )

        trace = ExecutionTrace(instance_id=instance.instance_id, instance_seed=instance.seed)
        database = open_database(instance.database)
        try:
            toolbox = ToolBox(instance, database, PythonRunner(limits), trace, limits)
            run_agent_loop(instance, _Batching(), toolbox, trace, limits=limits)
        finally:
            database.close()
        return trace

    def test_batching_tool_calls_does_not_multiply_the_budget(self, instance):
        limits = ExecutionLimits(max_turns=3, max_tool_calls_per_turn=2)
        trace = self._run(instance, calls_per_turn=50, limits=limits)

        # Without the per-turn cap this would be 150 calls for a 3-turn budget.
        assert len(trace.calls) <= limits.max_turns * limits.max_tool_calls_per_turn

    def test_the_total_cap_bounds_the_whole_instance(self, instance):
        limits = ExecutionLimits(max_turns=20, max_tool_calls_per_turn=5, max_tool_calls_total=7)
        trace = self._run(instance, calls_per_turn=5, limits=limits)
        assert len(trace.calls) <= 7


class TestACandidateCannotExcludeItsOwnInstance:
    """Excluded instances do not count against a candidate, so forcing an
    exclusion is worth real score. A tool crash must not grant one."""

    def test_a_tool_raising_on_candidate_arguments_is_a_tool_error(self, instance, monkeypatch):
        from capability_subnet.common.trace import ExecutionTrace
        from capability_subnet.sandbox.db_tool import open_database
        from capability_subnet.sandbox.python_runner import PythonRunner
        from capability_subnet.sandbox.tools import ToolBox

        trace = ExecutionTrace(instance_id=instance.instance_id, instance_seed=instance.seed)
        database = open_database(instance.database)
        try:
            toolbox = ToolBox(instance, database, PythonRunner(), trace)

            def _explode(_arguments):
                raise RecursionError("crafted input")

            monkeypatch.setattr(toolbox, "_read_manual", _explode)
            monkeypatch.setattr(toolbox, "_handlers", lambda: {"read_manual": toolbox._read_manual})

            call = toolbox.dispatch(1, "read_manual", {"section_or_query": "x"})
        finally:
            database.close()

        assert call.ok is False
        # The decisive assertion: the run stays scorable, so the candidate is
        # judged on this instance rather than having it dropped.
        assert trace.harness_error is None
        assert trace.is_scorable


class TestValidatorCannotSilentlySkipChecks:
    def test_an_unknown_window_length_is_reported_not_assumed_fresh(self):
        from capability_subnet.common.schemas import WeightEntry, WeightVector
        from capability_subnet.validator.client import validate_vector

        vector = WeightVector(
            window_id=1,
            computed_at_block=1,
            spec_version=1,
            entries=[WeightEntry(uid=1, hotkey="5A", weight=1.0)],
        )

        problems = validate_vector(
            vector,
            metagraph_size=4,
            hotkeys=["5Z", "5A", "5B", "5C"],
            current_window=None,
            max_stale_windows=3,
            burn_uid=0,
        )
        assert any(problem.code == "staleness_unknown" for problem in problems)

    def test_staleness_is_judged_against_the_engines_own_window(self):
        # A deployment with a short window rotates faster, so "three windows
        # behind" is a different amount of time. Judging it against a compiled-in
        # default would accept vectors long past their useful life.
        from capability_subnet.common.chain import window_id_for_block

        assert window_id_for_block(1000, 100) == 10
        assert window_id_for_block(1000, 500) == 2


class TestAnIncompleteBarCannotCrown:
    """A challenger must clear the strongest reference. If some references went
    unmeasured, the strongest of the rest may not be the strongest that exists —
    and the bar is quietly lower than the protocol promises."""

    def _state(self, **overrides):
        from capability_subnet.backend.engine_loop import WindowState

        state = WindowState(window_id=1)
        state.base_measured = True
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    def test_a_complete_reference_set_may_crown(self):
        complete, why = self._state().bar_is_complete()
        assert complete, why

    def test_an_unmeasured_reference_blocks_crowning(self):
        complete, why = self._state(
            missing_references={"reference:equal_ties_svd_merge": "endpoint down"}
        ).bar_is_complete()

        assert complete is False
        assert "equal_ties_svd_merge" in why

    def test_an_unmeasured_base_model_blocks_crowning(self):
        # Retention is relative to the base model, so without it the retention
        # gate would pass every candidate unconditionally.
        complete, why = self._state(base_measured=False).bar_is_complete()

        assert complete is False
        assert "base model" in why
