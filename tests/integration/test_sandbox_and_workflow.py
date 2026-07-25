"""The sandbox and the workflow it runs.

The claim under test is that the environment is *objectively judgeable*: it is
solvable by something that knows the answers, unsolvable by something that does
not, and each capability the workflow claims to measure is separately visible in
the scores.

The reference solver used here knows the ground truth by construction, which
makes it useless as a candidate and exactly right as a fixture — if it cannot
finish an instance, no candidate score from that instance would mean anything.
"""

from __future__ import annotations

import pytest

from capability_subnet.common.trace import ExecutionTrace
from capability_subnet.sandbox.db_tool import (
    SqliteMaintenanceDatabase,
    SqlRejected,
    validate_statement,
)
from capability_subnet.sandbox.limits import Deadline, ExecutionLimits, LimitExceeded
from capability_subnet.sandbox.orchestrator import SandboxConfig, run_instance
from capability_subnet.sandbox.python_runner import PythonRunner
from capability_subnet.sandbox.reference_solver import IMPAIRMENTS, ReferenceSolverClient
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import STAGES

FAST = SandboxConfig(limits=ExecutionLimits(python_timeout_seconds=10.0))


class TestEnvironmentIsSolvable:
    @pytest.mark.parametrize("seed", [1, 17, 4242, 99999])
    def test_hidden_instances_are_solvable(self, workflow, seed):
        instance = workflow.generate_instance(seed, split="hidden")
        result = run_instance(instance, ReferenceSolverClient(instance), config=FAST).result

        assert result.error is None
        assert result.end_to_end_success, {
            name: stage.detail for name, stage in result.stages.items() if not stage.passed
        }

    @pytest.mark.parametrize("seed", [3, 77, 12345])
    def test_out_of_distribution_instances_are_solvable(self, workflow, seed):
        # Mutations restate the problem; they do not make it unanswerable.
        instance = workflow.generate_instance(seed, split="ood")
        result = run_instance(instance, ReferenceSolverClient(instance), config=FAST).result

        assert result.end_to_end_success, instance.ood_mutations

    def test_a_solved_instance_stays_inside_the_published_limits(self, workflow):
        from capability_subnet.common import constants as C

        instance = workflow.generate_instance(2024, split="hidden")
        result = run_instance(instance, ReferenceSolverClient(instance), config=FAST).result

        assert result.turns_used <= C.MAX_AGENT_TURNS
        assert result.output_tokens <= C.MAX_OUTPUT_TOKENS


class TestEachCapabilityIsSeparatelyVisible:
    """Impairing one capability must move its own axis and no unrelated one.

    This is what makes the per-axis dethrone rule meaningful. If a package could
    abandon a capability without its axis moving, the comparator's "not worse on
    every other axis" requirement would be enforcing nothing.
    """

    EXPECTED_AXIS = {
        "skip_manual": "manual_interpretation",
        "wrong_fault_code": "fault_extraction",
        "bad_sql": "maintenance_sql",
        "broken_python": "diagnostic_python",
        "wrong_part": "inventory_action",
        "incomplete_safety": "safety_validation",
    }

    @pytest.mark.parametrize("impairment,axis", sorted(EXPECTED_AXIS.items()))
    def test_impairment_fails_its_own_axis(self, workflow, impairment, axis):
        instance = workflow.generate_instance(808, split="hidden")
        result = run_instance(
            instance,
            ReferenceSolverClient(instance, frozenset({impairment})),
            config=FAST,
        ).result

        assert not result.stages[axis].passed
        assert not result.end_to_end_success

    @pytest.mark.parametrize("impairment,axis", sorted(EXPECTED_AXIS.items()))
    def test_impairment_leaves_unrelated_axes_alone(self, workflow, impairment, axis):
        instance = workflow.generate_instance(808, split="hidden")
        result = run_instance(
            instance,
            ReferenceSolverClient(instance, frozenset({impairment})),
            config=FAST,
        ).result

        # The final report carries the same values as the earlier stages, so it
        # is expected to move with any of them.
        unrelated = [
            stage
            for stage in STAGES
            if stage not in (axis, "final_json") and not result.stages[stage].passed
        ]
        assert unrelated == []

    def test_a_forbidden_action_is_a_critical_unsafe_action(self, workflow):
        instance = workflow.generate_instance(808, split="hidden")
        result = run_instance(
            instance,
            ReferenceSolverClient(instance, frozenset({"unsafe_plan"})),
            config=FAST,
        ).result

        assert result.critical_unsafe_actions > 0
        assert not result.end_to_end_success

    def test_never_submitting_a_report_fails(self, workflow):
        instance = workflow.generate_instance(808, split="hidden")
        result = run_instance(
            instance,
            ReferenceSolverClient(instance, frozenset({"no_submit"})),
            config=FAST,
        ).result

        assert result.final_state_correct is False
        assert not result.stages["final_json"].passed

    def test_a_schema_violation_fails_the_report_stage(self, workflow):
        instance = workflow.generate_instance(808, split="hidden")
        result = run_instance(
            instance,
            ReferenceSolverClient(instance, frozenset({"malformed_json"})),
            config=FAST,
        ).result

        assert result.stages["final_json"].score == 0.0
        assert "schema invalid" in result.stages["final_json"].detail

    def test_every_named_impairment_prevents_success(self, workflow):
        instance = workflow.generate_instance(555, split="hidden")
        for impairment in IMPAIRMENTS:
            result = run_instance(
                instance,
                ReferenceSolverClient(instance, frozenset({impairment})),
                config=FAST,
            ).result
            assert not result.end_to_end_success, impairment


class TestHiddenMaterialStaysHidden:
    def test_the_visible_payload_carries_no_ground_truth(self, instance):
        import json

        visible = json.dumps(instance.visible_payload())
        truth = instance.truth

        assert truth.fault_code not in visible
        assert truth.required_part_number not in visible
        assert truth.manual_reference not in visible

    def test_the_tool_surface_never_reveals_hidden_cases(self, workflow):
        instance = workflow.generate_instance(1234, split="hidden")
        outcome = run_instance(instance, ReferenceSolverClient(instance), config=FAST)

        # The agent may only ever run the public input through the code tool.
        tool_calls = [call for call in outcome.trace.calls if call.name == "run_diagnostic_python"]
        for call in tool_calls:
            assert call.arguments["input_id"] == instance.diagnostic.public_input_id

    def test_an_unknown_diagnostic_input_is_refused(self, workflow):
        from capability_subnet.sandbox.db_tool import open_database
        from capability_subnet.sandbox.tools import ToolBox

        instance = workflow.generate_instance(1234, split="hidden")
        trace = ExecutionTrace(instance_id=instance.instance_id, instance_seed=instance.seed)
        database = open_database(instance.database)
        try:
            toolbox = ToolBox(instance, database, PythonRunner(), trace)
            call = toolbox.dispatch(
                1,
                "run_diagnostic_python",
                {"code": "def analyze(r, t): return {}", "input_id": "case_0"},
            )
        finally:
            database.close()

        assert call.result["ok"] is False
        assert "Unbekannte Eingabe" in call.result["error"]


class TestSqlTool:
    def test_a_correct_query_returns_the_reference_answer(self, instance):
        from datetime import timedelta

        from capability_subnet.workflows.industrial_maintenance_de_v1.database import (
            LOOKBACK_MONTHS,
            reference_query,
        )
        from capability_subnet.workflows.industrial_maintenance_de_v1.generator import (
            REFERENCE_DATE,
        )

        database = SqliteMaintenanceDatabase(instance.database)
        try:
            statement = reference_query(
                instance.database.columns,
                instance.machine_id,
                instance.truth.fault_code,
                REFERENCE_DATE - timedelta(days=LOOKBACK_MONTHS * 30),
                REFERENCE_DATE,
            )
            result = database.query(statement)
        finally:
            database.close()

        assert result.rows == instance.truth.sql_expected_rows

    @pytest.mark.parametrize(
        "statement",
        [
            "DROP TABLE maintenance_history",
            "DELETE FROM parts",
            "UPDATE parts SET obsolete = 0",
            "INSERT INTO parts VALUES ('x','y','z',1.0,0)",
            "SELECT 1; DROP TABLE parts",
            "PRAGMA table_info(parts)",
            "ATTACH DATABASE '/etc/passwd' AS leak",
        ],
    )
    def test_write_and_control_statements_are_refused(self, instance, statement):
        database = SqliteMaintenanceDatabase(instance.database)
        try:
            with pytest.raises(SqlRejected):
                database.query(statement)
        finally:
            database.close()

    def test_a_comment_cannot_smuggle_a_write_past_the_prefix_check(self):
        with pytest.raises(SqlRejected):
            validate_statement("/* SELECT */ DELETE FROM parts")

    def test_results_are_capped(self, instance):
        limits = ExecutionLimits(max_sql_rows=3)
        database = SqliteMaintenanceDatabase(instance.database, limits=limits)
        try:
            result = database.query("SELECT * FROM maintenance_history")
        finally:
            database.close()

        assert len(result.rows) <= 3
        assert result.truncated

    def test_a_syntax_error_is_a_recoverable_message_not_a_crash(self, instance):
        database = SqliteMaintenanceDatabase(instance.database)
        try:
            with pytest.raises(SqlRejected, match="SQL-Fehler"):
                database.query("SELECT nonexistent_column FROM maintenance_history")
        finally:
            database.close()


class TestPythonRunner:
    def test_correct_code_passes_every_case(self):
        from capability_subnet.sandbox.reference_solver import CORRECT_CODE

        outcome = PythonRunner().run(
            CORRECT_CODE,
            "analyze",
            [{"case_id": "c", "readings": [1.0, 5.0, 5.0, 1.0], "threshold": 2.0}],
        )
        assert outcome.ok
        assert outcome.results[0]["output"] == {
            "peak": 5.0,
            "exceedances": 2,
            "longest_run": 2,
        }

    def test_a_syntax_error_is_data_not_an_exception(self):
        outcome = PythonRunner().run("def analyze(:", "analyze", [])
        assert not outcome.ok
        assert "SyntaxError" in (outcome.fatal or "")

    def test_a_missing_function_is_reported(self):
        outcome = PythonRunner().run("x = 1", "analyze", [])
        assert not outcome.ok
        assert "not defined" in (outcome.fatal or "")

    def test_an_exception_inside_one_case_does_not_lose_the_others(self):
        source = (
            "def analyze(readings, threshold):\n"
            "    if not readings:\n"
            "        raise ValueError('boom')\n"
            "    return {'peak': max(readings), 'exceedances': 0, 'longest_run': 0}\n"
        )
        outcome = PythonRunner().run(
            source,
            "analyze",
            [
                {"case_id": "empty", "readings": [], "threshold": 1.0},
                {"case_id": "full", "readings": [3.0], "threshold": 1.0},
            ],
        )
        assert outcome.ok
        assert "error" in outcome.results[0]
        assert outcome.results[1]["output"]["peak"] == 3.0

    def test_an_infinite_loop_is_stopped(self):
        runner = PythonRunner(ExecutionLimits(python_timeout_seconds=3.0, python_cpu_seconds=2))
        outcome = runner.run(
            "def analyze(readings, threshold):\n    while True:\n        pass\n",
            "analyze",
            [{"case_id": "c", "readings": [1.0], "threshold": 0.5}],
        )
        assert not outcome.ok
        assert outcome.timed_out or "Zeitlimit" in (outcome.fatal or "")

    def test_oversized_source_is_refused_before_execution(self):
        runner = PythonRunner(ExecutionLimits(max_code_bytes=100))
        outcome = runner.run("# " + "x" * 500, "analyze", [])
        assert not outcome.ok
        assert "Bytes" in (outcome.fatal or "")

    def test_a_value_that_cannot_cross_the_boundary_is_an_error_not_a_crash(self):
        source = (
            "def analyze(readings, threshold):\n"
            "    return {'peak': object(), 'exceedances': 0, 'longest_run': 0}\n"
        )
        outcome = PythonRunner().run(
            source, "analyze", [{"case_id": "c", "readings": [1.0], "threshold": 0.5}]
        )
        assert outcome.ok
        assert "error" in outcome.results[0]


class TestLimits:
    def test_a_deadline_expires(self):
        deadline = Deadline(0.0)
        assert deadline.expired
        with pytest.raises(LimitExceeded):
            deadline.check()

    def test_a_fresh_deadline_has_budget(self):
        deadline = Deadline(60.0)
        assert not deadline.expired
        deadline.check()
