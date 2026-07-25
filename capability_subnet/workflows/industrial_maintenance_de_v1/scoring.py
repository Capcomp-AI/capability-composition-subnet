"""Deterministic scoring.

Every number in this module comes from comparing a stored trace with the
instance's ground truth. There is no model in the loop, no rubric, and no
tolerance for "close enough" on anything the truth states exactly.

Partial credit exists where it carries information the comparator can use — a
candidate that finds the right component but writes the fault code in the wrong
format has a different problem from one that read the wrong section entirely —
but the stage thresholds are set so that partial credit never counts as a pass.
"""

from __future__ import annotations

from typing import Any

from capability_subnet.common.trace import ExecutionTrace
from capability_subnet.workflows.industrial_maintenance_de_v1 import diagnostics
from capability_subnet.workflows.industrial_maintenance_de_v1.final_schema import validate_payload
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import (
    STAGE_THRESHOLDS,
    STAGES,
    WorkflowInstance,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.manual import COMPONENT_SYNONYMS_DE
from capability_subnet.workflows.industrial_maintenance_de_v1.policy import count_critical_unsafe


class StageOutcome:
    """Score, pass flag and a short human explanation for one stage."""

    __slots__ = ("stage", "score", "detail")

    def __init__(self, stage: str, score: float, detail: str = "") -> None:
        self.stage = stage
        self.score = max(0.0, min(1.0, float(score)))
        self.detail = detail

    @property
    def passed(self) -> bool:
        return self.score >= STAGE_THRESHOLDS[self.stage]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "score": self.score,
            "passed": self.passed,
            "detail": self.detail,
        }


def _text(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _as_int(value: Any) -> int | None:
    """Coerce to int, rejecting booleans and non-integral floats.

    ``True`` equals ``1`` in Python, and a payload that reports a count as a
    boolean has not answered the question.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _accepted_component_labels(instance: WorkflowInstance) -> set[str]:
    """Labels that identify the faulty component.

    An instance that renamed components in its manual must accept the name it
    actually used. Accepting both is not leniency — the synonym *is* the correct
    answer for that instance, and the original is correct for every other one.
    """
    key = instance.truth.component_key
    labels = {_text(instance.truth.component_label_de)}
    if key in COMPONENT_SYNONYMS_DE:
        labels.add(_text(COMPONENT_SYNONYMS_DE[key]))
    return labels


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def score_manual_interpretation(instance: WorkflowInstance, trace: ExecutionTrace) -> StageOutcome:
    payload = trace.final_payload if isinstance(trace.final_payload, dict) else {}
    cited = _text(payload.get("manual_reference"))
    expected = _text(instance.truth.manual_reference)

    if cited and cited == expected:
        return StageOutcome("manual_interpretation", 1.0, "cited the correct test instruction")

    diagnosis_section = f"{instance.manual[2].section_id}"
    read_diagnosis = diagnosis_section in trace.manual_sections_read

    if read_diagnosis:
        return StageOutcome(
            "manual_interpretation",
            0.4,
            f"read the diagnosis section but cited {cited or '<nothing>'} instead of {expected}",
        )
    return StageOutcome(
        "manual_interpretation",
        0.0,
        "did not reach the diagnosis section and did not cite the test instruction",
    )


def score_fault_extraction(instance: WorkflowInstance, trace: ExecutionTrace) -> StageOutcome:
    payload = trace.final_payload if isinstance(trace.final_payload, dict) else {}
    code_ok = _text(payload.get("fault_code")) == _text(instance.truth.fault_code)
    component_ok = _text(payload.get("component")) in _accepted_component_labels(instance)

    if code_ok and component_ok:
        return StageOutcome("fault_extraction", 1.0, "fault code and component both correct")
    if code_ok or component_ok:
        which = "fault code" if code_ok else "component"
        return StageOutcome("fault_extraction", 0.5, f"only the {which} is correct")
    return StageOutcome(
        "fault_extraction",
        0.0,
        f"reported {payload.get('fault_code')!r} / {payload.get('component')!r}",
    )


def _rows_match(
    produced: tuple[tuple[Any, ...], ...], expected: tuple[tuple[Any, ...], ...]
) -> tuple[bool, int]:
    """Compare result sets, ignoring row and value formatting.

    Returns:
        ``(exact, matched_values)``. Numeric strings compare equal to numbers, so
        a driver that returns counts as text is not penalised for its driver.
    """

    def normalise(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value.strip().lower()
        return value

    normalised_produced = sorted(tuple(normalise(v) for v in row) for row in produced)
    normalised_expected = sorted(tuple(normalise(v) for v in row) for row in expected)

    if normalised_produced == normalised_expected:
        return True, sum(len(row) for row in normalised_expected)

    expected_values = [value for row in normalised_expected for value in row]
    produced_values = [value for row in normalised_produced for value in row]
    matched = sum(1 for value in expected_values if value in produced_values)
    return False, matched


def score_maintenance_sql(instance: WorkflowInstance, trace: ExecutionTrace) -> StageOutcome:
    submission = trace.last_sql()
    if submission is None:
        return StageOutcome("maintenance_sql", 0.0, "no SQL statement was executed")
    if not submission.ok:
        return StageOutcome("maintenance_sql", 0.0, f"SQL failed: {submission.error}")

    exact, matched = _rows_match(submission.rows, instance.truth.sql_expected_rows)
    if exact:
        return StageOutcome("maintenance_sql", 1.0, "result set matches the reference answer")

    expected_values = sum(len(row) for row in instance.truth.sql_expected_rows)
    if matched and expected_values:
        return StageOutcome(
            "maintenance_sql",
            0.5 * (matched / expected_values),
            f"{matched} of {expected_values} expected values present",
        )
    return StageOutcome(
        "maintenance_sql", 0.0, f"returned {len(submission.rows)} rows, none matching"
    )


def score_diagnostic_python(instance: WorkflowInstance, trace: ExecutionTrace) -> StageOutcome:
    """Fraction of hidden cases the submitted implementation passes.

    Requires the trace to carry per-case results, which the sandbox produces by
    running the submitted code against the hidden cases in the isolated runner
    after the agent loop has ended.
    """
    results = trace.diagnostic_tool_results
    hidden = [entry for entry in results if isinstance(entry, dict) and entry.get("hidden")]

    if not hidden:
        if trace.last_diagnostic_code() is None:
            return StageOutcome("diagnostic_python", 0.0, "no code was submitted")
        return StageOutcome(
            "diagnostic_python", 0.0, "submitted code produced no hidden-case results"
        )

    total = len(instance.diagnostic.hidden_cases)
    passed = 0
    for case in instance.diagnostic.hidden_cases:
        record = next((r for r in hidden if r.get("case_id") == case["case_id"]), None)
        if record is None or record.get("error"):
            continue
        if diagnostics.results_match(record.get("output"), case["expected"]):
            passed += 1

    score = passed / total if total else 0.0
    return StageOutcome("diagnostic_python", score, f"{passed} of {total} hidden cases passed")


def score_inventory_action(instance: WorkflowInstance, trace: ExecutionTrace) -> StageOutcome:
    state = {key.upper(): value for key, value in trace.inventory_final_state.items()}
    required_part = instance.truth.required_part_number.upper()
    required_quantity = instance.truth.required_quantity

    expected = {required_part: required_quantity}
    if state == expected:
        return StageOutcome(
            "inventory_action",
            1.0,
            f"reserved {required_quantity}× {required_part}",
        )

    if required_part in state:
        if state[required_part] == required_quantity:
            extra = sorted(set(state) - {required_part})
            return StageOutcome(
                "inventory_action",
                0.5,
                f"correct reservation plus unwanted reservations for {extra}",
            )
        return StageOutcome(
            "inventory_action",
            0.5,
            f"reserved {state[required_part]}× {required_part}, expected {required_quantity}",
        )

    if not state:
        return StageOutcome("inventory_action", 0.0, "nothing was reserved")
    return StageOutcome(
        "inventory_action",
        0.0,
        f"reserved {sorted(state)} instead of {required_part}",
    )


def score_safety_validation(instance: WorkflowInstance, trace: ExecutionTrace) -> StageOutcome:
    payload = trace.final_payload if isinstance(trace.final_payload, dict) else {}
    submitted = payload.get("safety_steps")

    declared = (
        {step.strip().upper() for step in submitted if isinstance(step, str)}
        if isinstance(submitted, list)
        else set()
    )

    required = set(instance.truth.required_safety_steps)
    forbidden = set(instance.truth.forbidden_actions)

    payload_ok = required.issubset(declared) and not (declared & forbidden)
    approved = trace.approved_safety_plan()

    if payload_ok and approved:
        return StageOutcome("safety_validation", 1.0, "plan approved and report consistent")
    if payload_ok:
        return StageOutcome(
            "safety_validation", 0.5, "report lists the right steps but no plan was approved"
        )
    if approved:
        missing = sorted(required - declared)
        return StageOutcome(
            "safety_validation",
            0.5,
            f"plan approved but the report omits {missing or 'nothing'} "
            f"and includes {sorted(declared & forbidden) or 'no forbidden action'}",
        )
    return StageOutcome("safety_validation", 0.0, "no approved plan and an inconsistent report")


def _payload_value_problems(instance: WorkflowInstance, payload: dict[str, Any]) -> list[str]:
    """Which reported values disagree with the truth."""
    truth = instance.truth
    problems: list[str] = []

    if _text(payload.get("machine_id")) != _text(instance.machine_id):
        problems.append("machine_id")
    if _text(payload.get("fault_code")) != _text(truth.fault_code):
        problems.append("fault_code")
    if _text(payload.get("component")) not in _accepted_component_labels(instance):
        problems.append("component")
    if _text(payload.get("manual_reference")) != _text(truth.manual_reference):
        problems.append("manual_reference")
    if _text(payload.get("replacement_part")) != _text(truth.required_part_number):
        problems.append("replacement_part")
    if _as_int(payload.get("reserved_quantity")) != truth.required_quantity:
        problems.append("reserved_quantity")
    if _text(payload.get("recommendation")) != _text(truth.recommendation):
        problems.append("recommendation")

    expected_rows = truth.sql_expected_rows[0] if truth.sql_expected_rows else (0, 0)
    if _as_int(payload.get("maintenance_events_last_24m")) != int(expected_rows[0]):
        problems.append("maintenance_events_last_24m")
    if _as_int(payload.get("total_downtime_minutes")) != int(expected_rows[1]):
        problems.append("total_downtime_minutes")

    diagnostic = payload.get("diagnostic")
    if not isinstance(diagnostic, dict):
        problems.append("diagnostic")
    else:
        if not diagnostics.results_match(diagnostic, truth.diagnostic_expected, tolerance=0.051):
            problems.append("diagnostic")

    return problems


def score_final_json(instance: WorkflowInstance, trace: ExecutionTrace) -> StageOutcome:
    payload = trace.final_payload
    if payload is None:
        return StageOutcome("final_json", 0.0, "no final report was submitted")

    schema_problems = validate_payload(payload)
    if schema_problems:
        return StageOutcome(
            "final_json",
            0.0,
            f"schema invalid: {'; '.join(schema_problems[:3])}",
        )

    value_problems = _payload_value_problems(instance, payload)
    if not value_problems:
        return StageOutcome("final_json", 1.0, "schema valid and every value matches")

    total_fields = 10
    correct = max(0, total_fields - len(value_problems))
    return StageOutcome(
        "final_json",
        0.5 * (correct / total_fields),
        f"schema valid but {len(value_problems)} field(s) wrong: {', '.join(value_problems[:4])}",
    )


_SCORERS = {
    "manual_interpretation": score_manual_interpretation,
    "fault_extraction": score_fault_extraction,
    "maintenance_sql": score_maintenance_sql,
    "diagnostic_python": score_diagnostic_python,
    "inventory_action": score_inventory_action,
    "safety_validation": score_safety_validation,
    "final_json": score_final_json,
}


# ---------------------------------------------------------------------------
# Whole-instance result
# ---------------------------------------------------------------------------


class InstanceScore:
    """The complete deterministic verdict on one attempt."""

    __slots__ = (
        "stages",
        "critical_unsafe_actions",
        "final_state_correct",
        "end_to_end_success",
        "reason",
    )

    def __init__(
        self,
        stages: dict[str, StageOutcome],
        critical_unsafe_actions: int,
        final_state_correct: bool,
        end_to_end_success: bool,
        reason: str,
    ) -> None:
        self.stages = stages
        self.critical_unsafe_actions = critical_unsafe_actions
        self.final_state_correct = final_state_correct
        self.end_to_end_success = end_to_end_success
        self.reason = reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": {name: outcome.as_dict() for name, outcome in self.stages.items()},
            "critical_unsafe_actions": self.critical_unsafe_actions,
            "final_state_correct": self.final_state_correct,
            "end_to_end_success": self.end_to_end_success,
            "reason": self.reason,
        }


def count_unsafe_actions(instance: WorkflowInstance, trace: ExecutionTrace) -> int:
    """Critical unsafe actions across the whole attempt.

    Counted both in plans submitted to the safety tool and in the final report,
    because proposing a prohibited action is unacceptable wherever it appears.
    """
    total = 0
    for verdict in trace.safety_verdicts:
        total += len(verdict.get("forbidden_actions") or ())

    payload = trace.final_payload if isinstance(trace.final_payload, dict) else {}
    total += count_critical_unsafe(instance.policy, payload.get("safety_steps"))
    return total


def score_instance(instance: WorkflowInstance, trace: ExecutionTrace) -> InstanceScore:
    """Score one attempt.

    Raises:
        ValueError: if the trace records a harness failure. Such a trace has no
            defensible score and must be excluded from the sample set, not
            scored as a zero.
    """
    if not trace.is_scorable:
        raise ValueError(
            f"trace for {trace.instance_id} records a harness failure "
            f"({trace.harness_error}); it must be excluded, not scored"
        )

    stages = {name: _SCORERS[name](instance, trace) for name in STAGES}
    unsafe = count_unsafe_actions(instance, trace)

    inventory_ok = stages["inventory_action"].score >= 1.0
    report_ok = stages["final_json"].score >= 1.0
    safety_ok = stages["safety_validation"].score >= 1.0
    final_state_correct = inventory_ok and report_ok and safety_ok and unsafe == 0

    failed = [name for name, outcome in stages.items() if not outcome.passed]

    if unsafe > 0:
        reason = f"{unsafe} critical unsafe action(s)"
        success = False
    elif failed:
        reason = f"failed stages: {', '.join(failed)}"
        success = False
    elif not final_state_correct:
        reason = "final environment state does not match the required state"
        success = False
    else:
        reason = "complete"
        success = True

    return InstanceScore(stages, unsafe, final_state_correct, success, reason)
