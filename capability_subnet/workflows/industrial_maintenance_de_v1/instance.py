"""Instance and ground-truth data model.

An instance bundles everything a candidate can see (manual, sensor log, database
schema, inventory, policy) with everything only the scorer may see (the truth).
The two are separate objects on purpose: the sandbox hands the visible half to
the agent container and keeps the truth in a process the agent cannot reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The ordered stage chain. Later stages consume earlier outputs, which is what
#: makes this a workflow rather than a list of independent benchmark questions.
STAGES: tuple[str, ...] = (
    "manual_interpretation",
    "fault_extraction",
    "maintenance_sql",
    "diagnostic_python",
    "inventory_action",
    "safety_validation",
    "final_json",
)

#: Per-stage pass thresholds. A stage below its threshold breaks the chain, so
#: end-to-end success requires clearing every one of them.
STAGE_THRESHOLDS: dict[str, float] = {
    "manual_interpretation": 1.0,
    "fault_extraction": 1.0,
    "maintenance_sql": 1.0,
    "diagnostic_python": 0.8,
    "inventory_action": 1.0,
    "safety_validation": 1.0,
    "final_json": 1.0,
}

#: Stages the comparator treats as capability axes. Every one of them must stay
#: at least not-worse for a challenger to take the throne, which is what stops a
#: package from trading a capability away for a better average.
CRITICAL_AXES: tuple[str, ...] = STAGES

RECOMMENDATIONS = ("sofortige_instandsetzung", "geplante_wartung", "beobachtung")


@dataclass(frozen=True, slots=True)
class ManualSection:
    """One addressable section of the technical manual."""

    section_id: str
    title_de: str
    body_de: str

    def matches(self, query: str) -> bool:
        """Whether ``query`` addresses this section.

        Accepts the section number, a word from the title, or a phrase from the
        body. Retrieval is deliberately forgiving: the workflow measures whether
        a candidate can *interpret* the manual, not whether it can guess an
        exact-match search key.
        """
        needle = (query or "").strip().lower()
        if not needle:
            return False
        if needle == self.section_id.lower() or needle.startswith(self.section_id.lower()):
            return True
        return needle in self.title_de.lower() or needle in self.body_de.lower()


@dataclass(frozen=True, slots=True)
class DiagnosticTask:
    """The Python task, its visible input, and its hidden test cases."""

    function_name: str
    signature: str
    description_de: str
    #: Input the agent can run through the tool to sanity-check its code.
    public_input_id: str
    public_readings: tuple[float, ...]
    public_threshold: float
    #: Hidden cases the scorer runs the submitted code against. Never sent to
    #: the agent container.
    hidden_cases: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class InventoryItem:
    part_number: str
    description_de: str
    on_hand: int
    reserved: int
    location_de: str
    obsolete: bool = False
    superseded_by: str | None = None


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """The written policy the deterministic rule engine enforces."""

    policy_id: str
    required_steps: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    step_descriptions_de: dict[str, str] = field(default_factory=dict)
    forbidden_descriptions_de: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatabaseSnapshot:
    """The maintenance history the SQL stage runs against."""

    schema_description_de: str
    ddl: tuple[str, ...]
    rows: dict[str, tuple[tuple[Any, ...], ...]]
    columns: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Everything the scorer compares against. Never leaves the scorer process."""

    fault_code: str
    component_key: str
    component_label_de: str
    manual_reference: str
    sql_question_de: str
    sql_expected_rows: tuple[tuple[Any, ...], ...]
    sql_expected_columns: tuple[str, ...]
    diagnostic_expected: dict[str, Any]
    required_part_number: str
    required_quantity: int
    required_safety_steps: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    recommendation: str
    peak_value: float
    exceedance_count: int
    longest_run: int
    warn_threshold: float
    critical_threshold: float


@dataclass(frozen=True, slots=True)
class WorkflowInstance:
    """One complete, self-contained problem."""

    instance_id: str
    seed: int
    split: str
    machine_id: str
    machine_model_de: str
    machine_location_de: str
    sensor_key: str
    sensor_label_de: str
    sensor_unit: str
    task_prompt_de: str
    manual: tuple[ManualSection, ...]
    sensor_log: str
    database: DatabaseSnapshot
    diagnostic: DiagnosticTask
    inventory: tuple[InventoryItem, ...]
    policy: SafetyPolicy
    truth: GroundTruth
    ood_mutations: tuple[str, ...] = ()

    # -- views --------------------------------------------------------------

    def visible_payload(self) -> dict[str, Any]:
        """The half of the instance the agent container is allowed to hold.

        Building this explicitly, rather than filtering the full object at the
        boundary, is what guarantees the truth cannot leak by accident: a new
        field added to the truth is invisible here until someone deliberately
        adds it.
        """
        return {
            "instance_id": self.instance_id,
            "machine_id": self.machine_id,
            "machine_model": self.machine_model_de,
            "location": self.machine_location_de,
            "task_prompt": self.task_prompt_de,
            "sensor_log": self.sensor_log,
            "manual_sections": [
                {"section_id": section.section_id, "title": section.title_de}
                for section in self.manual
            ],
            "database_schema": self.database.schema_description_de,
            "diagnostic_task": {
                "function_name": self.diagnostic.function_name,
                "signature": self.diagnostic.signature,
                "description": self.diagnostic.description_de,
                "public_input_id": self.diagnostic.public_input_id,
            },
            "safety_policy_id": self.policy.policy_id,
            "ood_mutations": list(self.ood_mutations),
        }

    def manual_section(self, section_id: str) -> ManualSection | None:
        for section in self.manual:
            if section.section_id == section_id:
                return section
        return None

    def search_manual(self, query: str) -> list[ManualSection]:
        return [section for section in self.manual if section.matches(query)]

    def inventory_item(self, part_number: str) -> InventoryItem | None:
        needle = (part_number or "").strip().upper()
        for item in self.inventory:
            if item.part_number.upper() == needle:
                return item
        return None
