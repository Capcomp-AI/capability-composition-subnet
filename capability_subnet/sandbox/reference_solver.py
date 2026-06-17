"""A scripted solver used to test the harness, not to score anything.

This client implements the model interface but is not a model. It is constructed
with the instance and therefore already knows the answers, which makes it useless
as a candidate and extremely useful as a fixture:

* it proves the environment is *solvable* — if the reference solver cannot reach
  end-to-end success, the instance generator, the tools or the scorer is broken,
  and no candidate score from that instance would mean anything;
* it exercises every tool, every stage scorer and the whole trace pipeline
  without a GPU, which is what lets the test suite run anywhere;
* with impairments switched on it produces candidates that fail in specific,
  named ways, so the scorer and the comparator can be tested against known
  outcomes rather than against whatever a real model happened to do.

It is never wired into an evaluation path. The engine talks to served candidates
over the same interface and has no way to construct this class, because
constructing it requires the ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from capability_subnet.sandbox.model_client import ModelMessage, ModelReply
from capability_subnet.workflows.industrial_maintenance_de_v1 import tools_schema as T
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import WorkflowInstance

#: Named ways to be wrong. Each maps to a specific stage failure, so a test can
#: assert that impairing one capability moves exactly one axis.
IMPAIRMENTS = (
    "skip_manual",  # never reads the manual; cites no test instruction
    "wrong_fault_code",  # reports a plausible but incorrect fault code
    "bad_sql",  # forgets the machine filter, so sibling rows inflate the count
    "broken_python",  # code that passes the visible case and fails hidden ones
    "wrong_part",  # reserves the superseded part number
    "unsafe_plan",  # proposes a forbidden action
    "incomplete_safety",  # omits a required precaution
    "no_submit",  # runs out of turns without submitting
    "malformed_json",  # submits a payload that fails schema validation
)

CORRECT_CODE = '''
def analyze(readings, threshold):
    if not readings:
        return {"peak": 0.0, "exceedances": 0, "longest_run": 0}
    peak = max(readings)
    exceedances = 0
    longest = 0
    current = 0
    for value in readings:
        if value > threshold:
            exceedances += 1
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return {"peak": float(peak), "exceedances": int(exceedances), "longest_run": int(longest)}
'''

#: Passes the visible case by conflating the run length with the count. On the
#: visible window those two happen to differ, so this is caught immediately — the
#: point is that it is *plausible* code, not obviously broken code.
BROKEN_CODE = '''
def analyze(readings, threshold):
    if not readings:
        return {"peak": 0.0, "exceedances": 0, "longest_run": 0}
    over = [value for value in readings if value >= threshold]
    return {
        "peak": float(max(readings)),
        "exceedances": int(len(over)),
        "longest_run": int(len(over)),
    }
'''


@dataclass
class ReferenceSolverClient:
    """A deterministic solver over one instance."""

    instance: WorkflowInstance
    impairments: frozenset[str] = frozenset()
    tokens_per_turn: int = 220
    _step: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.impairments) - set(IMPAIRMENTS))
        if unknown:
            raise ValueError(f"unknown impairments: {unknown}; choose from {list(IMPAIRMENTS)}")

    # -- model interface ----------------------------------------------------

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
        *,
        seed: int,
        max_tokens: int,
    ) -> ModelReply:
        plan = self._plan()
        if self._step >= len(plan):
            self._step += 1
            return ModelReply(content="Vorgang abgeschlossen.", tool_calls=[])

        name, arguments = plan[self._step]
        self._step += 1

        return ModelReply(
            content=None,
            tool_calls=[
                {
                    "id": f"call_{self._step}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
            input_tokens=self.tokens_per_turn * 4,
            output_tokens=self.tokens_per_turn,
        )

    # -- plan ---------------------------------------------------------------

    def _plan(self) -> list[tuple[str, dict[str, Any]]]:
        instance = self.instance
        truth = instance.truth
        chapter = instance.manual[0].section_id.split(".")[0]

        steps: list[tuple[str, dict[str, Any]]] = []

        if "skip_manual" not in self.impairments:
            steps.extend(
                [
                    (T.READ_MANUAL, {"section_or_query": f"{chapter}.2"}),
                    (T.READ_MANUAL, {"section_or_query": f"{chapter}.3"}),
                    (T.READ_MANUAL, {"section_or_query": f"{chapter}.4"}),
                    (T.READ_MANUAL, {"section_or_query": f"{chapter}.1"}),
                ]
            )

        steps.append((T.QUERY_MAINTENANCE_DB, {"sql": self._sql()}))
        steps.append(
            (
                T.RUN_DIAGNOSTIC_PYTHON,
                {
                    "code": BROKEN_CODE if "broken_python" in self.impairments else CORRECT_CODE,
                    "input_id": instance.diagnostic.public_input_id,
                },
            )
        )

        part = self._part_number()
        steps.append((T.SEARCH_INVENTORY, {"part_number": part}))
        steps.append(
            (T.RESERVE_INVENTORY, {"part_number": part, "quantity": truth.required_quantity})
        )
        steps.append((T.CHECK_SAFETY_PLAN, {"plan": {"steps": self._safety_steps()}}))

        if "no_submit" not in self.impairments:
            steps.append((T.SUBMIT_FINAL_JSON, {"payload": self._payload()}))

        return steps

    def _sql(self) -> str:
        """The reference query, or one missing the machine filter."""
        from datetime import timedelta

        from capability_subnet.workflows.industrial_maintenance_de_v1.database import (
            DEFAULT_COLUMNS,
            LOOKBACK_MONTHS,
            reference_query,
        )
        from capability_subnet.workflows.industrial_maintenance_de_v1.generator import (
            REFERENCE_DATE,
        )

        cutoff = REFERENCE_DATE - timedelta(days=LOOKBACK_MONTHS * 30)
        columns = self.instance.database.columns

        if "bad_sql" in self.impairments:
            lookup = dict(
                zip(DEFAULT_COLUMNS["maintenance_history"], columns["maintenance_history"], strict=True)
            )
            return (
                f"SELECT COUNT(*) AS anzahl, "
                f"COALESCE(SUM({lookup['downtime_minutes']}), 0) AS ausfallzeit_gesamt "
                f"FROM maintenance_history "
                f"WHERE {lookup['fault_code']} = '{self.instance.truth.fault_code}' "
                f"AND {lookup['event_date']} BETWEEN '{cutoff.isoformat()}' "
                f"AND '{REFERENCE_DATE.isoformat()}'"
            )

        return reference_query(
            columns,
            self.instance.machine_id,
            self.instance.truth.fault_code,
            cutoff,
            REFERENCE_DATE,
        )

    def _part_number(self) -> str:
        correct = self.instance.truth.required_part_number
        if "wrong_part" not in self.impairments:
            return correct
        superseded = next(
            (
                item.part_number
                for item in self.instance.inventory
                if item.superseded_by == correct
            ),
            None,
        )
        return superseded or correct

    def _safety_steps(self) -> list[str]:
        required = list(self.instance.truth.required_safety_steps)
        if "unsafe_plan" in self.impairments and self.instance.truth.forbidden_actions:
            return required + [self.instance.truth.forbidden_actions[0]]
        if "incomplete_safety" in self.impairments and len(required) > 1:
            return required[:-1]
        return required

    def _payload(self) -> dict[str, Any]:
        instance = self.instance
        truth = instance.truth
        expected = truth.sql_expected_rows[0] if truth.sql_expected_rows else (0, 0)

        payload: dict[str, Any] = {
            "machine_id": instance.machine_id,
            "fault_code": truth.fault_code,
            "component": truth.component_label_de,
            "manual_reference": truth.manual_reference,
            "maintenance_events_last_24m": int(expected[0]),
            "total_downtime_minutes": int(expected[1]),
            "diagnostic": dict(truth.diagnostic_expected),
            "replacement_part": self._part_number(),
            "reserved_quantity": truth.required_quantity,
            "safety_steps": self._safety_steps(),
            "recommendation": truth.recommendation,
        }

        if "skip_manual" in self.impairments:
            payload["manual_reference"] = "unbekannt"
        if "wrong_fault_code" in self.impairments:
            payload["fault_code"] = "XXX-0-000"
        if "bad_sql" in self.impairments:
            payload["maintenance_events_last_24m"] = int(expected[0]) + 3
        if "broken_python" in self.impairments:
            payload["diagnostic"] = {
                "peak": truth.peak_value,
                "exceedances": truth.exceedance_count,
                "longest_run": truth.exceedance_count,
            }
        if "malformed_json" in self.impairments:
            payload.pop("recommendation", None)
            payload["unerwartetes_feld"] = "abc"

        return payload
