"""Tool dispatch.

Binds the seven published tool names to the services behind them and records
everything into the trace as it goes. Two rules govern every handler here:

* A tool result never reveals whether the candidate is right. The inventory
  service accepts a reservation for the wrong part; the safety engine reports
  which required steps are missing but not which fault was diagnosed; the code
  runner reports what the submitted function returned, not whether it matches.
  Anything else would turn the tool surface into an answer key.
* A malformed call is a recoverable error, not a crash. The candidate gets a
  message in the same language as the environment and can correct itself on the
  next turn, which is exactly the tool-recovery behaviour the workflow means to
  measure.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from capability_subnet.common.trace import ExecutionTrace, SqlSubmission, ToolCall
from capability_subnet.sandbox.db_tool import MaintenanceDatabase, SqlRejected
from capability_subnet.sandbox.limits import ExecutionLimits
from capability_subnet.sandbox.python_runner import PythonRunner, public_case
from capability_subnet.workflows.industrial_maintenance_de_v1 import tools_schema as T
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import WorkflowInstance
from capability_subnet.workflows.industrial_maintenance_de_v1.inventory import InventorySimulator
from capability_subnet.workflows.industrial_maintenance_de_v1.policy import (
    evaluate_plan,
    render_policy_de,
)

log = logging.getLogger(__name__)


class ToolBox:
    """The tool surface for one instance."""

    def __init__(
        self,
        instance: WorkflowInstance,
        database: MaintenanceDatabase,
        runner: PythonRunner,
        trace: ExecutionTrace,
        limits: ExecutionLimits | None = None,
    ) -> None:
        self.instance = instance
        self.database = database
        self.runner = runner
        self.trace = trace
        self.limits = limits or ExecutionLimits()
        self.inventory = InventorySimulator.from_items(instance.inventory)
        self.submitted = False

    # -- dispatch -----------------------------------------------------------

    def dispatch(self, turn: int, name: str, arguments: dict[str, Any]) -> ToolCall:
        """Run one tool call and record it."""
        started = time.monotonic()
        call = ToolCall(turn=turn, name=name, arguments=arguments)

        handler = self._handlers().get(name)
        if handler is None:
            call.ok = False
            call.error = f"Unbekanntes Werkzeug: {name}"
            call.result = {"error": call.error, "available_tools": list(T.TOOL_NAMES)}
        else:
            try:
                call.result = handler(arguments or {})
            except SqlRejected as exc:
                call.ok = False
                call.error = str(exc)
                call.result = {"error": str(exc)}
            except (TypeError, ValueError, KeyError) as exc:
                call.ok = False
                call.error = f"Ungültige Argumente: {exc}"
                call.result = {"error": call.error}
            except Exception as exc:  # noqa: BLE001
                # Recorded as a failed tool call and nothing more.
                #
                # This deliberately does NOT mark the trace as a harness failure.
                # The arguments came from the candidate, so a candidate that
                # found an input crashing a handler could otherwise force its own
                # instance to be excluded from scoring — and a candidate that
                # excludes exactly the instances it is failing raises its measured
                # completion rate for free. Infrastructure failures are detected
                # where they actually originate: opening the database, serving the
                # model, running the loop.
                log.exception(
                    "tool %s raised on candidate-supplied arguments; recording it as a "
                    "failed tool call",
                    name,
                )
                call.ok = False
                call.error = f"Werkzeugfehler: {exc}"
                call.result = {"error": call.error}

        call.seconds = time.monotonic() - started
        self.trace.calls.append(call)
        return call

    def _handlers(self):
        return {
            T.READ_MANUAL: self._read_manual,
            T.QUERY_MAINTENANCE_DB: self._query_db,
            T.RUN_DIAGNOSTIC_PYTHON: self._run_python,
            T.SEARCH_INVENTORY: self._search_inventory,
            T.RESERVE_INVENTORY: self._reserve_inventory,
            T.CHECK_SAFETY_PLAN: self._check_safety,
            T.SUBMIT_FINAL_JSON: self._submit,
        }

    # -- handlers -----------------------------------------------------------

    def _read_manual(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("section_or_query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("section_or_query muss ein nicht-leerer Text sein")

        self.trace.manual_queries.append(query)
        matches = self.instance.search_manual(query)

        if not matches:
            return {
                "found": False,
                "message": "Kein passender Abschnitt gefunden.",
                "available_sections": [
                    {"section_id": section.section_id, "title": section.title_de}
                    for section in self.instance.manual
                ],
            }

        for section in matches:
            if section.section_id not in self.trace.manual_sections_read:
                self.trace.manual_sections_read.append(section.section_id)

        return {
            "found": True,
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title_de,
                    "body": section.body_de,
                }
                for section in matches
            ],
        }

    def _query_db(self, arguments: dict[str, Any]) -> dict[str, Any]:
        sql = arguments.get("sql")
        if not isinstance(sql, str):
            raise ValueError("sql muss ein Text sein")

        try:
            result = self.database.query(sql)
        except SqlRejected as exc:
            self.trace.sql_submissions.append(SqlSubmission(statement=sql, error=str(exc)))
            raise

        self.trace.sql_submissions.append(
            SqlSubmission(statement=sql, columns=result.columns, rows=result.rows)
        )
        return result.as_dict()

    def _run_python(self, arguments: dict[str, Any]) -> dict[str, Any]:
        code = arguments.get("code")
        input_id = arguments.get("input_id")
        if not isinstance(code, str):
            raise ValueError("code muss ein Text sein")
        if not isinstance(input_id, str):
            raise ValueError("input_id muss ein Text sein")

        task = self.instance.diagnostic
        if input_id.strip() != task.public_input_id:
            return {
                "ok": False,
                "error": f"Unbekannte Eingabe {input_id!r}.",
                "available_inputs": [task.public_input_id],
            }

        self.trace.diagnostic_code_submissions.append(code)

        outcome = self.runner.run(
            code,
            task.function_name,
            [
                public_case(
                    task.public_input_id, list(task.public_readings), task.public_threshold
                )
            ],
        )

        if not outcome.ok:
            return {"ok": False, "error": outcome.fatal, "timed_out": outcome.timed_out}

        record = outcome.results[0] if outcome.results else {}
        self.trace.diagnostic_tool_results.append(record)

        if record.get("error"):
            return {"ok": False, "error": record["error"]}
        return {"ok": True, "input_id": input_id, "output": record.get("output")}

    def _search_inventory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        part_number = arguments.get("part_number")
        if not isinstance(part_number, str):
            raise ValueError("part_number muss ein Text sein")
        return self.inventory.search(part_number)

    def _reserve_inventory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        part_number = arguments.get("part_number")
        quantity = arguments.get("quantity")
        if not isinstance(part_number, str):
            raise ValueError("part_number muss ein Text sein")
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise ValueError("quantity muss eine ganze Zahl sein")
        return self.inventory.reserve(part_number, quantity)

    def _check_safety(self, arguments: dict[str, Any]) -> dict[str, Any]:
        plan = arguments.get("plan")
        if plan is None:
            raise ValueError("plan fehlt")

        verdict = evaluate_plan(self.instance.policy, plan)
        payload = verdict.as_dict()
        payload["policy"] = render_policy_de(self.instance.policy)
        self.trace.safety_verdicts.append(payload)
        return payload

    def _submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.submitted:
            return {
                "accepted": False,
                "message": "Der Abschlussbericht wurde bereits übermittelt.",
            }

        payload = arguments.get("payload")
        if payload is None:
            raise ValueError("payload fehlt")

        self.trace.final_payload = payload
        self.submitted = True
        return {"accepted": True, "message": "Abschlussbericht entgegengenommen."}

    # -- teardown -----------------------------------------------------------

    def finalise(self) -> None:
        """Capture the tool services' end state onto the trace.

        This is what the inventory and safety stages are actually scored on, so
        it happens exactly once, after the loop has ended, and before anything is
        torn down.
        """
        self.trace.inventory_final_state = self.inventory.final_state()
        self.trace.inventory_attempts = list(self.inventory.attempts)

    def run_hidden_diagnostics(self) -> None:
        """Run the last submitted code against the hidden cases.

        Deliberately after the agent loop: the hidden inputs must never be
        reachable from a turn the candidate can observe.
        """
        source = self.trace.last_diagnostic_code()
        if source is None:
            return

        from capability_subnet.sandbox.python_runner import hidden_cases

        outcome = self.runner.run(
            source,
            self.instance.diagnostic.function_name,
            hidden_cases(self.instance.diagnostic.hidden_cases),
        )

        if not outcome.ok:
            # A fatal failure means every hidden case failed, which is a real
            # result: the submitted code does not run.
            self.trace.diagnostic_tool_results.extend(
                {
                    "case_id": case["case_id"],
                    "hidden": True,
                    "error": outcome.fatal or "Ausführung fehlgeschlagen",
                }
                for case in self.instance.diagnostic.hidden_cases
            )
            return

        self.trace.diagnostic_tool_results.extend(outcome.results)
