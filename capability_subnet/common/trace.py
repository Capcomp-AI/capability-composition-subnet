"""Execution traces.

A trace is the complete, immutable record of one candidate's attempt at one
instance: every tool call it made, every result it got back, the final payload it
submitted, and the state the tool services were left in.

The trace is the *only* thing the scorer reads. The scorer never talks to the
model, never re-runs anything, and never asks another model for an opinion. That
is what makes the resulting numbers reproducible and the paired statistics valid:
re-scoring a stored trace a year later gives the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StopReason = Literal[
    "submitted",
    "turn_limit",
    "token_limit",
    "timeout",
    "model_error",
    "harness_error",
    "unrecoverable",
]


@dataclass(slots=True)
class ToolCall:
    """One tool invocation and its result."""

    turn: int
    name: str
    arguments: dict[str, Any]
    result: Any = None
    ok: bool = True
    error: str | None = None
    seconds: float = 0.0

    def summary(self) -> str:
        state = "ok" if self.ok else f"error: {self.error}"
        return f"turn {self.turn}: {self.name} ({state})"


@dataclass(slots=True)
class SqlSubmission:
    """One SQL statement the candidate ran, with what it returned."""

    statement: str
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class ExecutionTrace:
    """Everything one sandbox run produced."""

    instance_id: str
    instance_seed: int
    split: str = "hidden"

    calls: list[ToolCall] = field(default_factory=list)
    final_payload: Any = None
    stop_reason: StopReason = "harness_error"

    turns_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    peak_vram_gb: float = 0.0

    #: Set when the harness itself failed. A trace with this set is excluded from
    #: scoring rather than counted as a model failure — flaky infrastructure must
    #: never terminate a candidate.
    harness_error: str | None = None

    # -- captured side state ------------------------------------------------
    manual_queries: list[str] = field(default_factory=list)
    manual_sections_read: list[str] = field(default_factory=list)
    sql_submissions: list[SqlSubmission] = field(default_factory=list)
    diagnostic_code_submissions: list[str] = field(default_factory=list)
    diagnostic_tool_results: list[Any] = field(default_factory=list)
    inventory_final_state: dict[str, int] = field(default_factory=dict)
    inventory_attempts: list[dict[str, Any]] = field(default_factory=list)
    safety_verdicts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_scorable(self) -> bool:
        """Whether this trace may take part in scoring."""
        return self.harness_error is None

    def calls_to(self, name: str) -> list[ToolCall]:
        return [call for call in self.calls if call.name == name]

    def last_sql(self) -> SqlSubmission | None:
        """The last statement that executed without error, else the last attempt.

        Candidates routinely probe the schema with a couple of exploratory
        queries before the real one. Scoring the last *successful* statement
        matches what a technician would consider the answer.
        """
        successful = [item for item in self.sql_submissions if item.ok]
        if successful:
            return successful[-1]
        return self.sql_submissions[-1] if self.sql_submissions else None

    def last_diagnostic_code(self) -> str | None:
        return self.diagnostic_code_submissions[-1] if self.diagnostic_code_submissions else None

    def approved_safety_plan(self) -> bool:
        return any(verdict.get("approved") for verdict in self.safety_verdicts)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, stored alongside the sample row for audit."""
        return {
            "instance_id": self.instance_id,
            "instance_seed": self.instance_seed,
            "split": self.split,
            "stop_reason": self.stop_reason,
            "turns_used": self.turns_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": round(self.wall_seconds, 3),
            "harness_error": self.harness_error,
            "calls": [
                {
                    "turn": call.turn,
                    "name": call.name,
                    "arguments": call.arguments,
                    "ok": call.ok,
                    "error": call.error,
                }
                for call in self.calls
            ],
            "final_payload": self.final_payload,
            "inventory_final_state": self.inventory_final_state,
            "safety_verdicts": self.safety_verdicts,
        }
