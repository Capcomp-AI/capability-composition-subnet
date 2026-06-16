"""The fixed agent loop.

One loop, one system prompt, one tool surface, identical for every candidate.
That uniformity is the point: any difference in outcome between two candidates
has to come from the merged package, because nothing else about the run differs.

The loop is deliberately plain — observe, generate, call a tool, append the
result, repeat. It does no retries, no reflection prompting, no repair of
malformed output and no hints. A package that needs scaffolding to complete the
workflow has not solved the workflow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from capability_subnet.common import constants as C
from capability_subnet.common.trace import ExecutionTrace
from capability_subnet.sandbox.limits import Deadline, ExecutionLimits, LimitExceeded
from capability_subnet.sandbox.model_client import ModelClient, ModelMessage, parse_tool_call
from capability_subnet.sandbox.tools import ToolBox
from capability_subnet.workflows.industrial_maintenance_de_v1 import tools_schema as T
from capability_subnet.workflows.industrial_maintenance_de_v1.final_schema import (
    FINAL_PAYLOAD_SCHEMA,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import WorkflowInstance

log = logging.getLogger(__name__)

SYSTEM_PROMPT_DE = (
    "Du bist ein Instandhaltungsagent in einem industriellen Betrieb. Du arbeitest "
    "ausschließlich über die bereitgestellten Werkzeuge.\n\n"
    "Regeln:\n"
    "- Nutze in jedem Schritt genau ein Werkzeug.\n"
    "- Stütze jede Aussage auf Werkzeugausgaben, nicht auf Annahmen.\n"
    "- Grenzwerte, Fehlercodes, Prüfanweisungen, Ersatzteilnummern und "
    "Sicherungsmaßnahmen stehen ausschließlich im Handbuch.\n"
    "- Beende den Vorgang mit genau einem Aufruf von submit_final_json.\n"
    "- Gib keinen Freitext außerhalb von Werkzeugaufrufen aus."
)


@dataclass(slots=True)
class LoopResult:
    """Why the loop stopped."""

    stop_reason: str
    turns_used: int
    input_tokens: int
    output_tokens: int


def build_initial_messages(instance: WorkflowInstance) -> list[ModelMessage]:
    """The opening conversation.

    Everything the candidate needs to *start* is here — the task, the log, the
    database schema, the code contract and the report schema. Everything it needs
    to *finish* is behind a tool.
    """
    schema_text = json.dumps(FINAL_PAYLOAD_SCHEMA, ensure_ascii=False, indent=2, sort_keys=True)

    observation = (
        f"{instance.task_prompt_de}\n\n"
        f"=== ANLAGENPROTOKOLL ===\n{instance.sensor_log}\n\n"
        f"=== DATENBANKSCHEMA ===\n{instance.database.schema_description_de}\n\n"
        f"=== DIAGNOSEFUNKTION ===\n"
        f"Signatur: {instance.diagnostic.signature}\n"
        f"{instance.diagnostic.description_de}\n"
        f"Eingabebezeichner für run_diagnostic_python: "
        f"{instance.diagnostic.public_input_id}\n\n"
        f"=== SCHEMA DES ABSCHLUSSBERICHTS ===\n{schema_text}"
    )

    return [
        ModelMessage(role="system", content=SYSTEM_PROMPT_DE),
        ModelMessage(role="user", content=observation),
    ]


def run_agent_loop(
    instance: WorkflowInstance,
    client: ModelClient,
    toolbox: ToolBox,
    trace: ExecutionTrace,
    *,
    limits: ExecutionLimits | None = None,
) -> LoopResult:
    """Drive one candidate through one instance.

    Args:
        instance: the problem.
        client: the served candidate.
        toolbox: the tool surface, already bound to this instance.
        trace: filled in as the run proceeds; the caller owns it.

    Returns:
        Why the loop ended. The trace holds everything else.
    """
    limits = limits or ExecutionLimits()
    deadline = Deadline(limits.instance_timeout_seconds)

    messages = build_initial_messages(instance)
    seed = instance.seed & 0x7FFFFFFF

    input_tokens = 0
    output_tokens = 0
    turn = 0
    tool_calls_made = 0

    try:
        while turn < limits.max_turns:
            deadline.check()
            turn += 1

            remaining_tokens = limits.max_output_tokens - output_tokens
            if remaining_tokens <= 0:
                return LoopResult("token_limit", turn - 1, input_tokens, output_tokens)

            reply = client.complete(
                messages,
                toolbox_schemas(),
                seed=seed,
                max_tokens=min(2048, remaining_tokens),
            )

            if not reply.ok:
                trace.harness_error = reply.error
                return LoopResult("model_error", turn - 1, input_tokens, output_tokens)

            input_tokens += reply.input_tokens
            output_tokens += reply.output_tokens

            messages.append(
                ModelMessage(
                    role="assistant",
                    content=reply.content,
                    tool_calls=reply.tool_calls,
                )
            )

            if not reply.tool_calls:
                # No tool call means the candidate answered in prose. The
                # contract says every step goes through a tool, so this is a
                # wasted turn; the loop tells it so and continues.
                messages.append(
                    ModelMessage(
                        role="user",
                        content=(
                            "Es wurde kein Werkzeug aufgerufen. Führe den nächsten Schritt "
                            "über ein Werkzeug aus."
                        ),
                    )
                )
                continue

            # Only the first few calls of a reply are honoured. A model may emit
            # any number of tool calls in one turn, and executing all of them
            # would make the turn budget meaningless — fifty calls for the price
            # of one. Surplus calls are refused with an explanation rather than
            # dropped, so the candidate can retry them on its next turn.
            honoured = reply.tool_calls[: limits.max_tool_calls_per_turn]
            surplus = len(reply.tool_calls) - len(honoured)

            for raw_call in honoured:
                if tool_calls_made >= limits.max_tool_calls_total:
                    return LoopResult("turn_limit", turn, input_tokens, output_tokens)

                # Between calls, not merely between turns: a single turn holding
                # several slow calls could otherwise run far past the budget.
                if deadline.expired:
                    return LoopResult("timeout", turn, input_tokens, output_tokens)

                name, arguments, parse_error = parse_tool_call(raw_call)
                call_id = str(raw_call.get("id") or f"call_{turn}")

                if parse_error is not None:
                    messages.append(
                        ModelMessage(
                            role="tool",
                            tool_call_id=call_id,
                            name=name or "unknown",
                            content=json.dumps({"error": parse_error}, ensure_ascii=False),
                        )
                    )
                    continue

                tool_calls_made += 1
                call = toolbox.dispatch(turn, name, arguments)
                messages.append(
                    ModelMessage(
                        role="tool",
                        tool_call_id=call_id,
                        name=name,
                        content=_render_result(call.result),
                    )
                )

                if trace.harness_error is not None:
                    return LoopResult("harness_error", turn, input_tokens, output_tokens)

                if name == T.TERMINAL_TOOL and toolbox.submitted:
                    return LoopResult("submitted", turn, input_tokens, output_tokens)

            if surplus > 0:
                log.info(
                    "%s: refused %d surplus tool call(s) in turn %d",
                    instance.instance_id,
                    surplus,
                    turn,
                )
                messages.append(
                    ModelMessage(
                        role="user",
                        content=(
                            f"Es wurden {surplus} weitere Werkzeugaufrufe in diesem Schritt "
                            f"nicht ausgeführt. Pro Schritt werden höchstens "
                            f"{limits.max_tool_calls_per_turn} Aufrufe bearbeitet."
                        ),
                    )
                )

        return LoopResult("turn_limit", turn, input_tokens, output_tokens)

    except LimitExceeded as exc:
        log.info("instance %s stopped: %s", instance.instance_id, exc)
        return LoopResult("timeout", turn, input_tokens, output_tokens)


def toolbox_schemas() -> list[dict]:
    """The tool schemas, identical for every candidate and every instance."""
    return T.TOOL_SCHEMAS


def _render_result(result: object) -> str:
    """Serialise a tool result for the conversation.

    Truncation matters: a candidate that dumps a whole table would otherwise
    push the manual out of its own context and fail for a reason that has
    nothing to do with capability composition. The cut is announced, so the
    candidate can narrow its query rather than guess why the answer looks short.
    """
    try:
        text = json.dumps(result, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(result)

    ceiling = 24_000
    if len(text) > ceiling:
        return text[:ceiling] + f"\n… [Ausgabe nach {ceiling} Zeichen gekürzt]"
    return text


def measure_turns(trace: ExecutionTrace) -> int:
    """Distinct turns on which the candidate called at least one tool."""
    return len({call.turn for call in trace.calls})


def within_agent_limits(trace: ExecutionTrace) -> tuple[bool, str]:
    """Whether a completed run stayed inside the published agent limits."""
    if trace.turns_used > C.MAX_AGENT_TURNS:
        return False, f"{trace.turns_used} turns exceeds the limit of {C.MAX_AGENT_TURNS}"
    if trace.output_tokens > C.MAX_OUTPUT_TOKENS:
        return (
            False,
            f"{trace.output_tokens} output tokens exceeds the limit of {C.MAX_OUTPUT_TOKENS}",
        )
    return True, "within agent limits"
