"""Per-instance sandbox orchestration.

One instance, one sandbox: stand up the tool services, run the fixed agent loop
against the served candidate, capture the final state, run the hidden diagnostic
cases, score the trace deterministically, tear everything down.

The services are torn down unconditionally. An instance that crashed halfway
still releases its database schema and its temporary files, because a run that
leaks state would contaminate whichever candidate is evaluated next — and a
contaminated comparison is worse than a missing one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from capability_subnet.common.schemas import InstanceResult, StageResult
from capability_subnet.common.trace import ExecutionTrace
from capability_subnet.sandbox.agent_runner import measure_turns, run_agent_loop
from capability_subnet.sandbox.db_tool import open_database
from capability_subnet.sandbox.limits import ExecutionLimits
from capability_subnet.sandbox.model_client import ModelClient
from capability_subnet.sandbox.python_runner import PythonRunner
from capability_subnet.sandbox.tools import ToolBox
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import WorkflowInstance
from capability_subnet.workflows.industrial_maintenance_de_v1.scoring import score_instance

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SandboxConfig:
    """How one sandbox is wired up."""

    limits: ExecutionLimits = ExecutionLimits()
    #: PostgreSQL DSN for the hidden snapshot. Without one the snapshot is served
    #: from an in-memory SQLite database, which is what local runs use.
    postgres_dsn: str | None = None
    #: Peak GPU memory observed while serving this candidate, reported by the
    #: executor. ``None`` when the counter was unavailable — carried through as
    #: None so nothing downstream mistakes "unmeasured" for "used nothing".
    peak_vram_gb: float | None = None


@dataclass(slots=True)
class SandboxOutcome:
    """Everything one instance produced."""

    result: InstanceResult
    trace: ExecutionTrace

    @property
    def scored(self) -> bool:
        return self.result.is_valid_sample


def run_instance(
    instance: WorkflowInstance,
    client: ModelClient,
    *,
    config: SandboxConfig | None = None,
) -> SandboxOutcome:
    """Run and score one hidden workflow instance.

    Returns:
        The sample row and the trace it was derived from. A row whose ``error``
        is set records a harness failure and must be excluded from scoring, not
        counted against the candidate.
    """
    config = config or SandboxConfig()
    started = time.monotonic()

    trace = ExecutionTrace(
        instance_id=instance.instance_id,
        instance_seed=instance.seed,
        split=instance.split,
        peak_vram_gb=config.peak_vram_gb or 0.0,
    )

    database = None
    try:
        database = open_database(
            instance.database,
            dsn=config.postgres_dsn,
            schema=f"inst_{instance.seed:012d}",
            limits=config.limits,
        )
        runner = PythonRunner(config.limits)
        toolbox = ToolBox(instance, database, runner, trace, config.limits)

        loop = run_agent_loop(instance, client, toolbox, trace, limits=config.limits)

        trace.stop_reason = loop.stop_reason  # type: ignore[assignment]
        trace.input_tokens = loop.input_tokens
        trace.output_tokens = loop.output_tokens
        trace.turns_used = max(loop.turns_used, measure_turns(trace))

        toolbox.finalise()
        if trace.is_scorable:
            toolbox.run_hidden_diagnostics()

    except Exception as exc:  # noqa: BLE001
        log.exception("sandbox failed for %s", instance.instance_id)
        trace.harness_error = f"sandbox failure: {exc}"
    finally:
        trace.wall_seconds = time.monotonic() - started
        if database is not None:
            try:
                database.close()
            except Exception:  # noqa: BLE001
                log.warning("failed to close the database for %s", instance.instance_id)

    return SandboxOutcome(result=_build_result(instance, trace), trace=trace)


def _build_result(instance: WorkflowInstance, trace: ExecutionTrace) -> InstanceResult:
    """Turn a trace into a sample row."""
    row = InstanceResult(
        instance_id=instance.instance_id,
        instance_seed=instance.seed,
        split=instance.split,  # type: ignore[arg-type]
        turns_used=trace.turns_used,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        wall_seconds=trace.wall_seconds,
    )

    if not trace.is_scorable:
        row.error = trace.harness_error
        return row

    score = score_instance(instance, trace)
    row.stages = {
        name: StageResult(
            stage=name,
            score=outcome.score,
            passed=outcome.passed,
            detail=outcome.detail,
        )
        for name, outcome in score.stages.items()
    }
    row.final_state_correct = score.final_state_correct
    row.end_to_end_success = score.end_to_end_success
    row.critical_unsafe_actions = score.critical_unsafe_actions
    return row


def run_batch(
    instances: list[WorkflowInstance],
    client: ModelClient,
    *,
    config: SandboxConfig | None = None,
    on_result: Any = None,
) -> list[SandboxOutcome]:
    """Run a set of instances in sequence.

    Sequential on purpose. The GPU serving one candidate is assigned to one
    sandbox at a time, so overlapping instances would contend for it and make the
    latency measurement — which is a scored quantity — meaningless. Concurrency
    lives one level up, across candidates, not inside one candidate's run.
    """
    outcomes: list[SandboxOutcome] = []
    for index, instance in enumerate(instances, start=1):
        outcome = run_instance(instance, client, config=config)
        outcomes.append(outcome)
        if on_result is not None:
            on_result(index, len(instances), outcome)
        log.debug(
            "%s: %s (%.1fs, %d turns)",
            instance.instance_id,
            "success" if outcome.result.end_to_end_success else "failure",
            outcome.result.wall_seconds,
            outcome.result.turns_used,
        )
    return outcomes
