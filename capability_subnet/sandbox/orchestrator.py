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

from capability_subnet.common import constants as C
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
    runner: Any = None,
    concurrency: int = C.SANDBOX_BATCH_CONCURRENCY,
    timed_prefix: int = C.SANDBOX_LATENCY_SAMPLE,
) -> list[SandboxOutcome]:
    """Ask a package every instance in the draw, and time a prefix of them.

    ``runner`` is the workflow's own :func:`run_instance`, because workflows
    differ in how a package is *asked*, not only in what it is asked. Defaulting
    to this module's runner is a convenience for callers that know they are on
    the agent-loop workflow; anything driving a configured workflow must pass
    that workflow's runner, or it will ask every workflow the first one's way.

    Two phases, because accuracy and latency want opposite things from the GPU.

    The first ``timed_prefix`` instances run **one at a time**. Latency is a
    scored term and a hard gate, and a request that queued behind thirty others
    has a wall clock describing the batch rather than the package, so the timing
    has to come from somewhere uncontended. A deterministic prefix rather than a
    random sample, so an auditor replaying the window knows which rows carried
    it.

    Everything after that runs at ``concurrency``, and those rows are marked
    ``timed=False``. They score exactly as they always did — only the latency
    term and the p95 gate skip them. This is where the time goes: one at a time
    the card is decode-bound and idle between tokens of a single stream, and
    measured on this corpus batching at 32 is 14.7x faster.

    Results come back in instance order regardless of completion order, because
    a caller comparing two packages row by row must not see them permuted by
    which request happened to finish first.
    """
    from concurrent.futures import ThreadPoolExecutor

    call = runner or run_instance
    total = len(instances)
    outcomes: list[SandboxOutcome | None] = [None] * total
    completed = 0

    def record(index: int, outcome: SandboxOutcome, timed: bool) -> None:
        nonlocal completed
        # The row keeps its wall clock either way; the flag says whether the
        # number means anything about the package.
        outcome.result.timed = timed
        outcomes[index] = outcome
        completed += 1
        if on_result is not None:
            on_result(completed, total, outcome)
        log.debug(
            "%s: %s (%.1fs, %d turns%s)",
            instances[index].instance_id,
            "success" if outcome.result.end_to_end_success else "failure",
            outcome.result.wall_seconds,
            outcome.result.turns_used,
            "" if timed else ", batched",
        )

    head = max(0, min(timed_prefix, total))
    for index in range(head):
        record(index, call(instances[index], client, config=config), True)

    tail = list(range(head, total))
    if tail:
        workers = max(1, concurrency)
        if workers == 1:
            for index in tail:
                record(index, call(instances[index], client, config=config), True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                from concurrent.futures import as_completed

                futures = {
                    pool.submit(call, instances[i], client, config=config): i for i in tail
                }
                for future in as_completed(futures):
                    record(futures[future], future.result(), False)

    return [o for o in outcomes if o is not None]
