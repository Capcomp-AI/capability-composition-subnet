"""Asking one package one question.

The V1 workflow drives a twelve-turn loop over seven tool services and needs a
sandbox to do it. This one sends a message and reads the reply, so it needs
none — no database, no code runner, no tool surface, nothing the candidate could
reach that is not the question itself.

That is worth more than the simplicity. A single-turn arena has no state for a
candidate to corrupt and no tool for it to abuse, so the whole class of sandbox
escapes does not arise; and the run is cheap enough that a statistically honest
sample size is affordable, which is the constraint that actually binds.
"""

from __future__ import annotations

import logging
import time

from capability_subnet.common.schemas import InstanceResult, StageResult
from capability_subnet.sandbox.orchestrator import SandboxOutcome

log = logging.getLogger(__name__)

#: Room to reason before answering, without room to ramble indefinitely.
#: A package that needs more than this has failed the instruction, not run out
#: of space — a collapsed merge can spend several times
#: the base model's tokens and answering nothing.
MAX_OUTPUT_TOKENS = 1024


class LogicTrace:
    """What one answered instance produced.

    Deliberately small: the reply, the cost, and whether the harness itself
    failed. A trace is what an auditor re-scores from, so it holds exactly what
    the scorer reads and nothing that would let a published record be re-derived
    into something else.
    """

    __slots__ = ("harness_error", "input_tokens", "output_tokens", "reply", "wall_seconds")

    def __init__(self) -> None:
        self.reply: str | None = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_seconds = 0.0
        self.harness_error: str | None = None

    @property
    def is_scorable(self) -> bool:
        return self.harness_error is None and self.reply is not None

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": self.wall_seconds,
            "harness_error": self.harness_error,
        }


def run_instance(instance, client, *, config=None) -> SandboxOutcome:
    """Ask one question and score the answer.

    A serving failure is recorded as a harness error rather than a wrong answer.
    The distinction is the same one the engine makes everywhere: an endpoint that
    did not respond has told us nothing about the candidate, and scoring it zero
    would charge a miner for the operator's outage.
    """
    from capability_subnet.sandbox.model_client import ModelMessage
    from capability_subnet.workflows.lora_merger_logic_v1.scoring import score_instance

    del config  # no sandbox: this workflow gives the candidate nothing to reach
    trace = LogicTrace()
    started = time.monotonic()

    try:
        reply = client.complete(
            [ModelMessage(role="user", content=instance.question)],
            [],
            seed=instance.seed & 0x7FFFFFFF,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        if not reply.ok:
            trace.harness_error = reply.error
        else:
            trace.reply = reply.content or ""
            trace.input_tokens = reply.input_tokens
            trace.output_tokens = reply.output_tokens
    except Exception as exc:  # noqa: BLE001 - one instance must not end a batch
        log.exception("logic instance %s failed", instance.instance_id)
        trace.harness_error = f"harness failure: {exc}"
    finally:
        trace.wall_seconds = time.monotonic() - started

    row = InstanceResult(
        instance_id=instance.instance_id,
        instance_seed=instance.seed,
        split=instance.split,
        turns_used=1,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        wall_seconds=trace.wall_seconds,
    )

    if not trace.is_scorable:
        row.error = trace.harness_error
        return SandboxOutcome(result=row, trace=trace)

    outcomes = score_instance(instance, trace)
    row.stages = {
        name: StageResult(stage=name, score=o.score, passed=o.passed, detail=o.detail)
        for name, o in outcomes.items()
    }
    # One question, one answer: the instance is completed when it is answered
    # correctly. There is no partial credit to aggregate, which is what makes
    # the paired comparison over these rows straightforward.
    correct = outcomes[instance.task].passed
    row.final_state_correct = correct
    row.end_to_end_success = correct

    return SandboxOutcome(result=row, trace=trace)
