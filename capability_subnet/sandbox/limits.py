"""Execution limits.

Limits serve two purposes at once. They keep an evaluation bounded — a candidate
that loops forever must not stall the queue — and they *are* part of the
measurement: a package that needs fifteen turns where the champion needs six is
worse, and the turn limit is what turns that into a score rather than a bigger
bill.

Every limit here is published in the workflow contract. None of them are secret.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass

from capability_subnet.common import constants as C


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """The bounds one sandbox run operates under."""

    max_turns: int = C.MAX_AGENT_TURNS
    max_output_tokens: int = C.MAX_OUTPUT_TOKENS
    instance_timeout_seconds: float = C.SANDBOX_INSTANCE_TIMEOUT_SECONDS
    python_timeout_seconds: float = C.PYTHON_RUNNER_TIMEOUT_SECONDS
    sql_timeout_seconds: float = C.SQL_TIMEOUT_SECONDS

    #: Tool calls honoured within a single turn. A model may return many calls in
    #: one reply, and without this the turn budget is trivially evaded: fifty
    #: calls in one turn costs one turn. The cap makes the turn limit mean what
    #: it says.
    max_tool_calls_per_turn: int = 4

    #: Tool calls honoured across the whole instance. A backstop independent of
    #: how the calls are distributed across turns.
    max_tool_calls_total: int = 40

    #: Rows a single query may return. A candidate that dumps the whole table
    #: instead of aggregating gets truncated rather than starving the harness.
    max_sql_rows: int = 500

    #: Bytes of generated source the code tool accepts.
    max_code_bytes: int = 32 * 1024

    #: Address space ceiling for the isolated Python runner, in megabytes.
    python_memory_mb: int = 512

    #: CPU seconds for the isolated Python runner. Lower than the wall-clock
    #: timeout so a busy loop is killed by the kernel rather than waited out.
    python_cpu_seconds: int = 10


class LimitExceeded(Exception):
    """Raised when a run hits a limit. Not an error — an outcome."""

    def __init__(self, limit: str, detail: str = "") -> None:
        super().__init__(f"{limit} exceeded{': ' + detail if detail else ''}")
        self.limit = limit
        self.detail = detail


class Deadline:
    """Wall-clock budget for one instance.

    Checked between turns and before each tool call rather than enforced by a
    signal, so a run that overruns still produces a complete trace up to the
    point it stopped. A truncated trace is scoreable; a killed process is not.
    """

    __slots__ = ("_started", "_budget")

    def __init__(self, budget_seconds: float) -> None:
        self._started = time.monotonic()
        self._budget = budget_seconds

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    @property
    def remaining(self) -> float:
        return max(0.0, self._budget - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def check(self, what: str = "instance timeout") -> None:
        if self.expired:
            raise LimitExceeded(what, f"{self._budget:.0f}s budget consumed")


def apply_subprocess_limits(memory_mb: int, cpu_seconds: int) -> None:
    """Apply resource limits inside a freshly forked child.

    Passed as ``preexec_fn`` to the isolated Python runner. This is defence in
    depth, not the primary boundary — the primary boundary is the container the
    runner lives in, which has no network, a read-only root filesystem and no
    capabilities. These limits stop a merely runaway program from taking the host
    down with it before the container notices.
    """
    address_space = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    # No core dumps: a 512 MB core file per crashed submission would fill the
    # evaluation host within a window.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
