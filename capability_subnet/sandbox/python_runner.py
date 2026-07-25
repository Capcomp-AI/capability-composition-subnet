"""The generated-code tool.

Candidate-written Python is the only thing in this system that executes arbitrary
instructions, so it runs at the far end of two boundaries: a container with no
network, a read-only root filesystem and no capabilities, and inside that, a
fresh interpreter process with hard resource limits that never sees the parent's
memory.

The runner is used twice per instance. During the agent loop it answers the
candidate's own tool calls against a public input. After the loop ends it runs
the same submitted code against the hidden cases — which the agent container
never held — and those results are what the stage is scored on.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from capability_subnet.sandbox.limits import ExecutionLimits, apply_subprocess_limits

log = logging.getLogger(__name__)

#: Executed inside the child. Reads a job from stdin, executes the submitted
#: source in a namespace of its own, calls the requested function once per case,
#: and writes results to stdout as JSON.
#:
#: The submitted source is executed with an empty ``__builtins__`` override of
#: nothing — full builtins are available on purpose. Restricting builtins is
#: security theatre inside an already-isolated container, and it would silently
#: fail correct submissions that use ``sum`` or ``max``.
_CHILD_PROGRAM = r"""
import json
import sys

def main() -> int:
    job = json.loads(sys.stdin.read())
    source = job["source"]
    function_name = job["function_name"]
    cases = job["cases"]

    namespace = {"__name__": "candidate_diagnostic"}
    try:
        compiled = compile(source, "<candidate>", "exec")
        exec(compiled, namespace)
    except BaseException as exc:
        print(json.dumps({"fatal": f"{type(exc).__name__}: {exc}"}))
        return 0

    function = namespace.get(function_name)
    if not callable(function):
        print(json.dumps({"fatal": f"function {function_name!r} is not defined"}))
        return 0

    results = []
    for case in cases:
        entry = {"case_id": case["case_id"], "hidden": case.get("hidden", False)}
        try:
            output = function(list(case["readings"]), case["threshold"])
            json.dumps(output)  # reject values that cannot cross the boundary
            entry["output"] = output
        except BaseException as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)

    print(json.dumps({"results": results}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


#: Negative return codes that mean a resource limit fired rather than the code
#: crashing. ``subprocess`` reports a signal death as the negated signal number.
_RESOURCE_LIMIT_SIGNALS: dict[int, str] = {
    -9: "Zeitlimit oder Speichergrenze überschritten (Prozess beendet)",
    -24: "CPU-Zeitlimit überschritten",
    -25: "Dateigrößenlimit überschritten",
    -11: "Speicherzugriffsfehler oder Speichergrenze überschritten",
}


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What one batch of cases produced."""

    results: list[dict[str, Any]]
    fatal: str | None = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.fatal is None and not self.timed_out


class PythonRunner:
    """Runs candidate code in an isolated interpreter."""

    def __init__(self, limits: ExecutionLimits | None = None) -> None:
        self._limits = limits or ExecutionLimits()

    def run(
        self,
        source: str,
        function_name: str,
        cases: list[dict[str, Any]],
    ) -> RunOutcome:
        """Execute ``source`` and call ``function_name`` once per case.

        Every failure mode — syntax error, missing function, exception inside a
        case, timeout, crash — comes back as data rather than an exception. The
        candidate is allowed to submit broken code; that is a score, not an
        incident.
        """
        if not isinstance(source, str) or not source.strip():
            return RunOutcome([], fatal="kein Quelltext übermittelt")
        if len(source.encode("utf-8")) > self._limits.max_code_bytes:
            return RunOutcome(
                [], fatal=f"Quelltext überschreitet {self._limits.max_code_bytes} Bytes"
            )

        job = json.dumps({"source": source, "function_name": function_name, "cases": cases})

        try:
            completed = subprocess.run(
                # -I: isolated mode. Ignores PYTHON* environment variables and
                # keeps the current directory off sys.path, so submitted code
                # cannot import anything the harness happens to have lying next
                # to it.
                [sys.executable, "-I", "-S", "-c", _CHILD_PROGRAM],
                input=job,
                capture_output=True,
                text=True,
                timeout=self._limits.python_timeout_seconds,
                preexec_fn=lambda: apply_subprocess_limits(
                    self._limits.python_memory_mb, self._limits.python_cpu_seconds
                ),
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
            )
        except subprocess.TimeoutExpired:
            return RunOutcome([], fatal="Zeitlimit überschritten", timed_out=True)
        except OSError as exc:
            log.exception("could not start the isolated interpreter")
            return RunOutcome([], fatal=f"Ausführungsumgebung nicht verfügbar: {exc}")

        if completed.returncode != 0 and not completed.stdout.strip():
            # A negative return code means the kernel killed the process, which
            # is how the CPU and memory limits actually fire — they trip before
            # the wall-clock timeout does. Reporting that as a generic crash
            # would tell a miner their code was broken when it was merely too
            # expensive.
            if completed.returncode in _RESOURCE_LIMIT_SIGNALS:
                return RunOutcome(
                    [],
                    fatal=_RESOURCE_LIMIT_SIGNALS[completed.returncode],
                    timed_out=True,
                )

            detail = (completed.stderr or "").strip().splitlines()
            return RunOutcome(
                [],
                fatal=f"Prozess mit Code {completed.returncode} beendet: "
                f"{detail[-1] if detail else 'keine Ausgabe'}",
            )

        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return RunOutcome([], fatal="Ausführung lieferte keine verwertbare Ausgabe")

        if "fatal" in payload:
            return RunOutcome([], fatal=str(payload["fatal"]))
        return RunOutcome(list(payload.get("results", [])))


def public_case(input_id: str, readings: list[float], threshold: float) -> dict[str, Any]:
    """The visible case the candidate may run through the tool."""
    return {
        "case_id": input_id,
        "readings": list(readings),
        "threshold": threshold,
        "hidden": False,
    }


def hidden_cases(cases: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """The scoring cases, stripped of their expected outputs.

    The expected values stay with the scorer. Sending them into the runner would
    put them one crash-dump away from the trace, and the trace is published.
    """
    return [
        {
            "case_id": case["case_id"],
            "readings": list(case["readings"]),
            "threshold": case["threshold"],
            "hidden": True,
        }
        for case in cases
    ]
