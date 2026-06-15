"""The generated-code stage.

The candidate writes one small, dependency-free function. It can run that
function against a public input through the tool to check itself; the score comes
from hidden cases it never sees.

That split is what makes the stage meaningful. Code that special-cases the one
visible input passes the tool call and fails the hidden cases, which is exactly
the distinction the stage is meant to draw.
"""

from __future__ import annotations

import random
from typing import Any

from capability_subnet.workflows.industrial_maintenance_de_v1.instance import DiagnosticTask

FUNCTION_NAME = "analyze"
SIGNATURE = "def analyze(readings: list[float], threshold: float) -> dict"

DESCRIPTION_DE = (
    "Implementiere die Funktion `analyze(readings, threshold)`. `readings` ist eine "
    "Liste von Messwerten in zeitlicher Reihenfolge, `threshold` die Warngrenze. "
    "Die Funktion gibt ein Dictionary mit genau drei Schlüsseln zurück:\n"
    "  - 'peak': der größte Messwert (float)\n"
    "  - 'exceedances': die Anzahl der Messwerte, die die Warngrenze echt überschreiten (int)\n"
    "  - 'longest_run': die Länge der längsten zusammenhängenden Folge von Messwerten, "
    "die die Warngrenze echt überschreiten (int)\n"
    "Verwende keine externen Bibliotheken. Bei leerer Eingabe gilt "
    "peak = 0.0, exceedances = 0, longest_run = 0."
)

#: Hidden cases generated per instance.
HIDDEN_CASE_COUNT = 6


def reference_analyze(readings: list[float], threshold: float) -> dict[str, Any]:
    """The reference implementation the hidden cases are computed from.

    Strict comparison against the threshold throughout: a value exactly equal to
    the warning limit is not an exceedance. The manual and the task description
    both say "echt überschreiten", so this is stated, not implied.
    """
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
            longest = max(longest, current)
        else:
            current = 0

    return {"peak": float(peak), "exceedances": int(exceedances), "longest_run": int(longest)}


def _random_series(rng: random.Random, threshold: float) -> list[float]:
    """A series with a mix of runs and isolated spikes around ``threshold``."""
    length = rng.randint(8, 30)
    values: list[float] = []
    index = 0
    while index < length:
        if rng.random() < 0.35:
            run = rng.randint(1, 5)
            for _ in range(min(run, length - index)):
                values.append(round(threshold * rng.uniform(1.01, 1.4), 2))
                index += 1
        else:
            values.append(round(threshold * rng.uniform(0.5, 0.999), 2))
            index += 1
    return values


def build_task(
    rng: random.Random,
    *,
    public_readings: tuple[float, ...],
    public_threshold: float,
    public_input_id: str,
) -> DiagnosticTask:
    """Build the task, its public input, and its hidden cases."""
    hidden: list[dict[str, Any]] = []

    for index in range(HIDDEN_CASE_COUNT):
        if index == 0:
            readings: list[float] = []
            threshold = round(public_threshold, 2)
        elif index == 1:
            # Every value below the threshold: exercises the zero-run path.
            threshold = round(public_threshold, 2)
            readings = [round(threshold * rng.uniform(0.3, 0.98), 2) for _ in range(12)]
        elif index == 2:
            # A value exactly at the threshold must not count.
            threshold = round(public_threshold, 2)
            readings = [threshold, threshold, round(threshold * 1.2, 2), threshold]
        else:
            threshold = round(public_threshold * rng.uniform(0.6, 1.5), 2)
            readings = _random_series(rng, threshold)

        hidden.append(
            {
                "case_id": f"case_{index}",
                "readings": readings,
                "threshold": threshold,
                "expected": reference_analyze(readings, threshold),
            }
        )

    return DiagnosticTask(
        function_name=FUNCTION_NAME,
        signature=SIGNATURE,
        description_de=DESCRIPTION_DE,
        public_input_id=public_input_id,
        public_readings=public_readings,
        public_threshold=public_threshold,
        hidden_cases=tuple(hidden),
    )


def results_match(produced: Any, expected: dict[str, Any], *, tolerance: float = 1e-6) -> bool:
    """Compare one hidden-case result with the reference.

    Floats are compared with a tolerance; the two counts must match exactly.
    Booleans are rejected where integers are expected, because ``True == 1`` in
    Python and a function returning booleans is not implementing the contract.
    """
    if not isinstance(produced, dict):
        return False
    if set(produced) != {"peak", "exceedances", "longest_run"}:
        return False

    peak = produced["peak"]
    if isinstance(peak, bool) or not isinstance(peak, (int, float)):
        return False
    if abs(float(peak) - float(expected["peak"])) > tolerance:
        return False

    for key in ("exceedances", "longest_run"):
        value = produced[key]
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if value != expected[key]:
            return False

    return True
