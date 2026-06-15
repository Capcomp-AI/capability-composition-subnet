"""The final report schema.

The last stage is judged in two steps: the payload must validate against this
schema, and then every value in it must match the truth. Splitting the two gives
a useful signal — a package that produces well-formed JSON with wrong numbers
has a different problem from one that cannot produce JSON at all, and the stage
scores reflect that.
"""

from __future__ import annotations

from typing import Any

from capability_subnet.workflows.industrial_maintenance_de_v1.instance import RECOMMENDATIONS

FINAL_PAYLOAD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Instandsetzungsbericht",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "machine_id",
        "fault_code",
        "component",
        "manual_reference",
        "maintenance_events_last_24m",
        "total_downtime_minutes",
        "diagnostic",
        "replacement_part",
        "reserved_quantity",
        "safety_steps",
        "recommendation",
    ],
    "properties": {
        "machine_id": {"type": "string", "minLength": 1},
        "fault_code": {"type": "string", "minLength": 1},
        "component": {"type": "string", "minLength": 1},
        "manual_reference": {
            "type": "string",
            "minLength": 1,
            "description": "Prüfanweisungsnummer aus dem Abschnitt Störungsdiagnose.",
        },
        "maintenance_events_last_24m": {"type": "integer", "minimum": 0},
        "total_downtime_minutes": {"type": "integer", "minimum": 0},
        "diagnostic": {
            "type": "object",
            "additionalProperties": False,
            "required": ["peak", "exceedances", "longest_run"],
            "properties": {
                "peak": {"type": "number"},
                "exceedances": {"type": "integer", "minimum": 0},
                "longest_run": {"type": "integer", "minimum": 0},
            },
        },
        "replacement_part": {"type": "string", "minLength": 1},
        "reserved_quantity": {"type": "integer", "minimum": 1},
        "safety_steps": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "recommendation": {"type": "string", "enum": list(RECOMMENDATIONS)},
    },
}


def validate_payload(payload: Any) -> list[str]:
    """Validate a final payload against the schema.

    Returns:
        A list of human-readable problems, empty when the payload is valid. Every
        error is collected rather than only the first, so the trace shows the
        whole shape of the failure.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover - jsonschema is a hard dependency
        return _validate_without_jsonschema(payload)

    validator = Draft202012Validator(FINAL_PAYLOAD_SCHEMA)
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    ]


def _validate_without_jsonschema(payload: Any) -> list[str]:
    """Minimal structural fallback.

    Only reached if the optional validator is unavailable. It checks presence and
    coarse types so the stage still produces a defensible verdict rather than
    silently passing everything.
    """
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["<root>: expected an object"]

    for field in FINAL_PAYLOAD_SCHEMA["required"]:
        if field not in payload:
            problems.append(f"{field}: required property is missing")

    extra = sorted(set(payload) - set(FINAL_PAYLOAD_SCHEMA["properties"]))
    problems.extend(f"{field}: unexpected property" for field in extra)

    if payload.get("recommendation") not in RECOMMENDATIONS and "recommendation" in payload:
        problems.append("recommendation: not one of the permitted values")

    diagnostic = payload.get("diagnostic")
    if "diagnostic" in payload and not isinstance(diagnostic, dict):
        problems.append("diagnostic: expected an object")

    steps = payload.get("safety_steps")
    if "safety_steps" in payload and not isinstance(steps, list):
        problems.append("safety_steps: expected an array")

    return problems
