"""The fixed tool surface.

Seven tools, identical for every candidate and every instance. The set is fixed
because the workflow measures composition quality, not prompt engineering: if
candidates could see different tools, their scores would not be comparable.

Descriptions are in German to match the rest of the environment. They state what
each tool does and nothing about what the right answer is.
"""

from __future__ import annotations

from typing import Any

READ_MANUAL = "read_manual"
QUERY_MAINTENANCE_DB = "query_maintenance_db"
RUN_DIAGNOSTIC_PYTHON = "run_diagnostic_python"
SEARCH_INVENTORY = "search_inventory"
RESERVE_INVENTORY = "reserve_inventory"
CHECK_SAFETY_PLAN = "check_safety_plan"
SUBMIT_FINAL_JSON = "submit_final_json"

TOOL_NAMES: tuple[str, ...] = (
    READ_MANUAL,
    QUERY_MAINTENANCE_DB,
    RUN_DIAGNOSTIC_PYTHON,
    SEARCH_INVENTORY,
    RESERVE_INVENTORY,
    CHECK_SAFETY_PLAN,
    SUBMIT_FINAL_JSON,
)

#: The only tool that ends the run.
TERMINAL_TOOL = SUBMIT_FINAL_JSON


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": READ_MANUAL,
            "description": (
                "Liest einen Abschnitt des technischen Handbuchs. Akzeptiert eine "
                "Abschnittsnummer (z. B. '3.2') oder einen Suchbegriff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_or_query": {
                        "type": "string",
                        "description": "Abschnittsnummer oder Suchbegriff.",
                    }
                },
                "required": ["section_or_query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": QUERY_MAINTENANCE_DB,
            "description": (
                "Führt eine lesende SQL-Abfrage gegen die Wartungsdatenbank aus und "
                "gibt Spaltennamen und Zeilen zurück. Nur SELECT ist zulässig."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Vollständige SELECT-Anweisung."}
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": RUN_DIAGNOSTIC_PYTHON,
            "description": (
                "Führt eingereichten Python-Code in einer isolierten Umgebung aus. Der "
                "Code muss die geforderte Funktion definieren. Die Funktion wird mit der "
                "angegebenen Eingabe aufgerufen und ihr Rückgabewert zurückgeliefert."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python-Quelltext ohne externe Abhängigkeiten.",
                    },
                    "input_id": {
                        "type": "string",
                        "description": "Bezeichner der Eingabe, z. B. das Sensorfenster.",
                    },
                },
                "required": ["code", "input_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SEARCH_INVENTORY,
            "description": "Sucht eine Teilenummer im Lagerbestand.",
            "parameters": {
                "type": "object",
                "properties": {"part_number": {"type": "string", "description": "Teilenummer."}},
                "required": ["part_number"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": RESERVE_INVENTORY,
            "description": "Reserviert eine Menge einer Teilenummer im Lagerbestand.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_number": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1},
                },
                "required": ["part_number", "quantity"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CHECK_SAFETY_PLAN,
            "description": (
                "Legt einen Sicherungsplan der Prüfstelle vor. Gibt zurück, ob die "
                "Freigabe erteilt wird, welche Maßnahmen fehlen und welche "
                "Arbeitsschritte unzulässig sind."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Vorgesehene Sicherungsmaßnahmen.",
                            },
                            "actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Geplante Arbeitsschritte.",
                            },
                        },
                        "required": ["steps"],
                    }
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SUBMIT_FINAL_JSON,
            "description": (
                "Übermittelt den Abschlussbericht und beendet den Vorgang. Darf genau "
                "einmal aufgerufen werden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "description": "Abschlussbericht nach vorgegebenem Schema.",
                    }
                },
                "required": ["payload"],
                "additionalProperties": False,
            },
        },
    },
]


def schema_for(name: str) -> dict[str, Any]:
    for schema in TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return schema
    raise KeyError(f"no tool named {name!r}")
