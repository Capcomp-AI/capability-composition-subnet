"""Maintenance history snapshot.

The SQL stage is judged by *executing* the candidate's statement against this
snapshot and comparing the result set with the one the reference query produces.
Nothing about the text of the query is scored — only what it returns. A candidate
is free to phrase the join however it likes.

The DDL is deliberately restricted to types that mean the same thing in
PostgreSQL and SQLite: the hidden snapshot runs on PostgreSQL, while the public
pack ships as a SQLite file so miners can debug without standing up a server.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

from capability_subnet.workflows.industrial_maintenance_de_v1.instance import DatabaseSnapshot
from capability_subnet.workflows.industrial_maintenance_de_v1.machines import (
    TECHNICIANS_DE,
    Component,
    MachineInstance,
)

#: Column names in the default schema.
DEFAULT_COLUMNS: dict[str, tuple[str, ...]] = {
    "machines": ("machine_id", "model", "installed_on", "location"),
    "maintenance_history": (
        "event_id",
        "machine_id",
        "event_date",
        "component",
        "fault_code",
        "technician",
        "downtime_minutes",
        "resolved",
    ),
    "parts": ("part_number", "description", "component", "unit_price_eur", "obsolete"),
}

#: Column renames applied by out-of-distribution instances. The schema
#: description handed to the candidate is regenerated from the aliases, so the
#: information is there — it just is not where a memorised query expects it.
COLUMN_ALIASES: dict[str, dict[str, str]] = {
    "maintenance_history": {
        "event_id": "ereignis_id",
        "event_date": "ereignis_datum",
        "component": "bauteil",
        "fault_code": "fehlercode",
        "technician": "techniker",
        "downtime_minutes": "ausfallzeit_min",
        "resolved": "erledigt",
    },
    "machines": {
        "model": "baureihe",
        "installed_on": "inbetriebnahme",
        "location": "aufstellort",
    },
    "parts": {
        "description": "bezeichnung",
        "component": "bauteil",
        "unit_price_eur": "einzelpreis_eur",
        "obsolete": "gesperrt",
    },
}

#: Run the reference question asks about.
LOOKBACK_MONTHS = 24


def _apply_aliases(table: str, columns: tuple[str, ...], use_aliases: bool) -> tuple[str, ...]:
    if not use_aliases:
        return columns
    aliases = COLUMN_ALIASES.get(table, {})
    return tuple(aliases.get(column, column) for column in columns)


def _column_types(table: str) -> dict[str, str]:
    integers = {"event_id", "downtime_minutes", "resolved", "obsolete"}
    reals = {"unit_price_eur"}
    return {
        column: ("INTEGER" if column in integers else "REAL" if column in reals else "TEXT")
        for column in DEFAULT_COLUMNS[table]
    }


def build_ddl(columns: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Portable DDL for the three tables."""
    statements: list[str] = []
    for table in ("machines", "maintenance_history", "parts"):
        types = _column_types(table)
        defaults = DEFAULT_COLUMNS[table]
        actual = columns[table]
        rendered = ", ".join(
            f"{name} {types[default]}" for name, default in zip(actual, defaults, strict=True)
        )
        statements.append(f"CREATE TABLE {table} ({rendered})")
    return tuple(statements)


def build_snapshot(
    rng: random.Random,
    machine: MachineInstance,
    *,
    component: Component,
    fault_code: str,
    reference_date: date,
    use_column_aliases: bool = False,
    add_sibling_rows: bool = False,
) -> tuple[DatabaseSnapshot, tuple[tuple[Any, ...], ...], str, tuple[str, ...]]:
    """Generate the snapshot together with the reference answer.

    Args:
        component: the faulty component. Passed in rather than looked up from the
            fault code, because out-of-distribution instances rewrite the code's
            surface form and a reverse lookup would fail on exactly those.
        fault_code: the code as it appears in the manual and the database, which
            for a mutated instance is not the component's canonical code.

    Returns:
        ``(snapshot, expected_rows, question_de, expected_columns)``. The expected
        rows are computed here in Python rather than by running a reference query,
        so an error in the reference SQL cannot silently become the truth.
    """
    machine_type = machine.machine_type
    columns = {
        table: _apply_aliases(table, default, use_column_aliases)
        for table, default in DEFAULT_COLUMNS.items()
    }

    machines_rows: list[tuple[Any, ...]] = [
        (
            machine.machine_id,
            machine_type.model_de,
            machine.installed_on,
            machine.location_de,
        )
    ]

    # A sibling machine of the same model gives the query something to exclude.
    sibling_id = f"{machine.machine_id.rsplit('-', 1)[0]}-{rng.randrange(1000, 9999)}"
    if sibling_id == machine.machine_id:
        sibling_id = f"{sibling_id}X"
    machines_rows.append(
        (sibling_id, machine_type.model_de, machine.installed_on, machine.location_de)
    )

    history_rows: list[tuple[Any, ...]] = []
    expected_count = 0
    expected_downtime = 0
    event_id = 1

    cutoff = reference_date - timedelta(days=LOOKBACK_MONTHS * 30)

    def add_event(
        machine_id: str, event_date: date, component_label: str, code: str, downtime: int
    ) -> None:
        nonlocal event_id
        history_rows.append(
            (
                event_id,
                machine_id,
                event_date.isoformat(),
                component_label,
                code,
                rng.choice(TECHNICIANS_DE),
                downtime,
                1 if rng.random() < 0.85 else 0,
            )
        )
        event_id += 1

    # Matching events inside the run — these are the answer.
    for _ in range(rng.randint(2, 6)):
        days_ago = rng.randrange(5, LOOKBACK_MONTHS * 30 - 5)
        event_date = reference_date - timedelta(days=days_ago)
        downtime = rng.randrange(20, 480)
        add_event(machine.machine_id, event_date, component.label_de, fault_code, downtime)
        expected_count += 1
        expected_downtime += downtime

    # Same fault code, same machine, but outside the run.
    for _ in range(rng.randint(1, 3)):
        days_ago = rng.randrange(LOOKBACK_MONTHS * 30 + 30, LOOKBACK_MONTHS * 30 + 900)
        add_event(
            machine.machine_id,
            reference_date - timedelta(days=days_ago),
            component.label_de,
            fault_code,
            rng.randrange(20, 480),
        )

    # Different fault codes on the same machine, inside the run.
    others = [c for c in machine_type.components if c.key != component.key]
    for _ in range(rng.randint(2, 5)):
        other = rng.choice(others)
        add_event(
            machine.machine_id,
            reference_date - timedelta(days=rng.randrange(5, LOOKBACK_MONTHS * 30 - 5)),
            other.label_de,
            other.fault_code,
            rng.randrange(20, 480),
        )

    # The same fault code on the sibling machine, inside the run. A query that
    # forgets to filter by machine picks these up and gets the wrong count.
    sibling_events = rng.randint(2, 5) if add_sibling_rows else rng.randint(1, 3)
    for _ in range(sibling_events):
        add_event(
            sibling_id,
            reference_date - timedelta(days=rng.randrange(5, LOOKBACK_MONTHS * 30 - 5)),
            component.label_de,
            fault_code,
            rng.randrange(20, 480),
        )

    rng.shuffle(history_rows)
    history_rows = [(index + 1, *row[1:]) for index, row in enumerate(history_rows)]

    parts_rows: list[tuple[Any, ...]] = []
    for component in machine_type.components:
        parts_rows.append(
            (
                component.part_number,
                component.part_description_de,
                component.label_de,
                component.unit_price_eur,
                0,
            )
        )
        # A superseded number that looks plausible but must not be reserved.
        legacy = f"{component.part_number[0]}-{int(component.part_number.split('-')[1]) - 1:05d}"
        parts_rows.append(
            (
                legacy,
                f"{component.part_description_de} (Vorgängerausführung)",
                component.label_de,
                round(component.unit_price_eur * 0.92, 2),
                1,
            )
        )

    schema_description = _describe_schema(columns, cutoff, reference_date)

    snapshot = DatabaseSnapshot(
        schema_description_de=schema_description,
        ddl=build_ddl(columns),
        rows={
            "machines": tuple(machines_rows),
            "maintenance_history": tuple(history_rows),
            "parts": tuple(parts_rows),
        },
        columns=columns,
    )

    # The question deliberately does not name the fault code. Naming it would
    # hand the candidate the answer to the fault-extraction stage in the opening
    # prompt, and the SQL stage would no longer depend on anything earlier —
    # which is precisely what distinguishes a workflow from a list of
    # independent benchmark questions.
    question = (
        f"Wie viele Wartungsereignisse mit dem von dir ermittelten Fehlercode wurden für "
        f"die Anlage {machine.machine_id} im Zeitraum vom {cutoff.isoformat()} bis "
        f"{reference_date.isoformat()} erfasst, und wie hoch ist die Summe der zugehörigen "
        f"Ausfallzeiten in Minuten? Liefere genau eine Zeile mit zwei Spalten: Anzahl, "
        f"Summe der Ausfallzeit."
    )

    return (
        snapshot,
        ((expected_count, expected_downtime),),
        question,
        ("anzahl", "ausfallzeit_gesamt"),
    )


def _describe_schema(
    columns: dict[str, tuple[str, ...]], cutoff: date, reference_date: date
) -> str:
    """The schema description handed to the candidate."""
    lines = [
        "Schema der Wartungsdatenbank (nur lesender Zugriff):",
        "",
    ]
    for table in ("machines", "maintenance_history", "parts"):
        lines.append(f"Tabelle {table}({', '.join(columns[table])})")
    lines.extend(
        [
            "",
            f"Der Auswertungszeitraum ist {cutoff.isoformat()} bis {reference_date.isoformat()} "
            "(einschließlich).",
            "Datumsspalten sind als Text im Format YYYY-MM-DD abgelegt und lassen sich "
            "lexikographisch vergleichen.",
        ]
    )
    return "\n".join(lines)


def reference_query(
    columns: dict[str, tuple[str, ...]],
    machine_id: str,
    fault_code: str,
    cutoff: date,
    reference_date: date,
) -> str:
    """A correct query for the question.

    Published in the public pack as a worked example. It is never used to compute
    the truth — the truth is derived directly from the generated rows — so a bug
    here can only mislead a miner, never mis-score one.
    """
    history = columns["maintenance_history"]
    default = DEFAULT_COLUMNS["maintenance_history"]
    lookup = dict(zip(default, history, strict=True))
    return (
        f"SELECT COUNT(*) AS anzahl, "
        f"COALESCE(SUM({lookup['downtime_minutes']}), 0) AS ausfallzeit_gesamt "
        f"FROM maintenance_history "
        f"WHERE {lookup['machine_id']} = '{machine_id}' "
        f"AND {lookup['fault_code']} = '{fault_code}' "
        f"AND {lookup['event_date']} BETWEEN '{cutoff.isoformat()}' "
        f"AND '{reference_date.isoformat()}'"
    )
