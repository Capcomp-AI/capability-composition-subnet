"""Instance generation.

One integer seed produces one complete, self-consistent instance: machine,
manual, sensor log, database, code task, inventory, policy and ground truth. The
same seed always produces the same instance, on any machine, which is what lets
the public pack be distributed as a seed rather than as data and lets a disputed
evaluation be replayed exactly.

Out-of-distribution instances apply named mutations on top of the same pipeline.
They are not "harder" instances — they are the same problem stated differently,
which is precisely what distinguishes a package that understood the task from one
that memorised its surface.
"""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import date, timedelta

from capability_subnet.workflows.industrial_maintenance_de_v1 import database as db
from capability_subnet.workflows.industrial_maintenance_de_v1 import diagnostics, sensor_log
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import (
    GroundTruth,
    WorkflowInstance,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.inventory import build_inventory
from capability_subnet.workflows.industrial_maintenance_de_v1.machines import (
    MACHINE_TYPES,
    sample_machine,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.manual import (
    build_manual,
    procedure_id,
    required_quantity,
)
from capability_subnet.workflows.industrial_maintenance_de_v1.policy import build_policy

#: Every mutation an out-of-distribution instance may apply.
OOD_MUTATIONS: tuple[str, ...] = (
    "unit_conversion",
    "synonym_terminology",
    "schema_alias",
    "fault_code_format",
    "sibling_noise",
)

#: How many mutations one instance gets.
OOD_MUTATION_COUNT = (2, 3)

#: Fixed reference date. Anchoring it makes the maintenance history and the SQL
#: run reproducible; deriving it from the wall clock would make yesterday's
#: instance unreplayable today.
REFERENCE_DATE = date(2026, 6, 30)


def _instance_id(seed: int, split: str) -> str:
    return f"{split}-{seed:012d}"


def generate_instance(seed: int, *, split: str = "hidden") -> WorkflowInstance:
    """Build one instance from ``seed``.

    Args:
        seed: any integer. The full instance is a pure function of it and ``split``.
        split: ``hidden``, ``ood`` or ``public``. Only ``ood`` applies mutations;
            the other two differ solely in which seed range they are drawn from.
    """
    rng = random.Random((seed << 3) ^ hash_split(split))

    mutations: tuple[str, ...] = ()
    if split == "ood":
        count = rng.randint(*OOD_MUTATION_COUNT)
        mutations = tuple(sorted(rng.sample(OOD_MUTATIONS, k=count)))

    use_alt_units = "unit_conversion" in mutations
    use_synonyms = "synonym_terminology" in mutations
    use_aliases = "schema_alias" in mutations
    alt_code_format = "fault_code_format" in mutations
    sibling_noise = "sibling_noise" in mutations

    machine_type = rng.choice(MACHINE_TYPES)
    machine = sample_machine(rng, machine_type=machine_type)
    component = rng.choice(machine_type.components)
    sensor = machine_type.sensor(component.sensor_key)

    critical_event = rng.random() < 0.45

    series = sensor_log.build_series(
        rng,
        machine,
        component.sensor_key,
        use_alt_units=use_alt_units,
        critical_event=critical_event,
    )
    faulty = series[component.sensor_key]

    log_date = (REFERENCE_DATE - timedelta(days=rng.randrange(0, 20))).isoformat()
    rendered_log = sensor_log.render_log(machine, series, log_date=log_date)

    manual = build_manual(
        machine,
        use_synonyms=use_synonyms,
        use_alt_units=use_alt_units,
        alt_fault_code_format=alt_code_format,
        rng=rng,
    )

    displayed_fault_code = (
        component.fault_code.replace("-", ".") if alt_code_format else component.fault_code
    )

    snapshot, expected_rows, sql_question, expected_columns = db.build_snapshot(
        rng,
        machine,
        component=component,
        fault_code=displayed_fault_code,
        reference_date=REFERENCE_DATE,
        use_column_aliases=use_aliases,
        add_sibling_rows=sibling_noise,
    )

    diagnostic = diagnostics.build_task(
        rng,
        public_readings=faulty.values,
        public_threshold=faulty.warn_threshold,
        public_input_id=f"sensorfenster_{component.sensor_key}",
    )

    inventory = build_inventory(rng, machine)
    policy = build_policy(rng, component)

    recommendation = (
        "sofortige_instandsetzung"
        if faulty.peak > faulty.critical_threshold
        else "geplante_wartung"
    )

    truth = GroundTruth(
        fault_code=displayed_fault_code,
        component_key=component.key,
        component_label_de=component.label_de,
        manual_reference=procedure_id(machine_type.manual_chapter, component.fault_code),
        sql_question_de=sql_question,
        sql_expected_rows=expected_rows,
        sql_expected_columns=expected_columns,
        diagnostic_expected=diagnostics.reference_analyze(
            list(faulty.values), faulty.warn_threshold
        ),
        required_part_number=component.part_number,
        required_quantity=required_quantity(component.key),
        required_safety_steps=policy.required_steps,
        forbidden_actions=policy.forbidden_actions,
        recommendation=recommendation,
        peak_value=faulty.peak,
        exceedance_count=faulty.exceedances,
        longest_run=faulty.longest_run,
        warn_threshold=faulty.warn_threshold,
        critical_threshold=faulty.critical_threshold,
    )

    prompt = _build_prompt(
        machine_id=machine.machine_id,
        model=machine_type.model_de,
        chapter=machine_type.manual_chapter,
        sql_question=sql_question,
        diagnostic_input_id=diagnostic.public_input_id,
        policy_id=policy.policy_id,
    )

    return WorkflowInstance(
        instance_id=_instance_id(seed, split),
        seed=seed,
        split=split,
        machine_id=machine.machine_id,
        machine_model_de=machine_type.model_de,
        machine_location_de=machine.location_de,
        sensor_key=sensor.key,
        sensor_label_de=sensor.label_de,
        sensor_unit=faulty.unit,
        task_prompt_de=prompt,
        manual=manual,
        sensor_log=rendered_log,
        database=snapshot,
        diagnostic=diagnostic,
        inventory=inventory,
        policy=policy,
        truth=truth,
        ood_mutations=mutations,
    )


def _build_prompt(
    *,
    machine_id: str,
    model: str,
    chapter: str,
    sql_question: str,
    diagnostic_input_id: str,
    policy_id: str,
) -> str:
    """The instruction the agent receives.

    It states the task and the required outputs but never the answers: no
    threshold, no fault code, no part number. Every one of those has to come out
    of the manual, the log or a tool.
    """
    return (
        f"Du bist Instandhaltungstechniker für die Anlage {machine_id} ({model}).\n"
        f"Im Anlagenprotokoll ist eine Messwertabweichung aufgetreten. Bearbeite den "
        f"folgenden Vorgang vollständig:\n\n"
        f"1. Lies das technische Handbuch, Kapitel {chapter}. Ermittle die anlagenspezifischen "
        f"Grenzwerte und die zugehörige Prüfanweisung.\n"
        f"2. Bestimme aus dem Anlagenprotokoll die betroffene Messgröße, den Fehlercode und "
        f"das betroffene Bauteil.\n"
        f"3. Beantworte mit einer SQL-Abfrage gegen die Wartungsdatenbank: {sql_question}\n"
        f"4. Implementiere die geforderte Diagnosefunktion und führe sie über das Werkzeug "
        f"`run_diagnostic_python` mit der Eingabe `{diagnostic_input_id}` aus.\n"
        f"5. Suche das vorgeschriebene Ersatzteil im Bestand und reserviere die im Handbuch "
        f"angegebene Einbaumenge.\n"
        f"6. Lasse einen Sicherungsplan nach Vorschrift {policy_id} über "
        f"`check_safety_plan` freigeben.\n"
        f"7. Übermittle den Abschlussbericht ausschließlich über `submit_final_json`.\n\n"
        f"Gib keinen Freitext außerhalb der Werkzeugaufrufe aus."
    )


def hash_split(split: str) -> int:
    """Stable integer for a split name.

    Python's ``hash`` on strings is salted per process, so it cannot be used to
    derive a seed that must be identical across runs and hosts.
    """
    return {"public": 0x5000, "hidden": 0xA000, "ood": 0xF000}.get(split, 0)


def generate_batch(seeds: list[int], *, split: str = "hidden") -> list[WorkflowInstance]:
    """Generate a set of instances."""
    return [generate_instance(seed, split=split) for seed in seeds]


def sample_seeds(rng: random.Random, count: int, *, low: int = 1, high: int = 2**31) -> list[int]:
    """Draw distinct instance seeds."""
    if count > (high - low):
        raise ValueError("requested more distinct seeds than the range contains")
    seen: set[int] = set()
    while len(seen) < count:
        seen.add(rng.randrange(low, high))
    return sorted(seen)


def with_split(instance: WorkflowInstance, split: str) -> WorkflowInstance:
    """Relabel an instance's split without regenerating it."""
    return replace(instance, split=split, instance_id=_instance_id(instance.seed, split))
