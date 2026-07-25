"""German technical manual generation.

The manual is the only place the candidate can learn this machine's thresholds,
fault-code mapping, part numbers, safety requirements and decision rule. It is
generated in German technical register — compound nouns, DIN-style references,
imperative procedure text — because reading that register correctly is one of
the capabilities the workflow is meant to measure.

Nothing here is decorative: every fact the scorer checks appears in exactly one
section, so a candidate that skips the manual cannot reach the right answer by
pattern-matching the prompt.
"""

from __future__ import annotations

import random

from capability_subnet.workflows.industrial_maintenance_de_v1.instance import ManualSection
from capability_subnet.workflows.industrial_maintenance_de_v1.machines import (
    SAFETY_STEPS,
    MachineInstance,
)

#: Synonyms used by out-of-distribution instances. The manual then introduces the
#: synonym and uses it throughout, so the candidate has to follow the definition
#: rather than recognise a memorised label.
COMPONENT_SYNONYMS_DE: dict[str, str] = {
    "hydraulikpumpe": "Hydraulikaggregat",
    "oelkuehler": "Wärmetauschereinheit",
    "steuerventil": "Wegeventilblock",
    "antriebsmotor": "Bandantriebseinheit",
    "umlenkrollenlager": "Umlenkstation-Lagerung",
    "frequenzumrichter": "Antriebsregler",
    "verdichter": "Kältemittelverdichter",
    "expansionsventil": "Drosselorgan",
    "fuellstandssensor": "Niveaugeber",
    "spindelantrieb": "Hauptspindelantrieb",
    "spindellager": "Hauptspindellagerung",
    "vorschubantrieb": "Vorschubachse",
    "verdichterstufe": "Verdichterblock",
    "oelabscheider": "Ölabscheidesystem",
    "kaeltetrockner": "Drucklufttrocknungseinheit",
}


def procedure_id(chapter: str, fault_code: str) -> str:
    """Identifier of the test instruction for one fault.

    This is what the final payload must cite as ``manual_reference``. It exists
    only inside the diagnosis table, so citing it correctly is direct evidence
    that the candidate read the right row.
    """
    suffix = fault_code.replace("-", "").replace(".", "")[-3:]
    return f"PA-{chapter}.3-{suffix}"


def _format_threshold(value: float, unit: str) -> str:
    text = f"{value:.1f}".replace(".", ",")
    return f"{text} {unit}"


def build_manual(
    machine: MachineInstance,
    *,
    use_synonyms: bool = False,
    use_alt_units: bool = False,
    alt_fault_code_format: bool = False,
    rng: random.Random | None = None,
) -> tuple[ManualSection, ...]:
    """Generate the manual for one machine.

    Args:
        machine: the machine the manual documents.
        use_synonyms: introduce and then use alternative component names.
        use_alt_units: express thresholds in the sensors' alternative unit.
        alt_fault_code_format: write fault codes in a dotted rather than hyphenated
            form, so a candidate cannot rely on the surface shape of the code.
    """
    rng = rng or random.Random(0)
    machine_type = machine.machine_type
    chapter = machine_type.manual_chapter

    def component_label(component) -> str:
        if use_synonyms and component.key in COMPONENT_SYNONYMS_DE:
            return COMPONENT_SYNONYMS_DE[component.key]
        return component.label_de

    def fault_code(component) -> str:
        return (
            component.fault_code.replace("-", ".")
            if alt_fault_code_format
            else component.fault_code
        )

    sections: list[ManualSection] = []

    # --- overview ----------------------------------------------------------
    glossary = ""
    if use_synonyms:
        entries = "; ".join(
            f"{COMPONENT_SYNONYMS_DE[c.key]} (vormals {c.label_de})"
            for c in machine_type.components
            if c.key in COMPONENT_SYNONYMS_DE
        )
        glossary = (
            f"\n\nBegriffsbestimmung: In dieser Ausgabe werden folgende Benennungen "
            f"verwendet: {entries}."
        )

    sections.append(
        ManualSection(
            section_id=f"{chapter}.1",
            title_de=f"Sicherheitshinweise — {machine_type.model_de}",
            body_de=(
                f"Vor jeder Instandsetzung an der Anlage {machine.machine_id} "
                f"({machine_type.model_de}, Aufstellort {machine.location_de}) sind die für das "
                f"betroffene Bauteil vorgeschriebenen Sicherungsmaßnahmen vollständig "
                f"durchzuführen und im Freigabeschein zu dokumentieren.\n\n"
                + "\n".join(
                    f"- {component_label(component)}: "
                    + ", ".join(
                        f"{step} ({SAFETY_STEPS[step]})" for step in component.required_safety_steps
                    )
                    for component in machine_type.components
                )
                + "\n\nUnzulässig sind insbesondere Arbeiten am nicht entlasteten System, "
                "das Überbrücken des Sicherheitskreises, der Betrieb mit demontierter "
                "Schutzeinrichtung sowie Heißarbeiten ohne Freigabeschein." + glossary
            ),
        )
    )

    # --- thresholds --------------------------------------------------------
    threshold_lines = []
    for sensor in machine_type.sensors:
        warn = machine.warn_threshold(sensor.key)
        critical = machine.critical_threshold(sensor.key)
        unit = sensor.unit
        if use_alt_units and sensor.alt_unit:
            warn *= sensor.alt_factor
            critical *= sensor.alt_factor
            unit = sensor.alt_unit
        threshold_lines.append(
            f"- {sensor.label_de}: Warngrenze {_format_threshold(warn, unit)}, "
            f"Grenzwert {_format_threshold(critical, unit)} "
            f"(Nennbereich {_format_threshold(sensor.nominal_low * (sensor.alt_factor if use_alt_units and sensor.alt_unit else 1.0), unit)} "
            f"bis {_format_threshold(sensor.nominal_high * (sensor.alt_factor if use_alt_units and sensor.alt_unit else 1.0), unit)})"
        )

    unit_note = ""
    if use_alt_units:
        unit_note = (
            "\n\nHinweis: Die Messwerte der Anlagensteuerung werden in der Betriebseinheit "
            "protokolliert. Vor dem Vergleich mit den hier angegebenen Grenzwerten ist "
            "umzurechnen."
        )

    sections.append(
        ManualSection(
            section_id=f"{chapter}.2",
            title_de="Grenzwerte und Schwellenwerte",
            body_de=(
                f"Für die Anlage {machine.machine_id} gelten die folgenden anlagenspezifischen "
                f"Grenzwerte. Abweichungen gegenüber der Baureihe ergeben sich aus der "
                f"Abnahmemessung.\n\n" + "\n".join(threshold_lines) + unit_note
            ),
        )
    )

    # --- diagnosis ---------------------------------------------------------
    diagnosis_lines = []
    for component in machine_type.components:
        sensor = machine_type.sensor(component.sensor_key)
        diagnosis_lines.append(
            f"- Prüfanweisung {procedure_id(chapter, component.fault_code)}: "
            f"Überschreitet {sensor.label_de} die Warngrenze, so liegt der Fehler "
            f"{fault_code(component)} am Bauteil {component_label(component)} vor. "
            f"Die Anzahl der Überschreitungen sowie die längste zusammenhängende "
            f"Überschreitungsfolge sind zu protokollieren."
        )

    sections.append(
        ManualSection(
            section_id=f"{chapter}.3",
            title_de="Störungsdiagnose",
            body_de=(
                "Die Zuordnung von Messwertabweichung zu Fehlercode und Bauteil erfolgt "
                "ausschließlich nach der folgenden Tabelle. Die Prüfanweisungsnummer ist "
                "im Abschlussbericht anzugeben.\n\n" + "\n".join(diagnosis_lines)
            ),
        )
    )

    # --- spare parts -------------------------------------------------------
    part_lines = []
    for component in machine_type.components:
        quantity = required_quantity(component.key)
        part_lines.append(
            f"- {component_label(component)}: Ersatzteil {component.part_number} "
            f"({component.part_description_de}), Einbaumenge {quantity} Stück, "
            f"Listenpreis {component.unit_price_eur:.2f} EUR"
        )

    sections.append(
        ManualSection(
            section_id=f"{chapter}.4",
            title_de="Ersatzteile und Einbaumengen",
            body_de=(
                "Es sind ausschließlich die nachstehend aufgeführten aktuellen "
                "Ersatzteilnummern zu verwenden. Vorgängernummern sind gesperrt und "
                "dürfen nicht reserviert werden.\n\n" + "\n".join(part_lines)
            ),
        )
    )

    # --- decision rule -----------------------------------------------------
    sections.append(
        ManualSection(
            section_id=f"{chapter}.5",
            title_de="Instandsetzungsentscheidung",
            body_de=(
                "Die Dringlichkeit ergibt sich ausschließlich aus dem Spitzenwert der "
                "betroffenen Messgröße:\n\n"
                "- Spitzenwert oberhalb des Grenzwertes: sofortige_instandsetzung\n"
                "- Spitzenwert oberhalb der Warngrenze, jedoch nicht oberhalb des "
                "Grenzwertes: geplante_wartung\n"
                "- Spitzenwert innerhalb des Nennbereichs: beobachtung\n\n"
                "Der Abschlussbericht ist als JSON-Objekt nach dem vorgegebenen Schema zu "
                "übermitteln. Freitext außerhalb des JSON-Objekts ist unzulässig."
            ),
        )
    )

    return tuple(sections)


def required_quantity(component_key: str) -> int:
    """Installed quantity for a component.

    Bearings are replaced as a set; everything else is a single unit. The rule is
    stated in the spare-parts section, so a candidate that reads it gets this
    right and one that assumes "one of everything" does not.
    """
    return 2 if "lager" in component_key else 1
