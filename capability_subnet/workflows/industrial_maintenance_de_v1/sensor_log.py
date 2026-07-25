"""Sensor log generation.

The log is semi-structured controller output: a header block, then timestamped
rows for every sensor on the machine. Exactly one sensor is driven into
exceedance; the others stay inside their nominal band and act as distractors.

Extracting the right channel, in the right unit, and computing the right
statistics from it is the first half of the workflow. The numbers here are also
what the ground truth is computed from, so the log and the truth cannot drift
apart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from capability_subnet.workflows.industrial_maintenance_de_v1.machines import (
    MachineInstance,
    Sensor,
)

#: Number of samples per sensor in a log.
SAMPLE_COUNT = 24

#: Minutes between samples.
SAMPLE_INTERVAL_MINUTES = 5


@dataclass(frozen=True, slots=True)
class SensorSeries:
    """The generated readings for one channel, in the unit the log reports."""

    sensor: Sensor
    unit: str
    values: tuple[float, ...]
    warn_threshold: float
    critical_threshold: float

    @property
    def peak(self) -> float:
        return max(self.values)

    @property
    def exceedances(self) -> int:
        return sum(1 for value in self.values if value > self.warn_threshold)

    @property
    def longest_run(self) -> int:
        longest = 0
        current = 0
        for value in self.values:
            if value > self.warn_threshold:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest


def _round(value: float) -> float:
    """Round to one decimal.

    Every number the candidate sees is rounded before the truth is computed from
    it, so the truth is derived from exactly the values in the log — not from a
    higher-precision series the candidate never saw.
    """
    return round(value, 1)


def _generate_normal(rng: random.Random, sensor: Sensor) -> list[float]:
    span = sensor.nominal_high - sensor.nominal_low
    return [sensor.nominal_low + rng.uniform(0.1, 0.9) * span for _ in range(SAMPLE_COUNT)]


def _generate_faulty(
    rng: random.Random,
    sensor: Sensor,
    warn: float,
    critical: float,
    *,
    critical_event: bool,
) -> list[float]:
    """Drive one channel above its warning threshold.

    The exceedance is generated as one contiguous burst plus a couple of isolated
    spikes. That shape is deliberate: it makes the longest-run statistic differ
    from the plain exceedance count, so a diagnostic implementation that conflates
    the two fails the hidden tests.
    """
    span = sensor.nominal_high - sensor.nominal_low
    values = [sensor.nominal_low + rng.uniform(0.2, 0.8) * span for _ in range(SAMPLE_COUNT)]

    burst_length = rng.randint(3, 6)
    burst_start = rng.randint(2, SAMPLE_COUNT - burst_length - 3)
    ceiling = critical if critical_event else warn + (critical - warn) * 0.6

    for offset in range(burst_length):
        low = warn + (ceiling - warn) * 0.15
        high = ceiling * (1.06 if critical_event else 0.96)
        values[burst_start + offset] = rng.uniform(low, high)

    # Isolated spikes outside the burst, so exceedances > longest_run.
    spare = [
        index
        for index in range(SAMPLE_COUNT)
        if not (burst_start - 1 <= index <= burst_start + burst_length)
    ]
    for index in rng.sample(spare, k=min(2, len(spare))):
        values[index] = rng.uniform(warn * 1.01, warn + (ceiling - warn) * 0.5)

    if critical_event:
        peak_index = burst_start + rng.randrange(burst_length)
        values[peak_index] = critical * rng.uniform(1.04, 1.15)

    return values


def build_series(
    rng: random.Random,
    machine: MachineInstance,
    faulty_sensor_key: str,
    *,
    use_alt_units: bool,
    critical_event: bool,
) -> dict[str, SensorSeries]:
    """Generate every channel for one machine.

    Values are generated in the sensor's native unit and converted once, at the
    end, together with the thresholds. Converting in two places is how a
    generator ends up with a log that disagrees with its own ground truth.
    """
    series: dict[str, SensorSeries] = {}
    for sensor in machine.machine_type.sensors:
        factor = sensor.alt_factor if (use_alt_units and sensor.alt_unit) else 1.0
        unit = sensor.alt_unit if (use_alt_units and sensor.alt_unit) else sensor.unit
        warn_native = machine.warn_threshold(sensor.key)
        critical_native = machine.critical_threshold(sensor.key)

        if sensor.key == faulty_sensor_key:
            native = _generate_faulty(
                rng, sensor, warn_native, critical_native, critical_event=critical_event
            )
        else:
            native = _generate_normal(rng, sensor)

        series[sensor.key] = SensorSeries(
            sensor=sensor,
            unit=unit,
            values=tuple(_round(value * factor) for value in native),
            warn_threshold=_round(warn_native * factor),
            critical_threshold=_round(critical_native * factor),
        )
    return series


def render_log(machine: MachineInstance, series: dict[str, SensorSeries], *, log_date: str) -> str:
    """Render the controller dump the agent reads."""
    header = [
        "=== ANLAGENPROTOKOLL / CONTROLLER EXPORT ===",
        f"Anlage        : {machine.machine_id}",
        f"Baureihe      : {machine.machine_type.model_de}",
        f"Aufstellort   : {machine.location_de}",
        f"Protokolldatum: {log_date}",
        f"Abtastintervall: {SAMPLE_INTERVAL_MINUTES} min",
        "",
        "ZEIT      | KANAL                      | WERT       | EINHEIT",
        "-" * 68,
    ]

    lines: list[str] = []
    for index in range(SAMPLE_COUNT):
        minutes = index * SAMPLE_INTERVAL_MINUTES
        timestamp = f"{6 + minutes // 60:02d}:{minutes % 60:02d}:00"
        for sensor_key in sorted(series):
            entry = series[sensor_key]
            value = entry.values[index]
            lines.append(
                f"{timestamp}  | {entry.sensor.label_de:<26} | {value:>10.1f} | {entry.unit}"
            )

    footer = ["-" * 68, "=== ENDE PROTOKOLL ==="]
    return "\n".join(header + lines + footer)
