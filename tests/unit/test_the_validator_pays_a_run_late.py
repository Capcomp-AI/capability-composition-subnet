"""A weight vector states a closed run's leaderboard, so it is submitted a run late.

The validator used to measure a run and submit the vector it had just computed,
in the same pass. That vector is a leaderboard still being written: a candidate
measured early in the run competes against an empty field, one measured late
against a full one, and the vector moves under both as the queue is worked
through. Two validators that reached the queue in a different order submitted
different vectors from the same evidence.

So the pipeline is three runs deep — committed in N, measured in N+1, paid in
N+2 — and the last step reads back the run report rather than a value held in
memory, because a validator restarted between runs must still pay what it
measured.

Nothing here needs a chain or a GPU: the submission path is stubbed and the
reports are files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from capability_subnet.common import constants as C


class _Config:
    def __init__(self, full_path: Path) -> None:
        self.full_path = str(full_path)
        self.workflow_id = C.DEFAULT_WORKFLOW_ID
        self.netuid = 103
        self.disable_set_weights = False


class _Validator:
    """Just enough of the neuron to exercise the submission decision."""

    def __init__(self, tmp_path: Path) -> None:
        from capability_subnet.validator.neuron import ValidatorNeuron

        self.config = _Config(tmp_path)
        self.submitted: list = []
        self.burned: list[str] = []
        self._run_report_dir = ValidatorNeuron._run_report_dir.__get__(self)
        self._submit_measured_earlier = ValidatorNeuron._submit_measured_earlier.__get__(self)

    def _submit(self, vector, block: int) -> None:
        self.submitted.append(vector)

    def _burn(self, block: int, *, reason: str) -> None:
        self.burned.append(reason)

    def write_report(self, run_id: int, weights: list[dict]) -> None:
        directory = self._run_report_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"run-{run_id}.json").write_text(
            json.dumps({"run_id": run_id, "weights": weights})
        )


@pytest.fixture
def validator(tmp_path):
    return _Validator(tmp_path)


PAID = [
    {"uid": 7, "hotkey": "5Winner", "weight": 0.7, "role": "champion"},
    {"uid": 9, "hotkey": "5Second", "weight": 0.3, "role": "contributor"},
]


class TestItPaysForTheRunThatClosed:
    def test_run_413_submits_what_run_412_measured(self, validator):
        validator.write_report(412, PAID)
        validator.write_report(413, [{"uid": 1, "hotkey": "5Now", "weight": 1.0, "role": "x"}])

        validator._submit_measured_earlier(413, block=8_915_900)

        assert not validator.burned
        (vector,) = validator.submitted
        assert vector.run_id == 412, "it paid for the run it was measuring"
        assert [e.uid for e in vector.entries] == [7, 9]

    def test_the_gap_is_the_constant(self, validator):
        validator.write_report(500 - C.WEIGHT_LAG_RUNS, PAID)

        validator._submit_measured_earlier(500, block=1)

        assert validator.submitted[0].run_id == 500 - C.WEIGHT_LAG_RUNS

    def test_the_weights_survive_the_round_trip(self, validator):
        """It is read back from a file, so every field has to be in the file.

        The report used to carry uid, weight and role but not hotkey, which is
        enough to log a vector and not enough to rebuild one.
        """
        validator.write_report(412, PAID)

        validator._submit_measured_earlier(413, block=1)

        entries = {e.uid: e for e in validator.submitted[0].entries}
        assert entries[7].weight == pytest.approx(0.7)
        assert entries[7].hotkey == "5Winner"
        assert entries[7].role == "champion"


class TestWithoutEvidenceItBurns:
    """Never invent a vector. This is what the rest of the class does when the
    beacon is unreadable, and a missing report is the same kind of gap."""

    def test_no_report_for_that_run_burns(self, validator):
        validator.write_report(413, PAID)  # this run's, not the one being paid

        validator._submit_measured_earlier(413, block=1)

        assert not validator.submitted
        assert len(validator.burned) == 1
        assert "run 412" in validator.burned[0]

    def test_a_report_measuring_nobody_burns(self, validator):
        validator.write_report(412, [])

        validator._submit_measured_earlier(413, block=1)

        assert not validator.submitted
        assert "nobody" in validator.burned[0]

    def test_an_unreadable_report_burns_rather_than_raising(self, validator):
        directory = validator._run_report_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run-412.json").write_text("{ this is not json")

        validator._submit_measured_earlier(413, block=1)

        assert not validator.submitted
        assert len(validator.burned) == 1


def test_the_step_submits_the_earlier_run_rather_than_the_measured_one():
    """The call site, so the two cannot drift apart.

    _step_own computes a vector for the run it just measured. If it went back
    to submitting that, every test above would still pass.
    """
    from capability_subnet.validator import neuron

    source = Path(neuron.__file__).read_text(encoding="utf-8")
    assert "self._submit_measured_earlier(run_id, block)" in source
    assert "self._submit(outcome.weights, block)" not in source, (
        "the validator is paying from the run it is still measuring"
    )
