"""A whole window, decided by a validator with nothing above it.

No signed vector, no allow-list, no operator. What is defended here is that the
validator reaches a weight vector from its own measurements, that one broken
submission cannot tax the rest, that a copy cannot take a slot from the thing it
copied, and that suspecting a peer never costs a miner anything.
"""

from __future__ import annotations

import pytest

from capability_subnet.common.schemas import CandidateScores, GateVerdict
from capability_subnet.validator.evaluator import CandidateEvaluation
from capability_subnet.scoring.retention import ProbeOutcome
from capability_subnet.validator.window import BaseMeasurement, Candidate, run_window

BEACON = "0x" + "ab" * 32


def _candidate(uid: int, recipe, *, first_block: int = 100) -> Candidate:
    return Candidate(uid=uid, hotkey=f"5HOT{uid}", recipe=recipe, first_block=first_block)


#: A candidate that cleared everything. These tests are about how a window
#: turns measurements into weights, not about the gates — but an evaluation
#: carrying no verdicts is deliberately unusable, so a stub has to say it
#: passed rather than say nothing.
def _cleared() -> list[GateVerdict]:
    return [GateVerdict(name="stub", passed=True, detail="cleared in a fixture")]


def _measurement(hotkey: str, score: float, *, seeds=(), success=lambda s: s % 3 == 0):
    return CandidateEvaluation(
        candidate_id=hotkey,
        recipe_sha256="sha256:" + "0" * 64,
        artifact_sha256="sha256:" + "1" * 64,
        artifact_bytes=1024,
        scores=CandidateScores(qualified_score=score),
        per_instance={s: success(s) for s in seeds},
        gate_verdicts=_cleared(),
    )


def _scorer(scores: dict[str, float]):
    """A measure function that returns a fixed score per hotkey."""

    def measure(candidate: Candidate, inputs):
        return _measurement(
            candidate.hotkey, scores[candidate.hotkey], seeds=inputs.assignment.seeds
        )

    return measure


def _base(end_to_end: float = 0.0) -> BaseMeasurement:
    """A reference the fixtures can be held to."""
    return BaseMeasurement(
        end_to_end=end_to_end, probe=ProbeOutcome(correct=0, total=0)
    )


def _run(candidates, measure, *, measure_base=None, **kw):
    return run_window(
        candidates,
        window_id=1080,
        beacon=BEACON,
        hotkey="5SELF",
        block=7_000_000,
        measure=measure,
        measure_base=measure_base or (lambda assignment, sample: _base()),
        hidden_count=120,
        ood_count=20,
        **kw,
    )


class TestTheValidatorDecidesForItself:
    def test_it_produces_weights_from_its_own_measurements(self, recipe_factory):
        r = recipe_factory()
        out = _run(
            [_candidate(1, r), _candidate(2, r)],
            _scorer({"5HOT1": 0.40, "5HOT2": 0.10}),
        )
        assert out.weights is not None
        paid = {e.uid: e.weight for e in out.weights.entries if e.weight > 0}
        assert paid, "a window with measurable candidates must pay somebody"

    def test_it_derives_the_window_without_being_told(self, recipe_factory):
        """No operator hands it seeds — the beacon is the whole input."""
        r = recipe_factory()
        out = _run([_candidate(1, r)], _scorer({"5HOT1": 0.4}))
        assert len(out.sample.hidden_seeds) == 120
        assert out.sample.root_commitment == ""
        assert set(out.assignment.seeds) <= set(out.sample.hidden_seeds)

    def test_two_validators_on_one_beacon_share_a_core(self, recipe_factory):
        r = recipe_factory()
        a = _run([_candidate(1, r)], _scorer({"5HOT1": 0.4}))
        b = run_window(
            [_candidate(1, r)],
            window_id=1080,
            beacon=BEACON,
            hotkey="5OTHER",
            block=7_000_000,
            measure=_scorer({"5HOT1": 0.4}),
            measure_base=lambda assignment, sample: _base(),
            hidden_count=120,
            ood_count=20,
        )
        assert a.assignment.core == b.assignment.core
        assert a.assignment.tail != b.assignment.tail

    def test_a_better_candidate_outranks_a_worse_one(self, recipe_factory):
        r = recipe_factory()
        out = _run(
            [_candidate(1, r), _candidate(2, r)],
            _scorer({"5HOT1": 0.05, "5HOT2": 0.90}),
        )
        weights = {e.uid: e.weight for e in out.weights.entries}
        assert weights.get(2, 0) > weights.get(1, 0)


class TestOneBadSubmissionCannotTaxTheRest:
    def test_an_unmeasurable_candidate_is_absent_not_penalised(self, recipe_factory):
        r = recipe_factory()

        def measure(candidate, inputs):
            if candidate.uid == 2:
                raise RuntimeError("reconstruction exploded")
            return _measurement(candidate.hotkey, 0.5, seeds=inputs.assignment.seeds)

        out = _run([_candidate(1, r), _candidate(2, r)], measure)
        assert len(out.evaluations) == 2
        assert len(out.usable) == 1
        weights = {e.uid: e.weight for e in out.weights.entries}
        assert weights.get(1, 0) > 0
        assert weights.get(2, 0) == 0

    def test_a_window_where_nothing_measures_still_produces_a_vector(self, recipe_factory):
        r = recipe_factory()

        def measure(candidate, inputs):
            raise RuntimeError("nothing works today")

        out = _run([_candidate(1, r)], measure)
        assert out.usable == []
        assert out.weights is not None
        assert sum(e.weight for e in out.weights.entries) == pytest.approx(1.0)


class TestRankingIsByScoreAlone:
    def test_a_higher_score_outranks_an_earlier_commitment(self, recipe_factory):
        """Commit time gives no advantage. The higher measured score ranks first,
        even when a lower-scoring submission committed earlier."""
        r = recipe_factory()
        early_lower = _candidate(1, r, first_block=100)
        later_higher = _candidate(2, r, first_block=900)
        out = _run(
            [early_lower, later_higher],  # deliberately out of commitment order
            _scorer({"5HOT1": 0.5000, "5HOT2": 0.5001}),
        )
        weights = {e.uid: e.weight for e in out.weights.entries}
        assert weights.get(2, 0) > weights.get(1, 0)


class TestSuspicionNeverCostsAMiner:
    def _peer(self, seeds, *, honest: bool):
        rule = (lambda s: s % 3 == 0) if honest else (lambda s: s % 7 == 0)
        return {s: rule(s) for s in seeds}

    def test_a_divergent_peer_is_named(self, recipe_factory):
        r = recipe_factory()
        out = _run([_candidate(1, r)], _scorer({"5HOT1": 0.4}))
        core = set(out.assignment.core)
        peers = {
            "5PEER1": {"5HOT1": self._peer(core, honest=True)},
            "5PEER2": {"5HOT1": self._peer(core, honest=True)},
            "5LIAR": {"5HOT1": self._peer(core, honest=False)},
        }
        again = _run([_candidate(1, r)], _scorer({"5HOT1": 0.4}), peer_core_results=peers)
        assert "5LIAR" in again.flagged_peers

    def test_flagging_a_peer_does_not_change_what_the_miner_is_paid(self, recipe_factory):
        r = recipe_factory()
        plain = _run([_candidate(1, r)], _scorer({"5HOT1": 0.4}))
        core = set(plain.assignment.core)
        peers = {f"5PEER{i}": {"5HOT1": self._peer(core, honest=i != 3)} for i in range(1, 5)}
        with_peers = _run([_candidate(1, r)], _scorer({"5HOT1": 0.4}), peer_core_results=peers)
        assert with_peers.flagged_peers
        assert [(e.uid, e.weight) for e in plain.weights.entries] == [
            (e.uid, e.weight) for e in with_peers.weights.entries
        ]

    def test_peers_measuring_a_different_candidate_are_not_compared(self, recipe_factory):
        r = recipe_factory()
        out = _run(
            [_candidate(1, r)],
            _scorer({"5HOT1": 0.4}),
            peer_core_results={"5PEER": {"5SOMEONE-ELSE": {1: True}}},
        )
        assert out.flagged_peers == {}


class TestMeasuringCandidatesInParallel:
    """Concurrency may change when a candidate is measured, never what it is worth."""

    def test_results_follow_commit_order_not_completion_order(self, recipe_factory):
        """A slow candidate must not lose its place to a fast one.

        Ordering is the tie-break, and the tie-break is what stops a copy taking
        a slot from the package it copied. Threading it onto whichever GPU
        happened to free up first would decide that on the hardware.
        """
        import time

        r = recipe_factory()
        delays = {"5HOT1": 0.20, "5HOT2": 0.05, "5HOT3": 0.10}

        def measure(candidate: Candidate, inputs):
            time.sleep(delays[candidate.hotkey])
            return _measurement(candidate.hotkey, 0.5, seeds=inputs.assignment.seeds)

        candidates = [_candidate(1, r), _candidate(2, r), _candidate(3, r)]
        out = _run(candidates, measure, workers=3)

        assert [e.candidate_id for e in out.evaluations] == ["5HOT1", "5HOT2", "5HOT3"]

    def test_workers_do_not_change_the_weight_vector(self, recipe_factory):
        r = recipe_factory()
        scores = {"5HOT1": 0.40, "5HOT2": 0.25, "5HOT3": 0.10}
        candidates = [_candidate(1, r), _candidate(2, r), _candidate(3, r)]

        serial = _run(candidates, _scorer(scores), workers=1)
        parallel = _run(candidates, _scorer(scores), workers=4)

        assert serial.weights is not None and parallel.weights is not None
        assert {e.uid: round(e.weight, 12) for e in serial.weights.entries} == {
            e.uid: round(e.weight, 12) for e in parallel.weights.entries
        }

    def test_one_candidate_failing_does_not_stop_the_others(self, recipe_factory):
        r = recipe_factory()

        def measure(candidate: Candidate, inputs):
            if candidate.hotkey == "5HOT2":
                raise RuntimeError("this host could not serve it")
            return _measurement(candidate.hotkey, 0.5, seeds=inputs.assignment.seeds)

        out = _run([_candidate(1, r), _candidate(2, r), _candidate(3, r)], measure, workers=3)

        assert [e.candidate_id for e in out.evaluations] == ["5HOT1", "5HOT2", "5HOT3"]
        assert {e.candidate_id for e in out.usable} == {"5HOT1", "5HOT3"}
        failed = next(e for e in out.evaluations if e.candidate_id == "5HOT2")
        assert "could not serve it" in (failed.error or "")


class TestTheWindowMeasuresWhatItClaimsTo:
    """The scored terms are wired to something that measures them.

    Each of these covers a term that was silently absent from the default
    validator path: the out-of-distribution draw was extracted and never passed
    on, retention defaulted to 1.0 with no base probe to compare against, the
    reference defaulted to zero so "improvement" meant "score", and `usable`
    asked only whether the host had crashed. A window can be wrong in all four
    ways at once without a single error in a log, which is why they are pinned
    here rather than left to review.
    """

    def test_the_reference_is_measured_and_becomes_the_bar(self, recipe_factory):
        r = recipe_factory()
        seen = {}

        def measure(candidate, inputs):
            seen["reference"] = inputs.base.end_to_end
            seen["probe"] = inputs.base.probe
            return _measurement(candidate.hotkey, 0.5, seeds=inputs.assignment.seeds)

        out = _run(
            [_candidate(1, r)],
            measure,
            measure_base=lambda assignment, sample: BaseMeasurement(
                end_to_end=0.42, probe=ProbeOutcome(correct=36, total=40)
            ),
        )

        assert seen["reference"] == 0.42
        assert seen["probe"].correct == 36
        assert out.reference_e2e == 0.42

    def test_a_window_cannot_run_without_a_reference(self, recipe_factory):
        """No bar, no window. It used to default to zero and carry on."""
        r = recipe_factory()
        with pytest.raises(TypeError, match="measure_base"):
            run_window(
                [_candidate(1, r)],
                window_id=1080,
                beacon=BEACON,
                hotkey="5SELF",
                block=7_000_000,
                measure=_scorer({"5HOT1": 0.5}),
            )

    def test_the_out_of_distribution_draw_reaches_the_measurement(self, recipe_factory):
        r = recipe_factory()
        seen = {}

        def measure(candidate, inputs):
            seen["ood"] = inputs.ood_seeds
            return _measurement(candidate.hotkey, 0.5, seeds=inputs.assignment.seeds)

        out = _run([_candidate(1, r)], measure)

        assert seen["ood"] == out.sample.ood_seeds
        assert len(seen["ood"]) == 20, "the draw was extracted and then dropped"

    def test_the_probe_seed_reaches_the_measurement(self, recipe_factory):
        r = recipe_factory()
        seen = {}

        def measure(candidate, inputs):
            seen["probe_seed"] = inputs.probe_seed
            return _measurement(candidate.hotkey, 0.5, seeds=inputs.assignment.seeds)

        out = _run([_candidate(1, r)], measure)

        assert seen["probe_seed"] == out.sample.probe_seed
        assert seen["probe_seed"] != 0, "a zero probe seed is the same probe every window"

    def test_a_candidate_that_fails_a_gate_does_not_compete(self, recipe_factory):
        r = recipe_factory()

        def measure(candidate, inputs):
            evaluation = _measurement(candidate.hotkey, 0.9, seeds=inputs.assignment.seeds)
            evaluation.gate_verdicts = [
                GateVerdict(name="base_retention", passed=False, detail="0.700 against a 0.95 floor")
            ]
            return evaluation

        out = _run([_candidate(1, r)], measure)

        assert out.usable == [], "a candidate that failed a gate was ranked anyway"
        assert out.evaluations[0].gate_failures == [
            "base_retention: 0.700 against a 0.95 floor"
        ]

    def test_an_evaluation_whose_gates_never_ran_does_not_compete(self, recipe_factory):
        """An empty verdict list is not a pass.

        ``all([])`` is True, so a candidate that returned before the gates ran
        would otherwise read as having cleared every one of them.
        """
        r = recipe_factory()

        def measure(candidate, inputs):
            evaluation = _measurement(candidate.hotkey, 0.9, seeds=inputs.assignment.seeds)
            evaluation.gate_verdicts = []
            return evaluation

        assert _run([_candidate(1, r)], measure).usable == []
