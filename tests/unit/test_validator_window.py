"""A whole window, decided by a validator with nothing above it.

No signed vector, no allow-list, no operator. What is defended here is that the
validator reaches a weight vector from its own measurements, that one broken
submission cannot tax the rest, that a copy cannot take a slot from the thing it
copied, and that suspecting a peer never costs a miner anything.
"""

from __future__ import annotations

import pytest

from capability_subnet.common.schemas import CandidateScores
from capability_subnet.validator.evaluator import CandidateEvaluation
from capability_subnet.validator.window import Candidate, run_window

BEACON = "0x" + "ab" * 32


def _candidate(uid: int, recipe, *, first_block: int = 100) -> Candidate:
    return Candidate(uid=uid, hotkey=f"5HOT{uid}", recipe=recipe, first_block=first_block)


def _measurement(hotkey: str, score: float, *, seeds=(), success=lambda s: s % 3 == 0):
    return CandidateEvaluation(
        candidate_id=hotkey,
        recipe_sha256="sha256:" + "0" * 64,
        artifact_sha256="sha256:" + "1" * 64,
        artifact_bytes=1024,
        scores=CandidateScores(qualified_score=score),
        per_instance={s: success(s) for s in seeds},
    )


def _scorer(scores: dict[str, float]):
    """A measure function that returns a fixed score per hotkey."""

    def measure(candidate: Candidate, assignment):
        return _measurement(candidate.hotkey, scores[candidate.hotkey], seeds=assignment.seeds)

    return measure


def _run(candidates, measure, **kw):
    return run_window(
        candidates,
        window_id=1080,
        beacon=BEACON,
        hotkey="5SELF",
        block=7_000_000,
        measure=measure,
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

        def measure(candidate, assignment):
            if candidate.uid == 2:
                raise RuntimeError("reconstruction exploded")
            return _measurement(candidate.hotkey, 0.5, seeds=assignment.seeds)

        out = _run([_candidate(1, r), _candidate(2, r)], measure)
        assert len(out.evaluations) == 2
        assert len(out.usable) == 1
        weights = {e.uid: e.weight for e in out.weights.entries}
        assert weights.get(1, 0) > 0
        assert weights.get(2, 0) == 0

    def test_a_window_where_nothing_measures_still_produces_a_vector(self, recipe_factory):
        r = recipe_factory()

        def measure(candidate, assignment):
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

        def measure(candidate: Candidate, assignment):
            time.sleep(delays[candidate.hotkey])
            return _measurement(candidate.hotkey, 0.5, seeds=assignment.seeds)

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

        def measure(candidate: Candidate, assignment):
            if candidate.hotkey == "5HOT2":
                raise RuntimeError("this host could not serve it")
            return _measurement(candidate.hotkey, 0.5, seeds=assignment.seeds)

        out = _run([_candidate(1, r), _candidate(2, r), _candidate(3, r)], measure, workers=3)

        assert [e.candidate_id for e in out.evaluations] == ["5HOT1", "5HOT2", "5HOT3"]
        assert {e.candidate_id for e in out.usable} == {"5HOT1", "5HOT3"}
        failed = next(e for e in out.evaluations if e.candidate_id == "5HOT2")
        assert "could not serve it" in (failed.error or "")
