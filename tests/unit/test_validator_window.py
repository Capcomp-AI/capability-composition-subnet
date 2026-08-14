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


class TestACopyCannotTakeTheSlotItCopied:
    def test_an_indistinguishable_later_commitment_ranks_second(self, recipe_factory):
        """Two scores closer than this validator's slice can resolve are one
        measurement repeated, and the earlier commitment wins the tie."""
        r = recipe_factory()
        original = _candidate(1, r, first_block=100)
        copy = _candidate(2, r, first_block=900)
        out = _run(
            [copy, original],  # deliberately out of commitment order
            _scorer({"5HOT1": 0.5000, "5HOT2": 0.5001}),
        )
        weights = {e.uid: e.weight for e in out.weights.entries}
        assert weights.get(1, 0) > weights.get(2, 0)


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
