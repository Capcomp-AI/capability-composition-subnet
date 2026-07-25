"""Telling the two kinds of failure apart.

This distinction carries the whole one-shot-per-hotkey rule. A miner's mistake
must cost them their shot; the operator's must not. Getting it wrong in either
direction is a serious bug:

* classifying an operator failure as a miner failure terminates candidates for
  an evaluation the engine never completed,
* classifying a miner failure as an operator failure lets a broken recipe sit in
  the queue forever, blocking everyone behind it.

Neither is visible in a normal run, which is why they are tested directly.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from capability_subnet.backend.evaluation import EvaluablePackage, Evaluator
from capability_subnet.backend.executor.reconstruction import ArtifactCache, Reconstructor
from capability_subnet.backend.executor.serving import ServingError
from capability_subnet.merge_engine.loader import InMemoryAdapterSource, SafetensorsAdapterSource


class _StubServer:
    """A server that yields a handle whose client is never used."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    class _Handle:
        peak_vram_gb = 0.0

        def client(self, timeout: float = 120.0):
            raise AssertionError("no instance should have been run")

    @contextmanager
    def serve(self, adapter_path):
        if self.failure is not None:
            raise self.failure
        yield _StubServer._Handle()


def _evaluator(snapshot, source, tmp_path, *, server=None) -> Evaluator:
    return Evaluator(
        reconstructor=Reconstructor(snapshot, source, ArtifactCache(tmp_path / "cache"), workers=1),
        server=server or _StubServer(),
        adapter_pool_dir=tmp_path / "pool",
        stages=("stage_a",),
        min_valid_samples=1,
    )


class TestOperatorFailures:
    """These must never terminate a candidate."""

    def test_an_unmaterialised_pool_is_infrastructure(
        self, tiny_snapshot, recipe_factory, tmp_path
    ):
        # The pool has not finished downloading. Spending a hotkey's single shot
        # on that would be the engine charging a miner for its own state.
        evaluator = _evaluator(tiny_snapshot, InMemoryAdapterSource({}), tmp_path)

        output = evaluator.evaluate(
            EvaluablePackage(candidate_id="5Miner", recipe=recipe_factory()), [], []
        )

        assert not output.usable
        assert "not available on this host" in output.infrastructure_error
        assert output.gate_verdicts == []

    def test_a_serving_failure_is_infrastructure(
        self, tiny_snapshot, tiny_pool_dir, recipe_factory, tmp_path
    ):
        evaluator = _evaluator(
            tiny_snapshot,
            SafetensorsAdapterSource(tiny_pool_dir),
            tmp_path,
            server=_StubServer(ServingError("the endpoint fell over")),
        )

        output = evaluator.evaluate(
            EvaluablePackage(candidate_id="5Miner", recipe=recipe_factory()), [], []
        )

        assert not output.usable
        assert "fell over" in output.infrastructure_error

    def test_a_missing_single_adapter_reference_is_infrastructure(self, tiny_snapshot, tmp_path):
        evaluator = _evaluator(tiny_snapshot, InMemoryAdapterSource({}), tmp_path)

        output = evaluator.evaluate(
            EvaluablePackage(
                candidate_id="reference:best_single_adapter:alpha-capability-v1",
                adapter_id="alpha-capability-v1",
            ),
            [],
            [],
        )

        assert not output.usable
        assert "not materialised" in output.infrastructure_error


class TestMinerFailures:
    """These must score zero, as a recorded verdict rather than an exception."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("source_snapshot_sha256", "sha256:" + "0" * 64, "source snapshot"),
            ("base_revision", "some-other-revision", "base revision"),
        ],
    )
    def test_a_recipe_for_the_wrong_pool_is_a_gate_failure(
        self, tiny_snapshot, tiny_pool_dir, recipe_factory, tmp_path, field, value, expected
    ):
        evaluator = _evaluator(tiny_snapshot, SafetensorsAdapterSource(tiny_pool_dir), tmp_path)
        recipe = recipe_factory().model_copy(update={field: value})

        output = evaluator.evaluate(EvaluablePackage(candidate_id="5Miner", recipe=recipe), [], [])

        # Usable, so the candidate is judged rather than held…
        assert output.usable
        # …and it fails, with the reason recorded for the published report.
        assert not output.gates_passed
        assert any(expected in verdict.detail for verdict in output.gate_verdicts)

    def test_an_oversized_artifact_is_refused_before_a_gpu_is_touched(
        self, tiny_snapshot, tiny_pool_dir, recipe_factory, tmp_path, monkeypatch
    ):
        from capability_subnet.backend.scorer import gates

        evaluator = _evaluator(tiny_snapshot, SafetensorsAdapterSource(tiny_pool_dir), tmp_path)

        # Pretend every artifact is over the limit; the stub server asserts if
        # anything actually tries to run.
        monkeypatch.setattr(
            gates,
            "gate_artifact_size",
            lambda _bytes: gates.gate("artifact_size", False, "too big"),
        )

        output = evaluator.evaluate(
            EvaluablePackage(candidate_id="5Miner", recipe=recipe_factory()), [], []
        )

        assert output.usable
        assert not output.gates_passed
