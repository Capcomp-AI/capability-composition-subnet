"""A validator must measure the candidate it just built.

In `own` mode the validator reconstructs each submission locally and then scores
it through `--neuron.serve_url`. Nothing in that path applies the freshly-built
adapter to that endpoint, so every candidate is measured against whatever the
endpoint already holds — the same model for all of them.

The engine refuses to start in the mode that does this:

    serving_mode is 'external'. That mode cannot apply a candidate's adapter, so
    every candidate and every reference would be measured against the same
    static endpoint and no challenger could ever be distinguished.

These tests pin the shape of the gap rather than a fix, so that whichever way it
is closed, the property is stated: what gets served has to be what was built.
"""

from __future__ import annotations

import inspect

import pytest


class TestTheServedPackageIsTheBuiltOne:
    def test_the_evaluator_receives_a_client_it_cannot_point_at_an_artifact(self):
        """`evaluate_candidate` takes a bare client and an artifact directory.

        It has no way to bind them: the client is already addressed at a URL,
        and nothing is passed that could load the artifact into it.
        """
        from capability_subnet.validator.evaluator import evaluate_candidate

        params = inspect.signature(evaluate_candidate).parameters
        assert "client" in params
        assert "artifact_dir" in params
        source = inspect.getsource(evaluate_candidate)
        assert "load_lora" not in source and "lora-modules" not in source, (
            "if the evaluator learns to apply the adapter, this test should be "
            "replaced by one asserting that it does"
        )

    @pytest.mark.xfail(
        reason=(
            "own mode scores every candidate against a static endpoint that was "
            "never told about the artifact just reconstructed, so two different "
            "recipes are measured as the same model"
        ),
        strict=True,
    )
    def test_a_validator_applies_the_adapter_before_scoring(self):
        """The property that has to hold before own mode can pay anyone.

        Marked xfail(strict) rather than skipped: when the gap is closed this
        test fails loudly and gets rewritten, instead of sitting green and
        meaning nothing.
        """
        from capability_subnet.validator import neuron

        measure = inspect.getsource(neuron.ValidatorNeuron._measure)
        assert any(
            marker in measure for marker in ("serve(", "load_lora", "lora_modules", "ManagedVllm")
        ), "nothing in _measure applies the reconstructed adapter to the endpoint"
