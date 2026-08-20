"""A validator measures the candidate it just built.

In `own` mode the validator reconstructs each submission and then scores it. If
the scorer talks to a long-lived endpoint instead, every candidate is measured
against whatever that endpoint already holds: two different recipes produce the
same numbers, and the ranking has nothing to rank. The evaluation engine treats
that arrangement as unrunnable and refuses to start in it.

So the adapter is applied by starting a runtime for it, and the runtime is
stopped afterwards.
"""

from __future__ import annotations

import inspect

import pytest

from capability_subnet.common import constants as C
from capability_subnet.validator.serving import (
    CANDIDATE_MODEL,
    ServingError,
    build_command,
    utilization_for,
)


class TestTheServedPackageIsTheBuiltOne:
    def test_the_evaluator_serves_the_artifact_it_reconstructed(self):
        from capability_subnet.validator.evaluator import evaluate_candidate

        source = inspect.getsource(evaluate_candidate)
        assert "serve(str(artifact_dir))" in source, (
            "the scorer must be pointed at the artifact this call built, not at "
            "whatever endpoint it was handed"
        )

    def test_the_neuron_hands_the_evaluator_a_server(self):
        from capability_subnet.validator import neuron

        measure = inspect.getsource(neuron.ValidatorNeuron._measure)
        assert "serve_candidate" in measure and "serve=serve" in measure

    def test_the_command_applies_the_candidate_as_a_lora(self):
        command = build_command(
            "/artifacts/cand",
            base_model_path="/models/base",
            python_executable="/venv/bin/python",
            host="127.0.0.1",
            port=8000,
            gpu_memory_utilization=0.45,
        )
        assert "--enable-lora" in command
        assert f"{CANDIDATE_MODEL}=/artifacts/cand" in command

    def test_the_base_model_is_not_served_under_the_candidate_s_name(self):
        """Two entries sharing an id leave a request ambiguous between the
        merged package and the bare base model."""
        command = build_command(
            "/artifacts/cand",
            base_model_path="/models/base",
            python_executable="/venv/bin/python",
            host="127.0.0.1",
            port=8000,
            gpu_memory_utilization=0.45,
        )
        assert command[command.index("--served-model-name") + 1] != CANDIDATE_MODEL

    def test_nothing_may_swap_the_package_while_it_is_measured(self):
        from capability_subnet.validator.serving import _environment

        env = _environment("/venv/bin/python", "cuda")
        assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "0"
        assert env["HF_HUB_OFFLINE"] == "1"


class TestTheReservationIsTheSameEverywhere:
    """`peak_vram` is a gate, so it has to mean the same thing on every host.

    A fraction of the card does not: the same candidate measures 25.4 GiB on a
    validator running 0.78 of a 32 GB card and 22.9 GiB on one running 0.70, and
    one refuses what the other passes for a reason that is not the miner's.
    """

    @pytest.mark.parametrize("total", [24.0, 31.39, 47.4, 79.2])
    def test_the_same_absolute_memory_is_reserved_on_any_card(self, total):
        reserved = utilization_for(total) * total
        assert reserved == pytest.approx(C.SERVING_RESERVED_GIB, abs=1e-6)

    def test_a_card_too_small_is_refused_rather_than_squeezed(self):
        """Serving a package that does not fit would measure the card."""
        with pytest.raises(ServingError, match="rather than the package"):
            utilization_for(16.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_nonsense_card_size_raises(self, bad):
        with pytest.raises(ValueError):
            utilization_for(bad)
