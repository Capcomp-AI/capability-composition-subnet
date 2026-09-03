"""Endpoint mode takes somebody else's scores, and is not a relay.

A validator that cannot afford four 32 GB cards is otherwise a validator the
network does not have, so `--neuron.mode endpoint` sets weights from scores an
engine published. It is the weaker of the two claims and it is not the default.

What keeps it from being a relay is that every refusal below ends the same way:
this validator burns, with its own stake. These pin that, because the failure
mode of a mode like this is silent - a vector that is merely passed along looks
exactly like one that was checked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.config import build_config, parse_trusted_signers


class TestTheModeIsOfferedAndLocalIsTheDefault:
    def test_local_is_what_you_get_without_asking(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["capability-subnet-validator"])
        monkeypatch.delenv("CAPSUB_VALIDATOR_MODE", raising=False)

        assert build_config("validator").mode == "local"

    def test_endpoint_has_to_be_asked_for(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "capability-subnet-validator",
                "--neuron.mode",
                "endpoint",
                "--neuron.backend_url",
                "https://engine.example",
            ],
        )

        assert build_config("validator").mode == "endpoint"

    def test_endpoint_without_an_engine_is_refused_at_startup(self, monkeypatch):
        """Not a default to fall back on: there is nothing to fall back to.

        Endpoint mode reads scores from an engine. Starting without one gives a
        neuron that runs, reads nothing and sets no weights, which looks healthy
        from outside for as long as nobody checks the emissions.
        """
        monkeypatch.setattr(
            "sys.argv", ["capability-subnet-validator", "--neuron.mode", "endpoint"]
        )
        monkeypatch.delenv("CAPSUB_BACKEND_URL", raising=False)

        with pytest.raises(SystemExit, match="needs --neuron.backend_url"):
            build_config("validator")

    def test_nothing_else_is_accepted(self, monkeypatch):
        """A typo must not fall through to whichever branch is the else."""
        monkeypatch.setattr(
            "sys.argv", ["capability-subnet-validator", "--neuron.mode", "endpiont"]
        )

        with pytest.raises(SystemExit):
            build_config("validator")

    def test_the_allow_list_is_a_set_of_hotkeys(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["capability-subnet-validator", "--neuron.trusted_signers", "5ABC, 5DEF"],
        )

        assert parse_trusted_signers(build_config("validator").trusted_signers) == {
            "5ABC",
            "5DEF",
        }

    def test_an_empty_allow_list_enforces_nothing(self):
        """Which is why the neuron warns when it sees one."""
        assert parse_trusted_signers("") is None


class TestConfigBuildsAtAll:
    def test_the_validator_config_can_be_built(self, monkeypatch):
        """It could not, for nine days.

        `--neuron.burn_percentage` defaulted to `C.DEFAULT_BURN_PERCENTAGE`,
        which was deleted from constants on 2026-08-22. Every call raised
        AttributeError before argparse ran, so no validator could start and no
        test covered it.
        """
        monkeypatch.setattr("sys.argv", ["capability-subnet-validator"])
        config = build_config("validator")

        assert config.burn_percentage == 0.0
        assert not hasattr(C, "DEFAULT_BURN_PERCENTAGE")


class TestTheStaleTolerance:
    def test_it_allows_the_lag_the_protocol_builds_in(self):
        """A run is paid one run after it is measured, so the newest vector a
        healthy engine can offer already trails the chain by WEIGHT_LAG_RUNS.
        A tolerance at or below that would burn against a working engine."""
        from capability_subnet.validator import neuron

        assert "C.WEIGHT_LAG_RUNS + 1" in neuron.__loader__.get_source(neuron.__name__)


class TestEveryRefusalBurns:
    """The vector is refused for four distinct reasons; each burns."""

    @pytest.mark.parametrize(
        "reason",
        ["validate_vector", "spot_check_run", "check_draw_was_not_re_rolled"],
    )
    def test_a_failed_check_falls_back_to_burn(self, reason):
        from capability_subnet.validator import neuron

        source = neuron.__loader__.get_source(neuron.__name__)
        step = source[source.index("def _step_endpoint") : source.index("def _run_report_dir")]

        assert reason in step, f"{reason} is not consulted in endpoint mode"
        assert "safe_fallback" in step, "a refused vector must burn, not be skipped"

    def test_an_unreachable_endpoint_keeps_the_last_weights(self):
        """Not a burn. An endpoint that is down says nothing about the champion,
        and burning on a network blip pays nobody for a run that was fine."""
        from capability_subnet.validator import neuron

        source = neuron.__loader__.get_source(neuron.__name__)
        step = source[source.index("def _step_endpoint") : source.index("def _run_report_dir")]

        assert "BackendUnavailable" in step
        assert "leaving the last weights in force" in step


class TestPreflightIsLocalOnly:
    def test_endpoint_mode_does_not_demand_a_fleet(self):
        """The whole point: this mode exists for hosts without one."""
        from capability_subnet.validator import neuron

        source = neuron.__loader__.get_source(neuron.__name__)

        assert 'if self.mode == "local":\n            self._preflight_own_evaluation()' in source

    def test_local_mode_still_demands_one(self):
        from capability_subnet.validator.neuron import ValidatorNeuron

        config = SimpleNamespace(evaluation="own", serve_url="", pool_dir="pool", device="cuda")
        with pytest.raises(SystemExit):
            ValidatorNeuron._preflight_own_evaluation(SimpleNamespace(config=config))
