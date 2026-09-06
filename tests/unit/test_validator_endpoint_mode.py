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


class TestARunBoundaryIsItsOwnTrigger:
    """The interval is a floor on rewrites, not a schedule the run waits for.

    A run pays a different field from the one before it. When the boundary moves
    and the only thing that can start a pass is a 30-minute timer, the validator
    holds the previous run's winners until the timer happens to fire — and if an
    epoch step lands in that gap it is scored against a consensus that has
    already moved on, for weights that were right when it set them.
    """

    @staticmethod
    def _neuron(*, last_block, last_run, interval=150):
        return SimpleNamespace(
            config=SimpleNamespace(disable_set_weights=False, weight_interval=interval),
            last_weight_block=last_block,
            last_paying_run=last_run,
            run_blocks=C.DEFAULT_RUN_BLOCKS,
        )

    def test_a_new_run_starts_a_pass_inside_the_interval(self):
        from capability_subnet.common.chain import run_opens_block
        from capability_subnet.validator.neuron import ValidatorNeuron

        opened = run_opens_block(426, C.DEFAULT_RUN_BLOCKS)
        neuron = self._neuron(last_block=opened - 60, last_run=425)

        assert ValidatorNeuron.should_set_weights(neuron, opened + 3) is True

    def test_the_same_run_still_waits_for_the_interval(self):
        from capability_subnet.common.chain import run_opens_block
        from capability_subnet.validator.neuron import ValidatorNeuron

        opened = run_opens_block(426, C.DEFAULT_RUN_BLOCKS)
        neuron = self._neuron(last_block=opened + 1, last_run=426)

        assert ValidatorNeuron.should_set_weights(neuron, opened + 3) is False

    def test_the_interval_alone_is_still_enough(self):
        from capability_subnet.validator.neuron import ValidatorNeuron

        neuron = self._neuron(last_block=1_000, last_run=426)

        assert ValidatorNeuron.should_set_weights(neuron, 1_150) is True

    def test_a_validator_that_has_never_submitted_does_not_fire_early(self):
        """``None`` is "no submission yet", not "the run changed"."""
        from capability_subnet.validator.neuron import ValidatorNeuron

        neuron = self._neuron(last_block=1_000, last_run=None)

        assert ValidatorNeuron.should_set_weights(neuron, 1_010) is False

    def test_disabled_still_beats_a_boundary(self):
        from capability_subnet.validator.neuron import ValidatorNeuron

        neuron = self._neuron(last_block=0, last_run=425)
        neuron.config.disable_set_weights = True

        assert ValidatorNeuron.should_set_weights(neuron, 9_009_470) is False


class TestALaggingNodeDoesNotPayTheWrongRun:
    """Which run this is comes from a block height, so a behind node moves the
    validator into the previous run and it pays that run's winners with total
    confidence. Every downstream check passes — against the wrong run.
    """

    @staticmethod
    def _client(*, current_run, asked):
        def fetch_reports(run_id):
            asked.append(run_id)
            return []

        return SimpleNamespace(
            run_blocks=lambda: C.DEFAULT_RUN_BLOCKS,
            health=lambda: {"current_run": current_run},
            fetch_reports=fetch_reports,
        )

    def _run(self, monkeypatch, *, own_block, engine_run):
        """Returns the runs the validator went on to ask for reports about."""
        from capability_subnet.validator import client as client_module
        from capability_subnet.validator import neuron as neuron_module
        from capability_subnet.validator.neuron import ValidatorNeuron

        asked: list[int] = []
        monkeypatch.setattr(
            client_module,
            "BackendClient",
            lambda *a, **k: self._client(current_run=engine_run, asked=asked),
        )
        monkeypatch.setattr(neuron_module, "current_block", lambda _subtensor: own_block)

        me = SimpleNamespace(
            config=SimpleNamespace(backend_url="https://engine.example"),
            _trusted={"5ABC"},
            subtensor=object(),
            run_blocks=C.DEFAULT_RUN_BLOCKS,
            _published_run=ValidatorNeuron._published_run,
            _submit=lambda vector, block: pytest.fail("nothing should be submitted here"),
        )
        ValidatorNeuron._step_endpoint(me, own_block)
        return asked

    def test_it_holds_rather_than_paying_the_run_the_chain_has_left(self, monkeypatch):
        """The node still reads run 425 while the engine publishes run 426."""
        from capability_subnet.common.chain import run_opens_block

        behind = run_opens_block(425, C.DEFAULT_RUN_BLOCKS) + 10

        assert self._run(monkeypatch, own_block=behind, engine_run=426) == []

    def test_an_engine_behind_this_node_is_not_a_reason_to_stop(self, monkeypatch):
        """Only this validator being behind is dangerous.

        The other way round means the engine has not opened the run yet, and an
        empty report stream already handles that - so the pass carries on and
        asks, rather than refusing on the operator's say-so.
        """
        from capability_subnet.common.chain import run_opens_block

        ahead = run_opens_block(426, C.DEFAULT_RUN_BLOCKS) + 10

        assert self._run(monkeypatch, own_block=ahead, engine_run=425) == [425]


class TestTheCrossCheckIsAHintNotAnAuthority:
    def test_an_engine_that_will_not_say_disables_the_check(self):
        from capability_subnet.validator.neuron import ValidatorNeuron

        client = SimpleNamespace(health=lambda: {})

        assert ValidatorNeuron._published_run(client) is None

    def test_nonsense_is_not_a_run_number(self):
        from capability_subnet.validator.neuron import ValidatorNeuron

        client = SimpleNamespace(health=lambda: {"current_run": "soon"})

        assert ValidatorNeuron._published_run(client) is None

    def test_an_unreachable_engine_is_not_a_lag_signal(self):
        from capability_subnet.validator.client import BackendUnavailable
        from capability_subnet.validator.neuron import ValidatorNeuron

        def unreachable():
            raise BackendUnavailable("down")

        client = SimpleNamespace(health=unreachable)

        assert ValidatorNeuron._published_run(client) is None
