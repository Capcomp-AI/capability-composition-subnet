"""A validator that cannot measure must burn, not fall silent.

Going quiet is the one option strictly worse than either paying or burning. A
validator that stops submitting leaves its previous vector standing on chain, so
it keeps paying whoever it last named until the chain stops counting it — and
then pays nobody, while still looking alive.

Observed doing exactly that: the engine had published no vector yet, `/weights`
answered 404, and three validators sat silent for a day with the previous run's
weights still standing. That failure came from the delegated mode, which is gone;
the property it taught is not, because a validator measuring for itself has its
own way to be unable to proceed — it cannot read the block its run opened at,
and therefore has no draw.
"""

from __future__ import annotations


class TestAnUnreadableBeaconBurnsRatherThanReturning:
    def test_a_beacon_that_cannot_be_read_burns_this_run(self, monkeypatch):
        import capability_subnet.validator.neuron as module

        burned: list[str] = []

        class Neuron:
            config = type(
                "C",
                (),
                {
                    "disable_set_weights": False,
                    "netuid": 1,
                    "burn_percentage": 0.0,
                    "pool_dir": "pool",
                    "serve_url": "http://127.0.0.1:8000/v1",
                    "device": "cuda",
                },
            )()
            subtensor = None
            # Measuring here, which is what makes an unreadable beacon fatal:
            # endpoint mode never touches the chain for a draw.
            mode = "local"
            # The real method, so this exercises the beacon failure rather than a
            # stand-in for it.
            _step_own = module.ValidatorNeuron._step_own
            burn_uid = staticmethod(lambda: 0)
            _burn = staticmethod(lambda block, *, reason: burned.append(reason))
            resync = staticmethod(lambda: None)
            should_set_weights = staticmethod(lambda block: True)

        # The chain read that has no fallback: a fabricated beacon would be this
        # validator choosing which problems the candidates face.
        def no_beacon(subtensor, block):
            raise RuntimeError("chain unreachable")

        monkeypatch.setattr(module, "current_block", lambda st: 7_000_000)
        monkeypatch.setattr("capability_subnet.common.chain.block_beacon", no_beacon)

        module.ValidatorNeuron.step(Neuron())

        assert burned, "a validator with no draw must burn, not return"
        assert "no beacon" in burned[0]
