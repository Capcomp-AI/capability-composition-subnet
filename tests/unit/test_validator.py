class TestAnUnavailableEngineStillProducesAVector:
    """A validator that stops submitting is worse than one that burns.

    Every other failure path in `step` burns. The unavailable-engine path
    returned instead, so when the engine had published nothing yet and
    `/weights` answered 404, three validators sat silent for a day with the
    previous run's weights still standing on chain.
    """

    def test_backend_unavailable_burns_rather_than_returning(self, monkeypatch):
        import capability_subnet.validator.neuron as module

        calls = []

        class Neuron:
            # Explicitly the delegated path: this is a property of fetching a
            # vector from a backend, and a validator measuring for itself has no
            # backend to be unavailable.
            config = type(
                "C",
                (),
                {
                    "disable_set_weights": False,
                    "spot_check": False,
                    "evaluation": "delegated",
                },
            )()
            burn_uid = staticmethod(lambda: 0)
            _burn = staticmethod(lambda block, *, reason: calls.append(reason))
            resync = staticmethod(lambda: None)
            should_set_weights = staticmethod(lambda block: True)

            class client:
                @staticmethod
                def fetch_weights():
                    raise module.BackendUnavailable("404 Not Found")

        neuron = Neuron()
        monkeypatch.setattr(module, "current_block", lambda st: 100)
        neuron.subtensor = None
        module.ValidatorNeuron.step(neuron)

        assert calls, "an unavailable engine must still produce a burn, not silence"
        assert "404" in calls[0]
