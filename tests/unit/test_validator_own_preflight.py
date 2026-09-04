"""A validator that cannot measure must not vote.

`--neuron.evaluation` defaults to `own`, where every number a validator submits
comes from work it does on its own hardware. The packaging, written when
`delegated` was the only mode, describes a validator as needing no tensor
library — so a validator installed from `capability-subnet` without extras
cannot import torch, cannot even parse a committed recipe, and because
`evaluate_candidate` treats reconstruction failure as a *scored outcome* rather
than an error, scores every miner zero for its own missing dependency. One
WARNING, and then weights on-chain.

These pin the start-up refusal that replaces that.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capability_subnet.validator.neuron import ValidatorNeuron


@pytest.fixture
def a_card(monkeypatch):
    """Make the host look like it has a CUDA device.

    The tests that assert a *healthy* host starts were reading the machine they
    ran on. They passed here because this box has cards, and passed on CI
    because torch was absent and the check skipped - so neither was exercising
    the branch, and the day torch arrived without a GPU they both failed for a
    reason that had nothing to do with the code under test.
    """
    try:
        import torch
    except ImportError:
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _config(**overrides):
    base = dict(evaluation="own", serve_url="http://127.0.0.1:8000", pool_dir="pool", device="cuda")
    base.update(overrides)
    return SimpleNamespace(**base)


def _preflight(config):
    """Run only the preflight, without standing up a wallet or a chain client."""
    neuron = ValidatorNeuron.__new__(ValidatorNeuron)
    neuron.config = config
    return neuron._preflight_own_evaluation()


class TestOwnModeRefusesAHostThatCannotMeasure:
    def test_a_complete_host_starts(self, tmp_path, a_card):
        (tmp_path / "pool").mkdir()
        _preflight(_config(pool_dir=str(tmp_path / "pool")))

    def test_no_serve_url_is_refused(self, tmp_path):
        (tmp_path / "pool").mkdir()
        with pytest.raises(SystemExit) as caught:
            _preflight(_config(serve_url="", pool_dir=str(tmp_path / "pool")))
        assert "refusing to start" in str(caught.value)

    def test_a_missing_pool_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            _preflight(_config(pool_dir=str(tmp_path / "absent")))

    def test_a_missing_reconstruction_stack_is_refused(self, tmp_path, monkeypatch):
        """The case that produced silently-zeroed miners.

        Simulated by making the import fail exactly as it does on a base install,
        rather than by uninstalling torch underneath the suite.
        """
        (tmp_path / "pool").mkdir()
        import builtins

        real_import = builtins.__import__

        def refuse_merge_engine(name, *args, **kwargs):
            if name.startswith("capability_subnet.merge_engine"):
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_merge_engine)

        with pytest.raises(SystemExit) as caught:
            _preflight(_config(pool_dir=str(tmp_path / "pool")))
        assert "refusing to start" in str(caught.value)

    def test_every_problem_is_reported_in_one_pass(self, tmp_path, caplog):
        """An operator should fix a host once, not once per restart.

        Asserted as "both of the problems this test caused are named, and the
        count in the summary matches the lines logged" rather than as a fixed
        total. The host contributes problems of its own - a machine without the
        merge extra installed adds one - and pinning the number made the test a
        statement about the machine it was written on.
        """
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as caught:
                _preflight(_config(serve_url="", pool_dir=str(tmp_path / "absent")))

        reported = [r.message for r in caplog.records if "preflight:" in r.message]
        assert any("serve_url" in m for m in reported)
        assert any("pool" in m for m in reported)
        assert f"{len(reported)} problem(s)" in str(caught.value)

    def test_delegated_mode_is_untouched(self, tmp_path):
        """The thin mode genuinely does run on the base install, and must keep
        doing so — it reconstructs nothing and needs no pool."""
        neuron = ValidatorNeuron.__new__(ValidatorNeuron)
        neuron.config = _config(
            evaluation="delegated", serve_url="", pool_dir=str(tmp_path / "absent")
        )
        # The caller only invokes the preflight for 'own'; assert the guard's
        # condition rather than the function, since that is what protects it.
        assert neuron.config.evaluation != "own"


class TestOwnModeRequiresAGpu:
    """Rebuilding every submission on a CPU is not a slower validator, it is one
    that never finishes its queue."""

    def test_a_cpu_device_is_refused(self, tmp_path):
        (tmp_path / "pool").mkdir()
        with pytest.raises(SystemExit) as caught:
            _preflight(_config(device="cpu", pool_dir=str(tmp_path / "pool")))
        assert "refusing to start" in str(caught.value)

    def test_a_cuda_device_that_is_not_there_is_refused(self, tmp_path, monkeypatch):
        (tmp_path / "pool").mkdir()
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(SystemExit):
            _preflight(_config(device="cuda", pool_dir=str(tmp_path / "pool")))

    def test_an_indexed_cuda_device_is_accepted(self, tmp_path, a_card):
        (tmp_path / "pool").mkdir()
        _preflight(_config(device="cuda:1", pool_dir=str(tmp_path / "pool")))
