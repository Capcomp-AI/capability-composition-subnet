"""A measurement nobody took must not earn credit for having been taken.

The efficiency term is ``1 - peak/limit``, so the value chosen to stand in for
"unmeasured" decides whether the absence is a bonus or a penalty. Zero is the
tempting default and the wrong one: it is the best possible reading, awards the
maximum, and does so on every candidate a validator scores — which quietly makes
that validator's numbers incomparable with one that actually measured.
"""

from __future__ import annotations

from capability_subnet.common import constants as C
from capability_subnet.scoring.aggregate import artifact_efficiency

ARTIFACT = 333 * 1024 * 1024


class TestUnmeasuredVramIsNotABonus:
    def test_zero_would_have_paid_more_than_a_real_measurement(self):
        """The bug this defends against, stated as arithmetic."""
        unmeasured_as_zero = artifact_efficiency(ARTIFACT, 0.0)
        # Just under the gate, derived rather than hardcoded: the limit moved
        # from 24 to 32 GiB and a literal silently stopped being "near it".
        really_measured = artifact_efficiency(ARTIFACT, C.MAX_PEAK_VRAM_GB - 0.16)
        assert unmeasured_as_zero > really_measured * 3

    def test_the_limit_is_the_honest_stand_in(self):
        """Scoring an unmeasured package at the limit puts it at or below what a
        real measurement earns, so nothing is gained by not measuring."""
        unmeasured = artifact_efficiency(ARTIFACT, C.MAX_PEAK_VRAM_GB)
        for plausible in (18.0, 20.5, C.MAX_PEAK_VRAM_GB - 0.16):
            assert unmeasured <= artifact_efficiency(ARTIFACT, plausible)

    def test_the_evaluator_uses_the_limit_when_nothing_was_measured(self):
        """The default SandboxConfig leaves peak_vram_gb unset, which is exactly
        the case a validator hits before it has wired up NVML."""
        import inspect

        from capability_subnet.sandbox.orchestrator import SandboxConfig
        from capability_subnet.validator import evaluator

        assert SandboxConfig().peak_vram_gb is None, (
            "if this gains a numeric default, the substitution below stops being "
            "reachable and this test stops testing anything"
        )
        source = inspect.getsource(evaluator.evaluate_candidate)
        assert "C.MAX_PEAK_VRAM_GB if measured_vram is None" in source
        assert "config.peak_vram_gb or 0.0" not in source
