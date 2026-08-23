"""How much of a draw a host asks, and what that decides.

Not whether a challenger can win. Every hard gate is arithmetic — a candidate
clears the bar when its completion exceeds the reference's by the margin — so
any bar is crossable at any draw size, and no statistical test stands between a
package and the throne.

What the draw size decides is whether the same package gets the same answer
twice. The resolvable effect falls with the square root of the count, and a
margin near or below it is settled partly by which instances came up. That is a
trade to make deliberately, not a configuration to refuse.

Nothing here needs a GPU. The arithmetic is the whole point.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.scoring import gates
from capability_subnet.scoring.comparator import minimum_detectable_effect
from capability_subnet.validator.assignment import assign

BEACON = "0x" + "5c" * 32
SEEDS = tuple(range(C.DEFAULT_HIDDEN_INSTANCES))


class TestTheBarIsArithmetic:
    """The reason a bar below the resolvable effect is allowed at all."""

    #: A hair over and a hair under the bar. Wider than a float's error at
    #: these magnitudes, which is a real edge: subtracting the margin from a
    #: completion rate lands a few parts in 10^17 below it, so a candidate
    #: exactly on the bar is decided by representation. Well inside the
    #: measurement noise either way, and not what these check.
    OVER = C.DEFAULT_END_TO_END_MARGIN + 1e-6
    UNDER = C.DEFAULT_END_TO_END_MARGIN - 1e-6

    def test_clearing_the_margin_by_a_hair_passes(self):
        verdict = gates.gate_beats_strongest_reference(
            0.10 + self.OVER, "reference:base_model", 0.10, C.DEFAULT_END_TO_END_MARGIN
        )
        assert verdict.passed

    def test_missing_it_by_a_hair_fails(self):
        verdict = gates.gate_beats_strongest_reference(
            0.10 + self.UNDER, "reference:base_model", 0.10, C.DEFAULT_END_TO_END_MARGIN
        )
        assert not verdict.passed

    def test_no_statistical_test_stands_between_a_candidate_and_the_throne(self):
        """`gate_statistics` exists and is applied nowhere.

        It belonged to a paired-bootstrap dethrone rule that the grade ladder
        replaced. While it was live, a margin under the resolvable effect
        genuinely could never be confirmed; the constraint outlived the rule
        and was still being enforced against configurations that work.
        """
        from pathlib import Path

        from capability_subnet.validator import evaluator

        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        applied = source[source.index("verdicts = [") :]
        applied = applied[: applied.index("]")]

        assert "gate_statistics" not in applied
        assert "gate_beats_strongest_reference" in applied


class TestTheDrawDecidesReproducibility:
    def test_a_host_may_ask_the_whole_draw(self):
        whole = assign(SEEDS, hotkey="5Engine", beacon=BEACON, core_fraction=1.0, tail_fraction=0.0)

        assert len(whole.seeds) == C.DEFAULT_HIDDEN_INSTANCES

    def test_the_default_share_is_the_core_plus_the_tail(self):
        share = assign(SEEDS, hotkey="5Validator", beacon=BEACON)
        expected = round(len(SEEDS) * (C.DEFAULT_CORE_FRACTION + C.DEFAULT_TAIL_FRACTION))

        assert len(share.seeds) == pytest.approx(expected, abs=2)

    def test_a_share_resolves_less_than_the_whole(self):
        """So two hosts asking different amounts disagree more often, even
        measuring the same package on the same run."""
        share = assign(SEEDS, hotkey="5Validator", beacon=BEACON)

        assert minimum_detectable_effect(len(share.seeds)) > minimum_detectable_effect(
            C.DEFAULT_HIDDEN_INSTANCES
        )

    def test_a_zero_core_is_still_refused(self):
        """A host sharing no instances with any other cannot be compared to one."""
        with pytest.raises(ValueError, match="core_fraction"):
            assign(SEEDS, hotkey="5Nobody", beacon=BEACON, core_fraction=0.0)


def test_the_shipped_bar_sits_inside_the_shipped_noise():
    """Recorded rather than asserted away, because it is a live trade-off.

    At 1350 instances the draw resolves about 0.024 and the bar is 0.02, so a
    package whose true edge is the bar clears on some draws and misses on
    others. That was chosen: a lower barrier to entry, paid for in verdicts
    that do not always reproduce. Raising the draw to about 1970 would buy the
    reproducibility back.
    """
    resolvable = minimum_detectable_effect(C.DEFAULT_HIDDEN_INSTANCES)

    assert C.DEFAULT_END_TO_END_MARGIN < resolvable, (
        "the bar is now clear of the noise; the trade recorded here no longer applies"
    )
    assert resolvable < 0.03
