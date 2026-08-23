"""A host must ask enough instances to tell that the bar was cleared.

The resolvable effect falls with the square root of the instance count, so how
much of a draw a host asks and how large a margin a challenger must clear are
one decision. Split into two, they drift: a share that resolves 0.038 cannot
show that anything cleared 0.030, and every candidate reads as a near miss with
no way to tell a real one from noise.

Nothing here needs a GPU. The arithmetic is the whole point.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.scoring.comparator import minimum_detectable_effect
from capability_subnet.validator.assignment import assign

BEACON = "0x" + "5c" * 32
SEEDS = tuple(range(C.DEFAULT_HIDDEN_INSTANCES))


def resolvable(count: int) -> float:
    return minimum_detectable_effect(count)


class TestTheWholeDrawResolvesTheMargin:
    def test_the_shipped_instance_count_can_show_the_shipped_margin(self):
        assert resolvable(C.DEFAULT_HIDDEN_INSTANCES) < C.DEFAULT_END_TO_END_MARGIN

    def test_a_host_may_ask_all_of_it(self):
        whole = assign(SEEDS, hotkey="5Engine", beacon=BEACON, core_fraction=1.0, tail_fraction=0.0)

        assert len(whole.seeds) == C.DEFAULT_HIDDEN_INSTANCES
        assert resolvable(len(whole.seeds)) < C.DEFAULT_END_TO_END_MARGIN


class TestAValidatorsShareIsSmallerAndSaysSo:
    """Kept as a fact rather than a target. A validator asking a share is a
    deliberate design — a common core makes two of them comparable — and the
    consequence is that its own view resolves less than the whole draw does."""

    def test_the_default_share_is_the_core_plus_the_tail(self):
        share = assign(SEEDS, hotkey="5Validator", beacon=BEACON)
        expected = round(len(SEEDS) * (C.DEFAULT_CORE_FRACTION + C.DEFAULT_TAIL_FRACTION))

        assert len(share.seeds) == pytest.approx(expected, abs=2)

    def test_the_share_resolves_less_than_the_whole(self):
        share = assign(SEEDS, hotkey="5Validator", beacon=BEACON)

        assert resolvable(len(share.seeds)) > resolvable(C.DEFAULT_HIDDEN_INSTANCES)


class TestTheTwoConstantsAreOneDecision:
    def test_the_margin_sits_below_what_the_whole_draw_resolves(self):
        """If this fails, the bar cannot be shown to have been cleared by
        anyone, and lowering the instance count is what usually did it."""
        assert C.DEFAULT_END_TO_END_MARGIN > resolvable(C.DEFAULT_HIDDEN_INSTANCES), (
            f"{C.DEFAULT_HIDDEN_INSTANCES} instances resolve "
            f"{resolvable(C.DEFAULT_HIDDEN_INSTANCES):.4f}, which is wider than the "
            f"{C.DEFAULT_END_TO_END_MARGIN} margin a challenger has to clear"
        )

    def test_halving_the_draw_would_break_it(self):
        """The margin has this much room and no more, so a change to either
        constant is a change to both."""
        assert resolvable(C.DEFAULT_HIDDEN_INSTANCES // 2) > C.DEFAULT_END_TO_END_MARGIN
