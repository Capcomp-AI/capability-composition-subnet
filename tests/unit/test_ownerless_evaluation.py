"""Evaluation with no operator in it.

The properties below are the ones that make an owner unnecessary rather than
merely absent. A draw nobody holds a secret for has to be reproducible by
strangers and unpredictable before it exists; an assignment has to be checkable
by anyone and unshoppable by the validator it belongs to; and agreement has to
tolerate two honest validators on different silicon while still catching one
that did not do the work.
"""

from __future__ import annotations

import pytest

from capability_subnet.scoring.sampler import draw_run_open
from capability_subnet.validator.agreement import (
    DEFAULT_DISCORDANCE_BAND,
    compare,
    observed_discordance,
    outlier_validators,
)
from capability_subnet.validator.assignment import (
    assign,
    coverage,
)

BEACON = "0x" + "ab" * 32
OTHER_BEACON = "0x" + "cd" * 32


class TestTheDrawNeedsNobodysSecret:
    def test_two_strangers_derive_the_same_run(self):
        """The whole point: no operator, no shared secret, same instances."""
        a = draw_run_open(1080, beacon=BEACON, hidden_count=40, ood_count=10)
        b = draw_run_open(1080, beacon=BEACON, hidden_count=40, ood_count=10)
        assert a.hidden_seeds == b.hidden_seeds
        assert a.ood_seeds == b.ood_seeds
        assert a.probe_seed == b.probe_seed

    def test_it_commits_to_nothing_because_it_holds_nothing(self):
        assert draw_run_open(1080, beacon=BEACON, hidden_count=5, ood_count=1).root_commitment == ""

    def test_a_different_block_is_a_different_run(self):
        """A miner who has seen one run learns nothing about the next."""
        a = draw_run_open(1080, beacon=BEACON, hidden_count=40, ood_count=10)
        b = draw_run_open(1080, beacon=OTHER_BEACON, hidden_count=40, ood_count=10)
        assert set(a.hidden_seeds).isdisjoint(b.hidden_seeds)

    def test_hidden_and_ood_do_not_collide(self):
        s = draw_run_open(1080, beacon=BEACON, hidden_count=40, ood_count=10)
        assert set(s.hidden_seeds).isdisjoint(s.ood_seeds)
        assert len(set(s.hidden_seeds)) == 40

    def test_an_empty_beacon_is_refused(self):
        """With no secret to fall back on, an empty beacon is one fixed draw
        for every run of every deployment."""
        with pytest.raises(ValueError, match="beacon"):
            draw_run_open(1080, beacon="", hidden_count=5, ood_count=1)


class TestAssignmentIsPublicAndUnshoppable:
    SEEDS = tuple(range(1, 401))

    def test_every_validator_measures_the_same_core(self):
        """The core is what makes two validators comparable, so it cannot
        depend on who they are."""
        a = assign(self.SEEDS, hotkey="5AAA", beacon=BEACON)
        b = assign(self.SEEDS, hotkey="5BBB", beacon=BEACON)
        assert a.core == b.core

    def test_a_validator_cannot_shop_for_an_easier_core(self):
        cores = {assign(self.SEEDS, hotkey=f"5KEY{i}", beacon=BEACON).core for i in range(25)}
        assert len(cores) == 1

    def test_tails_differ_so_coverage_grows(self):
        a = assign(self.SEEDS, hotkey="5AAA", beacon=BEACON)
        b = assign(self.SEEDS, hotkey="5BBB", beacon=BEACON)
        assert a.tail != b.tail

    def test_the_tail_never_re_measures_the_core(self):
        a = assign(self.SEEDS, hotkey="5AAA", beacon=BEACON)
        assert set(a.core).isdisjoint(a.tail)

    def test_an_assignment_stays_inside_the_run(self):
        a = assign(self.SEEDS, hotkey="5AAA", beacon=BEACON)
        assert set(a.seeds) <= set(self.SEEDS)

    def test_anyone_can_recompute_what_a_validator_owed(self):
        """Nothing in the assignment is private, so a third party can check
        that a validator measured what it was supposed to."""
        first = assign(self.SEEDS, hotkey="5AAA", beacon=BEACON)
        second = assign(self.SEEDS, hotkey="5AAA", beacon=BEACON)
        assert first == second

    def test_validators_buy_coverage_rather_than_repetition(self):
        one = [assign(self.SEEDS, hotkey="5AAA", beacon=BEACON)]
        many = [assign(self.SEEDS, hotkey=f"5KEY{i}", beacon=BEACON) for i in range(12)]
        assert coverage(many, len(self.SEEDS)) > coverage(one, len(self.SEEDS))

    def test_a_nonsense_fraction_is_refused(self):
        with pytest.raises(ValueError):
            assign(self.SEEDS, hotkey="5AAA", beacon=BEACON, core_fraction=0.0)

    def test_it_survives_a_run_smaller_than_the_fractions(self):
        tiny = assign((1, 2), hotkey="5AAA", beacon=BEACON)
        assert len(tiny.core) >= 1
        assert set(tiny.seeds) <= {1, 2}


class TestAgreementToleratesHardwareButNotIdleness:
    CORE = tuple(range(1, 201))

    def _results(self, flip: int = 0) -> dict[int, bool]:
        out = {seed: seed % 3 == 0 for seed in self.CORE}
        for seed in list(self.CORE)[:flip]:
            out[seed] = not out[seed]
        return out

    def test_identical_validators_agree(self):
        v = compare(self._results(), self._results())
        assert v.consistent and v.discordant == 0

    def test_a_few_instances_flipping_is_device_divergence(self):
        """Six of seven merge methods are device-dependent by measurement, so
        a small disagreement must not be an accusation."""
        v = compare(self._results(), self._results(flip=8))
        assert v.consistent, v.detail

    def test_a_validator_that_did_not_do_the_work_is_caught(self):
        v = compare(self._results(), self._results(flip=120))
        assert not v.consistent
        assert "too far apart" in v.detail

    def test_too_few_shared_instances_is_not_a_verdict(self):
        a = {1: True, 2: False, 3: True}
        b = {1: False, 2: True, 3: False}
        v = compare(a, b)
        assert v.consistent and "not enough to judge" in v.detail

    def test_instances_only_one_side_measured_are_not_evidence(self):
        a = {seed: True for seed in range(1, 101)}
        b = {seed: True for seed in range(50, 200)}
        paired, discordant = observed_discordance(a, b)
        assert paired == 51 and discordant == 0

    def test_no_shared_instances_is_not_a_verdict(self):
        v = compare({1: True}, {2: True})
        assert v.consistent and v.paired == 0


class TestTheOddOneOutIsTheOneFlagged:
    CORE = tuple(range(1, 201))

    def _honest(self) -> dict[int, bool]:
        return {seed: seed % 3 == 0 for seed in self.CORE}

    def _liar(self) -> dict[int, bool]:
        return {seed: seed % 7 == 0 for seed in self.CORE}

    def test_the_liar_is_named_and_the_majority_is_not(self):
        flagged = outlier_validators(
            {"5A": self._honest(), "5B": self._honest(), "5C": self._honest(), "5D": self._liar()}
        )
        assert set(flagged) == {"5D"}

    def test_a_network_that_agrees_accuses_nobody(self):
        assert outlier_validators({k: self._honest() for k in ("5A", "5B", "5C")}) == {}

    def test_two_validators_cannot_form_a_majority(self):
        """With one peer each, "most of the others" is meaningless, and the
        wrong one would be as likely to be named as the right one."""
        assert outlier_validators({"5A": self._honest(), "5B": self._liar()}) == {}

    def test_a_wider_band_forgives_more(self):
        results = {
            "5A": self._honest(),
            "5B": self._honest(),
            "5C": self._honest(),
            "5D": self._liar(),
        }
        assert outlier_validators(results, band=DEFAULT_DISCORDANCE_BAND)
        assert outlier_validators(results, band=0.95) == {}
