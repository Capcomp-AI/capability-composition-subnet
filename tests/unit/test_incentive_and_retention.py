"""The rules that decide who gets paid, and what stops the network stalling.

Every test here covers a decision that was previously made differently and
whose old behaviour was not visibly wrong — a ratchet that only stalls after
several dethrones, a retention gate that always returns the same number, a
weight vector that pays nobody but the winner. None of them would show up as a
crash, so they are pinned explicitly.
"""

from __future__ import annotations

from capability_subnet.backend.baselines import references as ref
from capability_subnet.backend.comparator.comparator import (
    ComparatorConfig,
    decayed_champion_margin,
)
from capability_subnet.backend.scorer import gates
from capability_subnet.backend.scorer.retention import (
    ProbeOutcome,
    build_probe,
    relative_retention,
)
from capability_subnet.backend.weights.weight_writer import survival_tail, winner_take_all
from capability_subnet.common import constants as C
from capability_subnet.common.schemas import ChampionRecord


class TestTheBarDoesNotRatchet:
    """A challenger clears the references, not every previous winner in turn."""

    def test_the_incumbent_is_excluded_from_the_absolute_bar(self):
        scores = {
            ref.BASE_MODEL: 0.30,
            ref.EQUAL_TIES: 0.40,
            ref.INCUMBENT: 0.85,
        }
        # With the incumbent folded in, the bar would be 0.85 and every new
        # champion would have to beat the last by a further fixed margin — a
        # staircase that completion, being bounded by one, cannot climb far.
        collapsed = ref.collapse_single_adapters(scores, include_incumbent=False)
        assert ref.INCUMBENT not in collapsed
        assert max(collapsed.values()) == 0.40

    def test_the_incumbent_is_still_reported(self):
        """Excluded from the bar, not hidden — the report still shows it."""
        scores = {ref.BASE_MODEL: 0.30, ref.INCUMBENT: 0.85}
        assert ref.INCUMBENT in ref.collapse_single_adapters(scores)


class TestTheDefendersAdvantageDecays:
    """Holding the throne is an advantage for a while, not a freehold."""

    def test_a_fresh_champion_holds_its_full_margin(self):
        config = ComparatorConfig()
        assert decayed_champion_margin(config, blocks_held=0) == config.champion_margin

    def test_the_margin_falls_over_the_decay_window(self):
        config = ComparatorConfig()
        half = decayed_champion_margin(config, blocks_held=config.champion_margin_decay_blocks // 2)
        assert 0.0 < half < config.champion_margin

    def test_an_unchallenged_champion_eventually_holds_no_advantage(self):
        """Otherwise one package holds the throne forever and buys nothing."""
        config = ComparatorConfig()
        assert (
            decayed_champion_margin(config, blocks_held=config.champion_margin_decay_blocks * 2)
            == 0.0
        )

    def test_decay_can_be_switched_off(self):
        config = ComparatorConfig(champion_margin_decay_blocks=0)
        assert decayed_champion_margin(config, blocks_held=10**9) == config.champion_margin


class TestQueuedMinersAreNotPrunedAway:
    """Zero emission is what the chain evicts first."""

    def test_the_tail_pays_every_queued_miner(self):
        tail = survival_tail([(3, "5A"), (4, "5B"), (5, "5C")], 0.2)
        assert [entry.uid for entry in tail] == [3, 4, 5]
        assert all(entry.weight > 0.0 for entry in tail)
        assert abs(sum(entry.weight for entry in tail) - 0.2) < 1e-9

    def test_the_front_of_the_queue_is_worth_most(self):
        """It is closest to being evaluated, so the order carries information."""
        tail = survival_tail([(3, "5A"), (4, "5B"), (5, "5C")], 0.2)
        weights = [entry.weight for entry in tail]
        assert weights == sorted(weights, reverse=True)

    def test_the_champion_is_not_paid_twice(self):
        tail = survival_tail([(3, "5A"), (4, "5B")], 0.2, exclude_uid=3)
        assert [entry.uid for entry in tail] == [4]

    def test_the_tail_is_paid_even_with_an_empty_throne(self):
        """The moment the queue most needs protecting is before anyone has won."""
        vector = winner_take_all(
            None,
            window_id=1,
            block=100,
            spec_version=1,
            tail=[(3, "5A"), (4, "5B")],
            tail_share=0.2,
        )
        roles = {entry.uid: entry.role for entry in vector.entries}
        assert roles[3] == "queued" and roles[4] == "queued"
        assert abs(sum(e.weight for e in vector.entries) - 1.0) < 1e-9

    def test_the_champion_still_takes_the_bulk(self):
        champion = ChampionRecord(candidate_id="5W", hotkey="5W", uid=9, crowned_at_block=1)
        vector = winner_take_all(
            champion,
            window_id=1,
            block=100,
            spec_version=1,
            tail=[(3, "5A"), (4, "5B")],
            tail_share=0.2,
        )
        by_uid = {entry.uid: entry.weight for entry in vector.entries}
        assert by_uid[9] > 0.75
        assert abs(sum(by_uid.values()) - 1.0) < 1e-9

    def test_a_zero_tail_share_reproduces_winner_take_all(self):
        champion = ChampionRecord(candidate_id="5W", hotkey="5W", uid=9, crowned_at_block=1)
        vector = winner_take_all(
            champion, window_id=1, block=100, spec_version=1, tail=[(3, "5A")], tail_share=0.0
        )
        assert [(e.uid, e.weight) for e in vector.entries] == [(9, 1.0)]


class TestRetentionMeasuresSomethingElse:
    """The gate has to ask a question the workflow score cannot answer."""

    def test_the_probe_is_deterministic_in_its_seed(self):
        """An auditor regenerates exactly what the candidate was asked."""
        assert [i.prompt for i in build_probe(4242)] == [i.prompt for i in build_probe(4242)]

    def test_different_windows_ask_different_things(self):
        assert [i.prompt for i in build_probe(1)] != [i.prompt for i in build_probe(2)]

    def test_the_probe_covers_more_than_one_behaviour(self):
        behaviours = {item.behaviour for item in build_probe(7)}
        assert len(behaviours) >= 4

    def test_an_answer_with_padding_is_wrong(self):
        """The failure a merge actually causes: still correct, no longer obedient."""
        item = build_probe(11)[0]
        assert item.matches(item.expected)
        assert not item.matches(f"Sure! The answer is {item.expected}.")

    def test_losing_probe_accuracy_lowers_retention(self):
        base = ProbeOutcome(correct=40, total=40)
        degraded = ProbeOutcome(correct=30, total=40)
        assert relative_retention(degraded, base) == 0.75
        # And that is below the floor, which is the whole point.
        assert relative_retention(degraded, base) < C.BASE_RETENTION_FLOOR

    def test_beating_the_base_is_capped_at_full_retention(self):
        base = ProbeOutcome(correct=30, total=40)
        better = ProbeOutcome(correct=40, total=40)
        assert relative_retention(better, base) == 1.0

    def test_an_unmeasurable_base_reports_full_retention(self):
        """Nothing to have destroyed, so the gate must not fire."""
        assert relative_retention(ProbeOutcome(correct=0, total=40), ProbeOutcome()) == 1.0


class TestEngineFailuresDoNotSpendAMinersShot:
    """One shot per hotkey is only defensible if the engine never charges for its own."""

    def test_an_unreadable_memory_counter_is_an_engine_failure(self):
        verdict = gates.gate_peak_vram(None, require_measurement=True)
        assert not verdict.passed
        assert verdict.name in gates.INFRASTRUCTURE_GATES

    def test_a_package_that_is_genuinely_too_large_is_not(self):
        verdict = gates.gate_peak_vram(C.MAX_PEAK_VRAM_GB + 10.0)
        assert not verdict.passed
        assert verdict.name not in gates.INFRASTRUCTURE_GATES

    def test_too_few_scored_instances_is_an_engine_failure(self):
        verdict = gates.gate_sample_sufficiency([], minimum=20)
        assert not verdict.passed
        assert verdict.name in gates.INFRASTRUCTURE_GATES

    def test_latency_with_nothing_to_measure_is_an_engine_failure(self):
        verdict = gates.gate_latency([])
        assert not verdict.passed
        assert verdict.name in gates.INFRASTRUCTURE_GATES

    def test_the_two_kinds_of_failure_are_separated(self):
        verdicts = [
            gates.gate_sample_sufficiency([], minimum=20),
            gates.gate_peak_vram(C.MAX_PEAK_VRAM_GB + 10.0),
        ]
        assert len(gates.infrastructure_failures(verdicts)) == 1
        assert len(gates.candidate_failures(verdicts)) == 1


class TestLosingWellIsWorthSomething:
    """Almost every submission fails to dethrone, and they are not all equal."""

    @staticmethod
    def _scores(e2e: float, qualified: float = 0.5, tokens: float = 0.5, latency: float = 0.5):
        from capability_subnet.common.schemas import CandidateScores

        return CandidateScores(
            end_to_end=e2e,
            qualified_score=qualified,
            token_efficiency=tokens,
            latency=latency,
        )

    def _inputs(self, e2e, *, reference=0.40, champion=0.60, **kw):
        from capability_subnet.backend.scorer.contribution import ContributionInputs

        return ContributionInputs(
            scores=self._scores(e2e, **kw), reference_e2e=reference, champion_e2e=champion
        )

    def test_a_near_miss_outgrades_a_hopeless_attempt(self):
        from capability_subnet.backend.scorer.contribution import contribution_score

        near = contribution_score(self._inputs(0.58, qualified=0.62))
        hopeless = contribution_score(self._inputs(0.10, qualified=0.12))
        assert near > hopeless

    def test_failing_to_beat_the_reference_earns_no_improvement_credit(self):
        from capability_subnet.backend.scorer.contribution import improvement_over_reference

        assert improvement_over_reference(0.30, 0.40) == 0.0
        assert improvement_over_reference(0.40, 0.40) == 0.0
        assert improvement_over_reference(0.70, 0.40) > 0.0

    def test_improvement_is_scaled_by_the_headroom_that_remained(self):
        """Moving 0.90 -> 0.95 is a larger achievement than 0.10 -> 0.15."""
        from capability_subnet.backend.scorer.contribution import improvement_over_reference

        high = improvement_over_reference(0.95, 0.90)
        low = improvement_over_reference(0.15, 0.10)
        assert high > low

    def test_an_empty_throne_gives_full_proximity(self):
        from capability_subnet.backend.scorer.contribution import proximity_to_champion

        assert proximity_to_champion(0.20, None) == 1.0

    def test_matching_the_champion_gives_full_proximity(self):
        from capability_subnet.backend.scorer.contribution import proximity_to_champion

        assert proximity_to_champion(0.60, 0.60) == 1.0
        assert proximity_to_champion(0.70, 0.60) == 1.0

    def test_a_cheaper_package_outgrades_an_identical_expensive_one(self):
        """Two packages that finish equally are not equally valuable."""
        from capability_subnet.backend.scorer.contribution import contribution_score

        cheap = contribution_score(self._inputs(0.55, tokens=0.9, latency=0.9))
        costly = contribution_score(self._inputs(0.55, tokens=0.1, latency=0.1))
        assert cheap > costly

    def test_the_grade_is_published_broken_into_its_terms(self):
        """A miner that earned a partial share must be able to see why."""
        from capability_subnet.backend.scorer.contribution import explain

        terms = explain(self._inputs(0.55))
        assert set(terms) == {"quality", "improvement", "proximity", "cost", "contribution"}
        assert all(0.0 <= v <= 1.0 for v in terms.values())


class TestTokenSpendIsScored:
    """It was measured and reported but never scored."""

    @staticmethod
    def _rows(count, *, completed, output_tokens):
        from capability_subnet.common.schemas import InstanceResult

        return [
            InstanceResult(
                instance_id=f"i-{i}",
                instance_seed=i,
                split="hidden",
                end_to_end_success=i < completed,
                output_tokens=output_tokens,
            )
            for i in range(count)
        ]

    def test_a_cheaper_package_scores_higher(self):
        from capability_subnet.backend.scorer.aggregate import token_efficiency

        cheap = token_efficiency(self._rows(10, completed=5, output_tokens=500))
        costly = token_efficiency(self._rows(10, completed=5, output_tokens=5000))
        assert cheap > costly

    def test_giving_up_early_does_not_look_cheap(self):
        """Charged per *completed* instance, so cheap failure is expensive.

        Dividing by attempts would make a package that quits after one turn the
        most efficient thing on the board.
        """
        from capability_subnet.backend.scorer.aggregate import token_efficiency

        finishes = token_efficiency(self._rows(10, completed=8, output_tokens=1000))
        quits = token_efficiency(self._rows(10, completed=1, output_tokens=1000))
        assert finishes > quits

    def test_completing_nothing_scores_zero(self):
        from capability_subnet.backend.scorer.aggregate import token_efficiency

        assert token_efficiency(self._rows(10, completed=0, output_tokens=100)) == 0.0


class TestAnUnmeasuredAdapterBlocksItselfNotTheNetwork:
    """Certification answers two questions that gate different things.

    Collapsing them meant one unmeasured adapter kept the whole arena closed.
    Separated, a structurally sound but uncharacterised adapter sits in the
    registry — safe to load, not selectable — until somebody measures it.
    """

    @staticmethod
    def _entry(adapter_id, *, certified=True, measured=True, distractor=False):
        from capability_subnet.registry.adapters import AdapterEntry, CertificationRecord

        return AdapterEntry(
            adapter_id=adapter_id,
            capability="c",
            description="",
            license="Apache-2.0",
            license_allows_derivatives=True,
            provenance="",
            training_data_ref="",
            training_data_date_range="",
            known_overlaps=(),
            artifact_uri="",
            artifact_sha256="sha256:" + "0" * 64,
            rank=64,
            lora_alpha=64,
            converted_from_rank=None,
            certified=certified,
            is_distractor=distractor,
            certification=CertificationRecord(
                capability_score=0.8 if measured else None,
                base_retention=0.99 if measured else None,
            ),
        )

    def _registry(self, entries):
        from capability_subnet.registry.adapters import AdapterRegistry

        return AdapterRegistry(
            registry_version=1,
            workflow_id=C.DEFAULT_WORKFLOW_ID,
            base_revision="rev",
            canonical_rank=64,
            canonical_lora_alpha=64,
            canonical_target_modules=C.CANONICAL_TARGET_MODULES,
            adapters=tuple(entries),
        )

    def test_a_measured_adapter_is_selectable(self):
        assert self._entry("a").selectable

    def test_an_unmeasured_adapter_is_loadable_but_not_selectable(self):
        entry = self._entry("a", measured=False)
        assert entry.certified and not entry.selectable

    def test_a_structurally_rejected_adapter_is_neither(self):
        """Structural admission is not negotiable — these tensors get loaded
        into a process that also holds hidden evaluation material."""
        entry = self._entry("a", certified=False, measured=True)
        assert not entry.selectable

    def test_the_pool_miners_draw_from_excludes_unmeasured_adapters(self):
        registry = self._registry([self._entry("measured"), self._entry("pending", measured=False)])
        assert registry.selectable_ids == ("measured",)
        assert set(registry.ids) == {"measured", "pending"}
        assert registry.unmeasured() == ("pending",)

    def test_references_are_never_built_from_unmeasured_weights(self):
        """A reference built from an uncharacterised adapter would set the bar
        every miner must clear using weights nobody has looked at."""
        registry = self._registry([self._entry("measured"), self._entry("pending", measured=False)])
        assert registry.capability_adapters() == ("measured",)

    def test_selecting_an_unmeasured_adapter_is_reported_distinctly(self):
        """A different problem from naming an adapter that does not exist, and
        with a different fix — so it gets its own message."""
        registry = self._registry([self._entry("measured"), self._entry("pending", measured=False)])
        assert registry.unselectable_ids(["measured", "pending"]) == ["pending"]
        assert registry.unknown_ids(["measured", "pending"]) == []

    def test_certifying_an_adapter_changes_the_snapshot_digest(self):
        """Selectability decides which recipes are valid, so it must be part of
        the pool's identity. Two engines disagreeing about it would admit
        different submissions while claiming the same pool."""
        before = self._registry([self._entry("a"), self._entry("b", measured=False)])
        after = self._registry([self._entry("a"), self._entry("b")])
        assert before.snapshot_sha256() != after.snapshot_sha256()
