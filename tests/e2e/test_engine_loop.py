"""The whole engine, end to end.

Everything real except the model: a synthetic adapter pool, real reconstruction,
real workflow instances, the real sandbox, the real scorer, the real comparator,
the real store, real reports and real weight vectors. The candidates are scripted
solvers of known quality, which is what makes the assertions possible — the test
knows in advance which package *ought* to win.

The scripted solver stands in for a served model. Substituting it changes what a
candidate is good at, not how the engine decides, and how the engine decides is
what this file is about.
"""

from __future__ import annotations

import pytest

from capability_subnet.backend.baselines import references as ref
from capability_subnet.common import constants as C
from capability_subnet.common.hashing import canonical_json_str
from capability_subnet.common.schemas import ChampionRecord, QueueEntry
from capability_subnet.workflows import get_workflow
from tests.engine_fixture import _artifact_name_for, _store_recipe


class TestReferenceMeasurement:
    def test_opening_a_window_measures_every_reference(self, engine):
        loop, store, server, _ = engine
        state = loop.ensure_window(block=250)

        assert state.window_id == 2
        assert ref.BASE_MODEL in state.reference_scores

        # Every equal-weight merge, the owner recipe, and one entry per single
        # adapter.
        singles = [name for name in state.reference_scores if name.startswith(ref.BEST_SINGLE)]
        # Rotated, not exhaustive. Measuring every single-adapter reference each
        # window is correct and costs most of the window's GPU budget before any
        # challenger is looked at; a window that cannot finish never evaluates
        # anybody. Which ones are measured comes from the window id, so it is not
        # the operator's choice.
        expected = min(
            loop.settings.single_adapter_rotation,
            len(loop.snapshot.registry.capability_adapters()),
        )
        assert len(singles) == expected
        for reference in (ref.EQUAL_LINEAR, ref.EQUAL_TIES, ref.EQUAL_DARE_TIES, ref.OWNER_RECIPE):
            assert reference in state.reference_scores

    def test_reference_sample_rows_are_retained(self, engine):
        loop, store, _, _ = engine
        state = loop.ensure_window(block=250)

        rows = store.load_samples(state.window_id, ref.BASE_MODEL)
        assert len(rows) == 4

    def test_a_new_window_redraws_the_instances(self, engine):
        loop, store, _, settings = engine

        first = loop.ensure_window(block=150)
        second = loop.ensure_window(block=250)

        assert first.window_id != second.window_id
        first_seeds = {instance.seed for instance in first.hidden_instances}
        second_seeds = {instance.seed for instance in second.hidden_instances}
        assert first_seeds.isdisjoint(second_seeds)

    def test_the_same_window_is_not_re_measured(self, engine):
        loop, _, server, _ = engine
        loop.ensure_window(block=250)
        served = len(server.served)

        loop.ensure_window(block=260)  # same window
        assert len(server.served) == served


class TestChallengerEvaluation:
    def _admit(self, store, settings, recipe, hotkey="5Challenger", uid=3, block=10):
        _store_recipe(settings, recipe)
        store.upsert_queue_entry(
            QueueEntry(
                hotkey=hotkey,
                uid=uid,
                recipe_sha256=recipe.digest(),
                recipe_uri="https://example.test/r.json",
                first_block=block,
                admitted_at_block=block,
            )
        )

    def test_a_competent_challenger_takes_an_empty_throne(self, engine, recipe_factory):
        loop, store, server, settings = engine
        # Coefficients that make this a genuinely different package from the
        # equal-weight reference merges — without them the recipe reconstructs to
        # the same bytes as a baseline and cannot, by construction, beat it.
        recipe = recipe_factory(seed=11, global_weights={"alpha-capability-v1": 1.35})
        self._admit(store, settings, recipe)

        # Everything on the board reserves the wrong part; only the challenger's
        # package gets it right.
        server.default_impairments = frozenset({"wrong_part"})
        server.impairments_by_artifact = {_artifact_name_for(loop, recipe): frozenset()}

        output = loop.evaluate_next_challenger(block=250)
        assert output is not None and output.usable

        champion = store.get_champion()
        assert champion is not None, output.gate_verdicts
        assert champion.hotkey == "5Challenger"
        assert store.get_queue_entry("5Challenger").status == "champion"

    def test_a_recipe_identical_to_a_baseline_cannot_win(self, engine, recipe_factory):
        """Resubmitting an equal-weight merge is not a contribution.

        The default recipe here is an equal-weight trimmed merge over every
        capability adapter — which is exactly one of the permanent references. It
        reconstructs to the same bytes, so however well it scores it cannot clear
        the strongest reference by a margin, and it is terminated.

        This is the same property that makes copying the champion worthless.
        """
        loop, store, server, settings = engine
        recipe = recipe_factory(seed=11)  # no distinguishing coefficients
        self._admit(store, settings, recipe)

        server.default_impairments = frozenset({"wrong_part"})
        server.impairments_by_artifact = {_artifact_name_for(loop, recipe): frozenset()}

        loop.evaluate_next_challenger(block=250)

        assert store.get_champion() is None
        assert store.get_queue_entry("5Challenger").status == "terminated"

    def test_no_challenger_can_beat_references_that_are_already_perfect(
        self, engine, recipe_factory
    ):
        # The bar is real: when an equal-weight merge already completes every
        # instance, there is nothing left for composition to add, and the correct
        # outcome is that nobody is crowned.
        loop, store, server, settings = engine
        recipe = recipe_factory(seed=13)
        self._admit(store, settings, recipe)

        loop.evaluate_next_challenger(block=250)

        assert store.get_champion() is None
        assert store.get_queue_entry("5Challenger").status != "champion"

    def test_a_challenger_that_cannot_beat_the_references_is_terminated(
        self, engine, recipe_factory
    ):
        loop, store, server, settings = engine
        recipe = recipe_factory(seed=12)
        self._admit(store, settings, recipe)

        # The challenger is the only impaired package on the board.
        server.impairments_by_artifact = {
            _artifact_name_for(loop, recipe): frozenset({"wrong_part", "bad_sql"})
        }

        output = loop.evaluate_next_challenger(block=250)

        assert output is not None
        assert not output.gates_passed
        assert store.get_queue_entry("5Challenger").status == "terminated"
        assert store.get_champion() is None

    def test_a_terminated_challenger_advances_the_queue(self, engine, recipe_factory):
        loop, store, server, settings = engine

        first = recipe_factory(seed=21)
        second = recipe_factory(seed=22)
        self._admit(store, settings, first, hotkey="5First", uid=3, block=10)
        self._admit(store, settings, second, hotkey="5Second", uid=4, block=20)

        server.impairments_by_artifact = {
            _artifact_name_for(loop, first): frozenset({"wrong_part", "bad_sql"})
        }

        loop.evaluate_next_challenger(block=250)
        assert store.get_queue_entry("5First").status == "terminated"

        # The queue head is now the second commitment, in commit order.
        assert store.next_challenger().hotkey == "5Second"

    def test_evaluation_stores_sample_rows_and_a_report(self, engine, recipe_factory):
        loop, store, server, settings = engine
        recipe = recipe_factory(seed=31)
        self._admit(store, settings, recipe)

        loop.ensure_window(block=250)
        loop.evaluate_next_challenger(block=250)

        assert store.has_samples(2, "5Challenger")
        reports = store.list_reports(window_id=2)
        assert any(report.candidate_id == "5Challenger" for _, report in reports)

    def test_the_compatibility_history_records_the_recipe(self, engine, recipe_factory):
        loop, store, _, settings = engine
        recipe = recipe_factory(seed=41, global_weights={"alpha-capability-v1": 1.25})
        self._admit(store, settings, recipe)

        loop.ensure_window(block=250)
        loop.evaluate_next_challenger(block=250)

        records = store.load_compatibility()
        assert records
        record = records[0]
        assert record["combination_type"] == recipe.merge.combination_type
        assert record["global_weights"]["alpha-capability-v1"] == pytest.approx(1.25)
        assert set(record["contribution_by_group"]) == set(loop.snapshot.manifest.layer_groups)

    def test_an_infrastructure_failure_returns_the_candidate_to_the_queue(
        self, engine, recipe_factory
    ):
        loop, store, server, settings = engine
        recipe = recipe_factory(seed=51)
        self._admit(store, settings, recipe)
        loop.ensure_window(block=250)

        from capability_subnet.backend.executor.serving import ServingError

        def _broken(adapter_path):
            raise ServingError("the endpoint fell over")

        server.serve = _broken

        output = loop.evaluate_next_challenger(block=250)

        assert output is not None and not output.usable
        # The one shot must not be spent on the engine's bad night.
        assert store.get_queue_entry("5Challenger").status == "queued"


class TestWeightPublication:
    def test_an_empty_throne_burns(self, engine):
        loop, store, _, _ = engine
        loop.publish_weights(block=250)

        vector = store.latest_weights()
        assert vector is not None
        # Nothing has been evaluated, so there is nothing to grade and nobody
        # queued: the whole share burns.
        assert [e.uid for e in vector.entries] == [C.BURN_UID]

    def test_a_champion_takes_the_base_share_and_the_rest_burns(self, engine):
        """Under the graded mode an uncontested champion is not paid everything.

        The graded pool exists to pay demonstrated contribution. With no
        qualified contributors it is burned rather than folded into the
        champion's share — holding an uncontested throne is not an achievement
        the network should pay a bonus for.
        """
        loop, store, _, _ = engine
        store.set_champion(ChampionRecord(candidate_id="5Winner", hotkey="5Winner", uid=6))
        loop.publish_weights(block=250)

        vector = store.latest_weights()
        by_uid = {e.uid: e.weight for e in vector.entries}
        assert by_uid[6] == pytest.approx(loop.settings.champion_base_share)
        assert by_uid[C.BURN_UID] == pytest.approx(1.0 - loop.settings.champion_base_share)
        assert vector.champion_hotkey == "5Winner"

    def test_winner_take_all_remains_available(self, engine):
        """The old mode is still selectable, and still pays everything."""
        import dataclasses

        loop, store, _, _ = engine
        loop.settings = dataclasses.replace(
            loop.settings, incentive_mode=C.MODE_WINNER_TAKE_ALL, tail_share=0.0
        )
        store.set_champion(ChampionRecord(candidate_id="5Winner", hotkey="5Winner", uid=6))
        loop.publish_weights(block=250)

        vector = store.latest_weights()
        assert vector.entries[0].uid == 6
        assert vector.entries[0].weight == pytest.approx(1.0)

    def test_the_published_vector_is_well_formed(self, engine):
        loop, store, _, _ = engine
        store.set_champion(ChampionRecord(candidate_id="5Winner", hotkey="5Winner", uid=6))
        loop.publish_weights(block=250)

        vector = store.latest_weights()
        uids, weights = vector.as_uid_weight_lists()
        assert len(uids) == len(weights)
        assert sum(weights) == pytest.approx(1.0)
        assert len(set(uids)) == len(uids)


class TestStoreIntegrity:
    def test_a_champion_is_written_with_its_justifying_report(self, store):
        from capability_subnet.backend.evaluation import EvaluationOutput
        from capability_subnet.backend.reports.publisher import build_report

        output = EvaluationOutput(candidate_id="5Winner")
        report = build_report(
            output,
            window_id=1,
            block=100,
            workflow_id=C.DEFAULT_WORKFLOW_ID,
            base_revision="rev",
            source_snapshot_sha256="sha256:" + "0" * 64,
            evaluator_image_digest="test",
            miner_hotkey="5Winner",
            miner_uid=2,
            verdict="dethrone",
        )
        champion = ChampionRecord(candidate_id="5Winner", hotkey="5Winner", uid=2)

        store.set_champion(champion, report=report)

        assert store.get_champion().hotkey == "5Winner"
        assert store.list_reports(window_id=1)

    def test_a_second_commitment_does_not_reset_a_queue_entry(self, store):
        store.upsert_queue_entry(
            QueueEntry(
                hotkey="5Miner",
                uid=1,
                recipe_sha256="sha256:" + "a" * 64,
                recipe_uri="https://example.test/a",
                first_block=10,
                admitted_at_block=10,
            )
        )
        store.set_status("5Miner", "terminated", "lost")

        store.upsert_queue_entry(
            QueueEntry(
                hotkey="5Miner",
                uid=1,
                recipe_sha256="sha256:" + "b" * 64,
                recipe_uri="https://example.test/b",
                first_block=20,
                admitted_at_block=20,
            )
        )

        entry = store.get_queue_entry("5Miner")
        assert entry.status == "terminated"
        assert entry.recipe_sha256.endswith("a" * 64)

    def test_the_queue_is_ordered_by_commit_block(self, store):
        for hotkey, block in (("5C", 300), ("5A", 100), ("5B", 200)):
            store.upsert_queue_entry(
                QueueEntry(
                    hotkey=hotkey,
                    uid=1,
                    recipe_sha256=f"sha256:{hotkey[-1] * 64}",
                    recipe_uri="https://example.test/r",
                    first_block=block,
                    admitted_at_block=block,
                )
            )
        assert store.next_challenger().hotkey == "5A"
        assert [entry.hotkey for entry in store.list_queue()] == ["5A", "5B", "5C"]


class TestPublishedSurfaces:
    def test_the_contract_serialises_and_names_every_method(self, tiny_snapshot):
        workflow = get_workflow(C.DEFAULT_WORKFLOW_ID)
        contract = workflow.build_contract(tiny_snapshot)

        canonical_json_str(contract)  # must be serialisable without loss

        assert set(contract["recipe"]["merge_methods"]) == set(C.ALLOWED_MERGE_METHODS)
        assert contract["stages"]["order"] == list(workflow.stages)
        assert contract["hard_gates"]["artifact_size_bytes"] == C.MAX_ARTIFACT_BYTES

    def test_the_api_serves_engine_state(self, engine):
        from fastapi.testclient import TestClient

        from capability_subnet.backend.api import create_app

        loop, store, _, settings = engine
        store.set_champion(ChampionRecord(candidate_id="5Winner", hotkey="5Winner", uid=6))
        loop.publish_weights(block=250)

        client = TestClient(create_app(settings))

        assert client.get("/health").json()["status"] == "ok"
        # Looked up by uid rather than by position: entries are ordered by uid
        # for the chain's benefit, so indexing the first one asserts an ordering
        # the protocol never promised.
        entries = {e["uid"]: e for e in client.get("/weights").json()["entries"]}
        assert entries[6]["role"] == "champion"
        assert client.get("/champion").json()["hotkey"] == "5Winner"
        assert client.get("/queue").json()["count"] >= 0
        assert "workflow_id" in client.get("/contract").json()
        assert client.get("/reports/sha256:" + "0" * 64).status_code == 404

    def test_the_dashboard_renders_from_engine_state(self, engine):
        from capability_subnet.platform.dashboard import render

        loop, store, _, _ = engine
        store.set_champion(ChampionRecord(candidate_id="5Winner", hotkey="5Winner", uid=6))
        loop.publish_weights(block=250)

        page = render(store, workflow_id=C.DEFAULT_WORKFLOW_ID, generated_at="fixed")

        assert "<!doctype html>" in page
        assert "5Winner" in page
        assert "<script" not in page.lower()  # no scripts, nothing external
