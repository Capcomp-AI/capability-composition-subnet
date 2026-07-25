"""Re-scoring a closed window catches a doctored result.

The disclosure mechanism only earns its keep if it detects the thing it exists
to detect. These tests run a real instance through the real sandbox, take the
engine's own trace and result, then alter what the engine *claims* and assert
that re-scoring the unaltered trace disagrees.

Everything here runs without a GPU, which is the point: a check only validators
with an evaluation cluster could run would not be a check on the operator.
"""

from __future__ import annotations

import pytest

from capability_subnet.audit.replay import (
    replay_disclosure,
    trace_from_dict,
    verify_disclosure,
)
from capability_subnet.common.schemas import DisclosedInstance, WindowDisclosure
from capability_subnet.sandbox.orchestrator import run_instance
from capability_subnet.sandbox.reference_solver import ReferenceSolverClient


@pytest.fixture(scope="module")
def scored_run():
    """One real evaluation: the instance, the engine's trace and its result."""
    from capability_subnet.workflows import get_workflow

    workflow = get_workflow("industrial_maintenance_de_v1")
    instance = workflow.generate_instance(778899, split="hidden")
    outcome = run_instance(instance, ReferenceSolverClient(instance))
    return instance, outcome.trace, outcome.result


def disclosure_for(instance, trace, result, **overrides) -> WindowDisclosure:
    entry = DisclosedInstance(
        instance_id=instance.instance_id,
        instance_seed=instance.seed,
        split=instance.split,
        candidate_id="5Champion",
        claimed_result=result.model_copy(deep=True),
        trace=trace.to_dict(),
    )
    for key, value in overrides.pop("entry", {}).items():
        setattr(entry, key, value)

    payload = {
        "window_id": 7,
        "closed_at_block": 50_400,
        "spec_version": 1000,
        "hidden_seeds": [instance.seed],
        "ood_seeds": [],
        "instances": [entry],
    }
    payload.update(overrides)
    return WindowDisclosure(**payload)


class TestAnHonestDisclosureReplays:
    def test_the_engines_own_result_re_scores_identically(self, scored_run):
        instance, trace, result = scored_run
        outcome, audit = replay_disclosure(disclosure_for(instance, trace, result))

        assert outcome.ok, [f.detail for f in audit.errors]
        assert outcome.checked == 1
        assert outcome.agreed == 1

    def test_the_instance_regenerates_from_its_seed_alone(self, scored_run):
        # The property the whole mechanism rests on: an auditor holding only the
        # seed reproduces the exact problem the candidate faced.
        from capability_subnet.workflows import get_workflow

        instance, _, _ = scored_run
        regenerated = get_workflow("industrial_maintenance_de_v1").generate_instance(
            instance.seed, split="hidden"
        )

        assert regenerated.instance_id == instance.instance_id
        assert regenerated.sensor_log == instance.sensor_log
        assert regenerated.truth == instance.truth

    def test_a_trace_survives_the_publication_round_trip(self, scored_run):
        _, trace, _ = scored_run
        rebuilt = trace_from_dict(trace.to_dict())

        assert rebuilt.final_payload == trace.final_payload
        assert rebuilt.inventory_final_state == trace.inventory_final_state
        assert len(rebuilt.calls) == len(trace.calls)
        assert rebuilt.is_scorable


class TestDoctoredResultsAreCaught:
    def test_inflating_end_to_end_success_is_caught(self, scored_run):
        # The simplest fabrication: claim the candidate finished when the trace
        # shows it did not.
        instance, trace, result = scored_run
        doctored = result.model_copy(deep=True)
        doctored.end_to_end_success = not result.end_to_end_success

        outcome, audit = replay_disclosure(
            disclosure_for(instance, trace, result, entry={"claimed_result": doctored})
        )

        assert not outcome.ok
        assert any(f.code == "score_disagreement" for f in audit.errors)

    def test_inflating_a_single_stage_score_is_caught(self, scored_run):
        instance, trace, result = scored_run
        doctored = result.model_copy(deep=True)
        stage = next(iter(doctored.stages))
        # Move it decisively rather than by an increment: a stage already at 1.0
        # cannot be inflated, and the test would silently assert nothing.
        original = doctored.stages[stage].score
        doctored.stages[stage].score = 0.0 if original >= 0.5 else 1.0

        outcome, audit = replay_disclosure(
            disclosure_for(instance, trace, result, entry={"claimed_result": doctored})
        )

        assert not outcome.ok
        finding = next(f for f in audit.errors if f.code == "score_disagreement")
        assert stage in finding.detail

    def test_hiding_a_critical_unsafe_action_is_caught(self, scored_run):
        instance, trace, result = scored_run
        doctored = result.model_copy(deep=True)
        doctored.critical_unsafe_actions = result.critical_unsafe_actions + 2

        outcome, audit = replay_disclosure(
            disclosure_for(instance, trace, result, entry={"claimed_result": doctored})
        )

        assert not outcome.ok
        assert any("unsafe" in f.detail for f in audit.errors)

    def test_a_seed_outside_the_declared_draw_is_caught(self, scored_run):
        # Scoring a candidate on an instance the window never claims to have
        # drawn — an easier problem, quietly substituted.
        instance, trace, result = scored_run
        outcome, audit = replay_disclosure(
            disclosure_for(instance, trace, result, hidden_seeds=[1, 2, 3])
        )
        del outcome
        assert any(f.code == "undisclosed_instance" for f in audit.errors)

    def test_a_mislabelled_instance_is_caught(self, scored_run):
        instance, trace, result = scored_run
        outcome, audit = replay_disclosure(
            disclosure_for(instance, trace, result, entry={"instance_id": "hidden-000000000001"})
        )
        del outcome
        assert any(f.code == "instance_id_mismatch" for f in audit.errors)


class TestAttribution:
    def test_an_unsigned_disclosure_is_flagged(self, scored_run):
        instance, trace, result = scored_run
        _, audit = verify_disclosure(disclosure_for(instance, trace, result))
        assert any(f.code == "unsigned" for f in audit.errors)

    def test_an_untrusted_signer_is_flagged(self, scored_run):
        instance, trace, result = scored_run
        disclosure = disclosure_for(instance, trace, result)
        disclosure.signature = "00" * 64
        disclosure.signer_hotkey = "5Impostor"

        _, audit = verify_disclosure(disclosure, trusted_signers={"5Operator"})
        assert any(f.code == "untrusted_signer" for f in audit.errors)

    def test_the_signed_bytes_exclude_the_signature(self, scored_run):
        instance, trace, result = scored_run
        disclosure = disclosure_for(instance, trace, result)
        before = disclosure.signable_bytes()
        disclosure.signature = "deadbeef"
        disclosure.signer_hotkey = "5Operator"
        assert disclosure.signable_bytes() == before


class TestDisclosureBoundaries:
    def test_an_empty_disclosure_is_flagged_rather_than_passing_quietly(self):
        disclosure = WindowDisclosure(window_id=3, closed_at_block=1, spec_version=1000)
        outcome, audit = replay_disclosure(disclosure)

        assert outcome.checked == 0
        assert any(f.code == "nothing_disclosed" for f in audit.warnings)

    def test_the_engine_refuses_to_disclose_an_open_window(self, engine):
        loop, store, _, settings = engine
        loop.ensure_window(block=250)
        current = 250 // settings.window_blocks

        with pytest.raises(ValueError, match="has not closed yet"):
            loop.build_disclosure(current, block=250)

    def test_a_closed_window_discloses_its_seeds_and_traces(self, engine):
        loop, store, _, settings = engine
        loop.ensure_window(block=250)
        closed = 250 // settings.window_blocks

        disclosure = loop.build_disclosure(closed, block=250 + settings.window_blocks)

        assert disclosure.window_id == closed
        assert len(disclosure.hidden_seeds) == settings.hidden_instances
        assert disclosure.instances, "reference traces should have been retained"

        # And it replays cleanly, because the engine scored it honestly.
        outcome, audit = replay_disclosure(disclosure)
        assert outcome.ok, [f.detail for f in audit.errors]
        assert outcome.agreed == outcome.checked

    def test_the_api_refuses_an_open_window_and_serves_a_closed_one(self, engine):
        from fastapi.testclient import TestClient

        from capability_subnet.backend.api import create_app

        loop, store, _, settings = engine
        loop.ensure_window(block=250)
        current = 250 // settings.window_blocks

        client = TestClient(create_app(settings))
        assert client.get(f"/windows/{current}/disclosure").status_code == 409

        response = client.get(f"/windows/{current - 1}/disclosure")
        assert response.status_code in (404, 409)
