"""Re-scoring a closed run catches a doctored result.

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
from capability_subnet.common.schemas import DisclosedInstance, RunDisclosure
from capability_subnet.sandbox.orchestrator import run_instance
from capability_subnet.sandbox.reference_solver import ReferenceSolverClient
from capability_subnet.testing import MAINTENANCE_WORKFLOW_ID


@pytest.fixture(scope="module")
def scored_run():
    """One real evaluation: the instance, the engine's trace and its result."""
    from capability_subnet.workflows import get_workflow

    workflow = get_workflow("industrial_maintenance_de_v1")
    instance = workflow.generate_instance(778899, split="hidden")
    outcome = run_instance(instance, ReferenceSolverClient(instance))
    return instance, outcome.trace, outcome.result


def disclosure_for(instance, trace, result, **overrides) -> RunDisclosure:
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
        # Stated explicitly: a disclosure names its own workflow, and this helper
        # builds a maintenance run.
        "workflow_id": MAINTENANCE_WORKFLOW_ID,
        "run_id": 7,
        "closed_at_block": 50_400,
        "hidden_seeds": [instance.seed],
        "ood_seeds": [],
        "instances": [entry],
    }
    payload.update(overrides)
    return RunDisclosure(**payload)


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
        # Scoring a candidate on an instance the run never claims to have
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


class TestTheValidatorRefusesToPayForAFabricatedRun:
    """The spot check is what makes a validator a verifier rather than a relay.

    Replay was already correct and already exposed over the API; nothing
    consumed it. A published record nobody reads before paying is documentation,
    not a control — so these tests cover the path where refusing actually costs
    the operator something.
    """

    @staticmethod
    def _client(disclosure):
        """A backend client whose disclosure endpoint returns ``disclosure``.

        A stand-in rather than a live engine: what is under test is the
        validator's decision, and threading a real HTTP server through it would
        test the server.
        """

        class _Client:
            def fetch_disclosure(self, run_id):
                if disclosure is None:
                    from capability_subnet.validator.client import BackendUnavailable

                    raise BackendUnavailable(f"run {run_id} is not disclosed")
                return disclosure

        return _Client()

    def test_an_honest_run_passes(self, scored_run):
        from capability_subnet.validator.client import spot_check_run

        instance, trace, result = scored_run
        ok, detail = spot_check_run(
            self._client(disclosure_for(instance, trace, result)), run_id=7
        )
        assert ok, detail
        assert "re-scored to the same result" in detail

    def test_a_run_whose_scores_do_not_follow_from_its_traces_is_refused(self, scored_run):
        """The failure the whole published record exists to make detectable."""
        from capability_subnet.validator.client import spot_check_run

        instance, trace, result = scored_run
        doctored = result.model_copy(deep=True)
        doctored.end_to_end_success = not result.end_to_end_success

        ok, detail = spot_check_run(
            self._client(
                disclosure_for(instance, trace, result, entry={"claimed_result": doctored})
            ),
            run_id=7,
        )
        assert not ok
        assert "does not re-score" in detail

    def test_an_undisclosed_run_is_not_treated_as_dishonesty(self, scored_run):
        """Absence of a disclosure is absence of evidence.

        A run the engine has not published yet, or an engine briefly
        unreachable, must not cost the champion its emission — that would turn
        an ordinary outage into a punishment and give validators an incentive to
        race the disclosure.
        """
        from capability_subnet.validator.client import spot_check_run

        ok, detail = spot_check_run(self._client(None), run_id=7)
        assert ok
        assert "not available" in detail

    def test_the_spot_check_needs_no_tensor_stack(self):
        """It runs on the VPS the validator is documented to be."""
        import inspect

        from capability_subnet.validator import client

        source = inspect.getsource(client)
        assert "import torch" not in source
