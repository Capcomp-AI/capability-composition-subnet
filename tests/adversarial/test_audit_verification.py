"""Independent verification catches an inconsistent record.

These tests take the operator's side: they construct reports that *claim*
something the rest of the same report does not support, and assert the verifier
says so.

That is the whole value of the exercise. A centralised engine can publish
whatever it likes, so the useful question is not "is the engine honest" but
"would we be able to tell". Each test below is one way of telling.
"""

from __future__ import annotations

import pytest

from capability_subnet.audit.verify import (
    audit_run,
    recompute_qualified_score,
    verify_report,
    verify_weight_vector,
)
from capability_subnet.common import constants as C
from capability_subnet.common.schemas import (
    AxisVerdict,
    CandidateScores,
    ComparatorOutcome,
    EvaluationReport,
    GateVerdict,
    PairedComparison,
    WeightEntry,
    WeightVector,
)


def scores(**overrides) -> CandidateScores:
    values = {
        "end_to_end": 0.80,
        "stage_balance": 0.90,
        "ood": 0.70,
        "retention": 1.00,
        "token_efficiency": 1.00,
        "artifact_efficiency": 0.50,
    }
    values.update(overrides)
    payload = CandidateScores(**values)
    payload.qualified_score = recompute_qualified_score(
        EvaluationReport(
            run_id=0,
            evaluated_at_block=0,
            miner_hotkey="",
            candidate_id="",
            base_revision="",
            source_snapshot_sha256="",
            evaluator_image_digest="",
            scores=payload,
        )
    )
    return payload


def comparator(**overrides) -> ComparatorOutcome:
    values = {
        "per_axis_verdicts": [
            AxisVerdict(
                axis="fault_extraction",
                challenger_mean=0.9,
                champion_mean=0.7,
                paired_samples=60,
                verdict="dominant",
                margin=0.02,
                tolerance=0.01,
            ),
            AxisVerdict(
                axis="final_json",
                challenger_mean=0.9,
                champion_mean=0.9,
                paired_samples=60,
                verdict="not_worse",
                margin=0.02,
                tolerance=0.01,
            ),
        ],
        "dominant_count": 1,
        "min_dominant_required": 1,
        "any_worse_axis": False,
        "end_to_end_margin_required": 0.03,
        "end_to_end_margin_observed": 0.08,
        "paired": PairedComparison(
            reference_id="incumbent",
            paired_instances=60,
            challenger_mean=0.80,
            reference_mean=0.72,
            difference=0.08,
            bootstrap_lcb=0.031,
            bootstrap_resamples=10_000,
            confidence=0.95,
            passed=True,
        ),
        "dethrones": True,
        "reason": "dethrones the incumbent",
    }
    values.update(overrides)
    return ComparatorOutcome(**values)


def report(**overrides) -> EvaluationReport:
    values = {
        "run_id": 12,
        "evaluated_at_block": 90_000,
        "miner_hotkey": "5Challenger",
        "miner_uid": 7,
        "candidate_id": "5Challenger",
        "base_revision": "rev-pinned",
        "source_snapshot_sha256": "sha256:" + "a" * 64,
        "evaluator_image_digest": "sha256:" + "b" * 64,
        "hard_gates": [
            GateVerdict(name="artifact_size", passed=True),
            GateVerdict(name="peak_vram", passed=True),
            GateVerdict(name="baseline", passed=True),
        ],
        "scores": scores(),
        "contribution": {"contribution": 0.42},
        "baseline_scores": {
            "reference:base_model": 0.20,
            "reference:best_single_adapter": 0.55,
            "reference:equal_ties_svd_merge": 0.72,
            "incumbent": 0.72,
        },
        "strongest_reference_id": "incumbent",
        "strongest_reference_score": 0.72,
        "comparator": comparator(),
        "verdict": "dethrone",
        "verdict_reason": "dethrones the incumbent",
    }
    values.update(overrides)
    return EvaluationReport(**values)


class TestAConsistentRecordPasses:
    def test_a_well_formed_report_raises_no_errors(self):
        result = verify_report(report(), trusted_signers=None)
        # Unsigned is flagged; nothing else should be.
        assert [f.code for f in result.errors] == ["unsigned"]

    def test_the_qualified_score_recomputes(self):
        payload = report()
        assert recompute_qualified_score(payload) == pytest.approx(
            payload.scores.qualified_score, abs=1e-9
        )


class TestFabricatedScores:
    def test_a_total_that_does_not_match_its_components_is_caught(self):
        # The single most direct fabrication: inflate the headline number and
        # leave the components alone.
        payload = report()
        payload.scores.qualified_score = 0.99

        result = verify_report(payload)
        assert any(f.code == "score_mismatch" for f in result.errors)

    def test_serialisation_rounding_is_not_mistaken_for_fabrication(self):
        payload = report()
        payload.scores.qualified_score = round(payload.scores.qualified_score, 4)

        result = verify_report(payload)
        assert not any(f.code == "score_mismatch" for f in result.errors)


class TestALoweredBar:
    def test_understating_the_strongest_reference_is_caught(self):
        # The subtle one: publish the real baseline scores, then claim a weaker
        # one was the bar so a mediocre challenger clears it.
        payload = report(
            strongest_reference_id="reference:best_single_adapter",
            strongest_reference_score=0.55,
        )
        result = verify_report(payload)

        finding = next(f for f in result.errors if f.code == "understated_bar")
        assert "equal_ties_svd_merge" in finding.detail or "incumbent" in finding.detail

    def test_a_bar_disagreeing_with_its_own_baseline_entry_is_caught(self):
        payload = report(strongest_reference_score=0.40)
        result = verify_report(payload)
        assert any(f.code == "reference_score_mismatch" for f in result.errors)


class TestUnsupportedVerdicts:
    def test_crowning_despite_a_failed_gate_is_caught(self):
        payload = report()
        payload.hard_gates = [
            GateVerdict(name="artifact_size", passed=True),
            GateVerdict(name="safety", passed=False, detail="1 critical unsafe action"),
        ]
        result = verify_report(payload)
        assert any(f.code == "dethrone_despite_failed_gate" for f in result.errors)

    def test_crowning_with_no_gates_at_all_is_caught(self):
        payload = report(hard_gates=[])
        result = verify_report(payload)
        assert any(f.code == "ungated_dethrone" for f in result.errors)

    def test_crowning_while_worse_on_an_axis_is_caught(self):
        # The rule that stops a package trading a capability away. A report can
        # state the regression and crown anyway; the verifier will not accept it.
        payload = report(
            comparator=comparator(
                any_worse_axis=True,
                per_axis_verdicts=[
                    AxisVerdict(
                        axis="safety_validation",
                        challenger_mean=0.3,
                        champion_mean=0.9,
                        paired_samples=60,
                        verdict="worse",
                        margin=0.02,
                        tolerance=0.01,
                    )
                ],
            )
        )
        result = verify_report(payload)
        assert any(f.code == "dethrone_with_regression" for f in result.errors)

    def test_crowning_without_significance_is_caught(self):
        payload = report(
            comparator=comparator(
                paired=PairedComparison(
                    reference_id="incumbent",
                    paired_instances=60,
                    challenger_mean=0.80,
                    reference_mean=0.79,
                    difference=0.01,
                    bootstrap_lcb=-0.004,
                    bootstrap_resamples=10_000,
                    confidence=0.95,
                    passed=False,
                )
            )
        )
        result = verify_report(payload)
        assert any(f.code == "dethrone_without_significance" for f in result.errors)

    def test_crowning_below_the_margin_is_caught(self):
        payload = report(comparator=comparator(end_to_end_margin_observed=0.004))
        result = verify_report(payload)
        assert any(f.code == "margin_not_met" for f in result.errors)

    def test_crowning_with_no_comparison_recorded_is_caught(self):
        payload = report(comparator=None)
        result = verify_report(payload)
        assert any(f.code == "dethrone_without_comparison" for f in result.errors)

    def test_terminating_a_candidate_that_passed_everything_is_caught(self):
        payload = report(verdict="terminated", comparator=None)
        result = verify_report(payload)
        assert any(f.code == "unexplained_termination" for f in result.errors)


class TestScopeAndAttribution:
    def test_a_report_against_a_different_pool_is_caught(self):
        result = verify_report(report(), expected_snapshot="sha256:" + "c" * 64)
        assert any(f.code == "snapshot_mismatch" for f in result.errors)

    def test_a_report_against_a_different_base_is_caught(self):
        result = verify_report(report(), expected_base_revision="some-other-rev")
        assert any(f.code == "base_mismatch" for f in result.errors)

    def test_an_untrusted_signer_is_caught(self):
        payload = report()
        payload.signature = "00" * 64
        payload.signer_hotkey = "5Impostor"

        result = verify_report(payload, trusted_signers={"5Operator"})
        assert any(f.code == "untrusted_signer" for f in result.errors)

    def test_an_unpinned_evaluator_is_a_warning_not_an_error(self):
        # It does not contradict anything; it just means the report cannot
        # identify the software that produced it.
        result = verify_report(report(evaluator_image_digest="unpinned"))
        assert any(f.code == "unpinned_evaluator" for f in result.warnings)
        assert not any(f.code == "unpinned_evaluator" for f in result.errors)


class TestWeightVectors:
    def _vector(self, **overrides) -> WeightVector:
        values = {
            "run_id": 12,
            "computed_at_block": 90_000,
            "entries": [WeightEntry(uid=7, hotkey="5Challenger", weight=1.0)],
            "champion_hotkey": "5Challenger",
        }
        values.update(overrides)
        return WeightVector(**values)

    def test_a_vector_matching_its_reports_raises_only_the_signature_finding(self):
        result = verify_weight_vector(self._vector(), [report()])
        assert [f.code for f in result.errors] == ["unsigned"]

    def test_paying_someone_no_report_crowns_is_caught(self):
        vector = self._vector(
            entries=[WeightEntry(uid=99, hotkey="5Stranger", weight=1.0)],
            champion_hotkey="5Stranger",
        )
        result = verify_weight_vector(vector, [report()])
        assert any(f.code == "unsupported_champion" for f in result.errors)

    def test_paying_a_permanent_reference_is_caught(self):
        # A reference on the throne means no miner has beaten an off-the-shelf
        # merge. The share must burn, not go to the operator.
        vector = self._vector(
            entries=[WeightEntry(uid=3, hotkey="reference:equal_ties_svd_merge", weight=1.0)],
            champion_hotkey=None,
        )
        result = verify_weight_vector(vector, [report()])
        assert any(f.code == "reference_paid" for f in result.errors)

    def test_a_burn_entry_is_not_treated_as_a_recipient(self):
        vector = self._vector(
            entries=[WeightEntry(uid=0, hotkey="", weight=1.0, role="burn")],
            champion_hotkey=None,
        )
        result = verify_weight_vector(vector, [])
        assert not any(f.code in ("reference_paid", "multiple_recipients") for f in result.errors)


class TestWholeRun:
    def test_a_consistent_run_passes(self):
        result = audit_run([report()], None, trusted_signers=None)
        assert [f.code for f in result.errors] == ["unsigned"]
        assert result.reports_checked == 1

    def test_two_dethrones_in_one_run_is_flagged(self):
        second = report(miner_hotkey="5Second", candidate_id="5Second")
        result = audit_run([report(), second], None)
        assert any(f.code == "multiple_dethrones" for f in result.warnings)

    def test_the_summary_reports_what_was_checked(self):
        result = audit_run([report(), report()], None)
        assert "2 report(s) checked" in result.summary()


class TestOnlyAGradedQualifierMayBePaid:
    """Every payment claims the recipient cleared every hard gate and earned a
    grade. Both are checkable from the published reports and neither is checked
    by the chain, so without this a vector could pay anyone and still verify."""

    @staticmethod
    def _vector(**kwargs):
        from capability_subnet.common.schemas import WeightVector

        defaults = dict(
            workflow_id=C.DEFAULT_WORKFLOW_ID,
            run_id=12,
            computed_at_block=90_000,
            champion_hotkey="5Challenger",
        )
        defaults.update(kwargs)
        return WeightVector(**defaults)

    def test_paying_someone_with_no_passing_report_is_caught(self):
        vector = self._vector(
            entries=[WeightEntry(uid=8, hotkey="5Ghost", weight=1.0, role="runner_up")]
        )
        result = verify_weight_vector(vector, [report()])

        assert any(f.code == "unqualified_recipient" for f in result.errors)

    def test_paying_someone_whose_report_records_no_grade_is_caught(self):
        vector = self._vector(
            entries=[WeightEntry(uid=7, hotkey="5Challenger", weight=1.0, role="runner_up")]
        )
        result = verify_weight_vector(vector, [report(contribution={})])

        assert any(f.code == "ungraded_recipient" for f in result.errors)

    def test_a_graded_qualifier_with_a_supporting_report_passes(self):
        vector = self._vector(
            entries=[WeightEntry(uid=7, hotkey="5Challenger", weight=1.0, role="champion")]
        )
        result = verify_weight_vector(vector, [report()])

        assert [f.code for f in result.errors] == ["unsigned"]

    def test_paying_without_naming_a_champion_is_the_ordinary_case(self):
        """A run that pays a full ladder and crowns nobody is not an error.

        It used to be: payment required dethroning, so a paying vector with no
        champion described a field that had not cleared the bar. Payment now
        follows the hard gates and the throne moves on its own margin, so the
        two come apart routinely -- runs 416, 417 and 418 each paid ten miners
        and moved no throne. The audit refusing that would refuse every vector
        this subnet now publishes.
        """
        vector = self._vector(
            champion_hotkey=None,
            entries=[WeightEntry(uid=7, hotkey="5Challenger", weight=1.0, role="runner_up")],
        )
        result = verify_weight_vector(vector, [report()])

        assert not any(f.code == "paid_without_a_throne" for f in result.errors)

    def test_a_champion_who_is_not_paid_is_caught(self):
        """The reverse, which is still wrong.

        The throne records who leads the run that paid. Naming a champion the
        vector does not pay describes a run it did not compute.
        """
        vector = self._vector(
            champion_hotkey="5Absent",
            entries=[WeightEntry(uid=7, hotkey="5Challenger", weight=1.0, role="runner_up")],
        )
        result = verify_weight_vector(vector, [report()])

        assert any(f.code == "champion_not_paid" for f in result.errors)

    def test_a_champion_paid_less_than_a_runner_up_is_caught(self):
        """Rank one takes the leading share; a champion below it is a ladder
        that was built in the wrong order."""
        vector = self._vector(
            champion_hotkey="5Challenger",
            entries=[
                WeightEntry(uid=7, hotkey="5Challenger", weight=0.2, role="champion"),
                WeightEntry(uid=9, hotkey="5Other", weight=0.8, role="runner_up"),
            ],
        )
        result = verify_weight_vector(vector, [report()])

        assert any(f.code == "champion_not_the_leading_share" for f in result.errors)

    def test_a_wholly_burned_vector_needs_no_champion(self):
        vector = self._vector(
            champion_hotkey=None,
            entries=[WeightEntry(uid=C.BURN_UID, hotkey="", weight=1.0, role="burn")],
        )
        result = verify_weight_vector(vector, [report()])

        assert not any(f.code == "paid_without_a_throne" for f in result.errors)
