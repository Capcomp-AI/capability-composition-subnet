"""Independent verification of what the engine published.

The engine is centralised, so its output is only as trustworthy as the checks
anyone can run against it. This module is those checks, written so that a
validator, a miner or an outside observer can run them with nothing but the
published reports — no hidden instances, no GPU, no cooperation from the
operator.

What can be verified without the hidden data turns out to be most of what
matters:

* the report is signed by an operator on the verifier's own allow-list;
* the qualified score follows from its own published components;
* the claimed strongest reference really is the strongest one published;
* the verdict follows from the gates and the comparator outcome;
* the weight vector follows from the reports.

None of that proves the hidden instances were fair. It does mean a fabricated
number has to be fabricated *consistently*, across a signed record that is
published the moment it is produced and cannot later be edited without breaking
its own digest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import EvaluationReport, WeightVector
from capability_subnet.common.signing import verify_payload
from capability_subnet.scoring.references import is_reference

Severity = Literal["error", "warning", "info"]

#: Tolerance when recomputing a published score from its components. Reports
#: carry float values rounded through JSON, so an exact comparison would flag
#: serialisation rather than dishonesty.
SCORE_TOLERANCE = 5e-4


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing the verifier noticed."""

    code: str
    severity: Severity
    detail: str
    subject: str = ""

    def __str__(self) -> str:
        where = f" [{self.subject}]" if self.subject else ""
        return f"{self.severity.upper():7} {self.code}{where}: {self.detail}"


@dataclass(slots=True)
class AuditResult:
    """Everything the verifier found."""

    findings: list[Finding] = field(default_factory=list)
    reports_checked: int = 0

    def add(self, code: str, severity: Severity, detail: str, subject: str = "") -> None:
        self.findings.append(Finding(code, severity, detail, subject))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        """Whether anything published contradicts anything else published."""
        return not self.errors

    def summary(self) -> str:
        if not self.findings:
            return f"{self.reports_checked} report(s) checked, nothing to report"
        return (
            f"{self.reports_checked} report(s) checked: "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def recompute_qualified_score(report: EvaluationReport) -> float:
    """Recompute the qualified score from the report's own components.

    The formula is published in the contract, so a report that states both its
    components and its total is stating the same thing twice. If the two
    disagree, one of them was not produced by the published formula.
    """
    scores = report.scores
    return (
        C.WEIGHT_END_TO_END * scores.end_to_end
        + C.WEIGHT_STAGE_BALANCE * scores.stage_balance
        + C.WEIGHT_OOD * scores.ood
        + C.WEIGHT_RETENTION * scores.retention
        + C.WEIGHT_LATENCY * scores.latency
        + C.WEIGHT_ARTIFACT_EFFICIENCY * scores.artifact_efficiency
    )


def verify_report(
    report: EvaluationReport,
    *,
    trusted_signers: set[str] | None = None,
    expected_snapshot: str | None = None,
    expected_base_revision: str | None = None,
    result: AuditResult | None = None,
) -> AuditResult:
    """Check one report against itself and against the pool it claims to use."""
    result = result or AuditResult()
    result.reports_checked += 1
    subject = report.candidate_id

    # -- attribution --------------------------------------------------------
    if not report.signature or not report.signer_hotkey:
        result.add(
            "unsigned",
            "error",
            "the report carries no signature, so it cannot be attributed to any operator",
            subject,
        )
    elif trusted_signers is not None and report.signer_hotkey not in trusted_signers:
        result.add(
            "untrusted_signer",
            "error",
            f"signed by {report.signer_hotkey}, which is not on the allow-list",
            subject,
        )
    elif not verify_payload(report, report.signature, report.signer_hotkey):
        result.add(
            "bad_signature",
            "error",
            f"the signature attributed to {report.signer_hotkey} does not verify",
            subject,
        )

    # -- scope --------------------------------------------------------------
    if expected_snapshot and report.source_snapshot_sha256 != expected_snapshot:
        result.add(
            "snapshot_mismatch",
            "error",
            f"evaluated against pool {report.source_snapshot_sha256[:19]}…, "
            f"expected {expected_snapshot[:19]}…",
            subject,
        )
    if expected_base_revision and report.base_revision != expected_base_revision:
        result.add(
            "base_mismatch",
            "error",
            f"evaluated against base {report.base_revision!r}, expected {expected_base_revision!r}",
            subject,
        )
    if report.evaluator_image_digest in ("", "unpinned"):
        result.add(
            "unpinned_evaluator",
            "warning",
            "the evaluator image is unpinned, so this report does not identify "
            "the software that produced it",
            subject,
        )

    # -- arithmetic ---------------------------------------------------------
    recomputed = recompute_qualified_score(report)
    published = report.scores.qualified_score
    if abs(recomputed - published) > SCORE_TOLERANCE:
        result.add(
            "score_mismatch",
            "error",
            f"qualified score is published as {published:.6f} but its own components "
            f"give {recomputed:.6f}",
            subject,
        )

    # -- the bar ------------------------------------------------------------
    if report.baseline_scores:
        strongest = max(report.baseline_scores.values())
        if report.strongest_reference_score < strongest - SCORE_TOLERANCE:
            winner = max(report.baseline_scores, key=lambda k: report.baseline_scores[k])
            result.add(
                "understated_bar",
                "error",
                f"the strongest reference is stated as {report.strongest_reference_id} at "
                f"{report.strongest_reference_score:.4f}, but {winner} is published at "
                f"{strongest:.4f}. The bar this candidate cleared was lower than the "
                f"published references support.",
                subject,
            )
        claimed = report.baseline_scores.get(report.strongest_reference_id)
        if (
            claimed is not None
            and abs(claimed - report.strongest_reference_score) > SCORE_TOLERANCE
        ):
            result.add(
                "reference_score_mismatch",
                "error",
                f"{report.strongest_reference_id} is listed at {claimed:.4f} among the "
                f"baselines but at {report.strongest_reference_score:.4f} as the bar",
                subject,
            )

    # -- verdict ------------------------------------------------------------
    _verify_verdict(report, result, subject)

    return result


def _verify_verdict(report: EvaluationReport, result: AuditResult, subject: str) -> None:
    """Check that the decision follows from the evidence in the same report."""
    verdict = report.verdict
    gates_passed = report.gates_passed
    failed = report.failed_gates()

    if verdict == "dethrone":
        if not report.hard_gates:
            result.add(
                "ungated_dethrone",
                "error",
                "a champion was crowned on a report with no gate verdicts at all",
                subject,
            )
        elif not gates_passed:
            result.add(
                "dethrone_despite_failed_gate",
                "error",
                f"a champion was crowned despite failing: {', '.join(failed)}",
                subject,
            )

        comparator = report.comparator
        if comparator is None:
            result.add(
                "dethrone_without_comparison",
                "error",
                "a champion was crowned with no comparator outcome recorded",
                subject,
            )
        else:
            if not comparator.dethrones:
                result.add(
                    "dethrone_contradicts_comparator",
                    "error",
                    f"the comparator did not support a dethrone: {comparator.reason}",
                    subject,
                )
            if comparator.any_worse_axis:
                worse = [v.axis for v in comparator.per_axis_verdicts if v.verdict == "worse"]
                result.add(
                    "dethrone_with_regression",
                    "error",
                    f"a champion was crowned while worse on {worse}. The rule requires "
                    "not-worse on every axis it does not dominate.",
                    subject,
                )
            if comparator.dominant_count < comparator.min_dominant_required:
                result.add(
                    "insufficient_dominance",
                    "error",
                    f"dominant on {comparator.dominant_count} axes, "
                    f"{comparator.min_dominant_required} required",
                    subject,
                )
            paired = comparator.paired
            if paired is not None and not paired.passed:
                result.add(
                    "dethrone_without_significance",
                    "error",
                    f"a champion was crowned with a paired lower bound of "
                    f"{paired.bootstrap_lcb:+.5f}, which is not above zero",
                    subject,
                )
            if (
                comparator.end_to_end_margin_observed
                < comparator.end_to_end_margin_required - SCORE_TOLERANCE
            ):
                result.add(
                    "margin_not_met",
                    "error",
                    f"end-to-end margin {comparator.end_to_end_margin_observed:+.4f} is below "
                    f"the required {comparator.end_to_end_margin_required:+.4f}",
                    subject,
                )

    elif verdict == "terminated":
        decisive = report.comparator is not None and not report.comparator.dethrones
        if gates_passed and not decisive:
            result.add(
                "unexplained_termination",
                "error",
                "a candidate was terminated although it passed every gate and no "
                "comparator outcome explains it",
                subject,
            )


# ---------------------------------------------------------------------------
# Weight vectors
# ---------------------------------------------------------------------------


def verify_weight_vector(
    vector: WeightVector,
    reports: list[EvaluationReport],
    *,
    trusted_signers: set[str] | None = None,
    result: AuditResult | None = None,
) -> AuditResult:
    """Check that a weight vector follows from the published reports."""
    result = result or AuditResult()
    subject = f"window {vector.window_id}"

    if not vector.signature or not vector.signer_hotkey:
        result.add("unsigned", "error", "the weight vector carries no signature", subject)
    elif trusted_signers is not None and vector.signer_hotkey not in trusted_signers:
        result.add(
            "untrusted_signer",
            "error",
            f"signed by {vector.signer_hotkey}, which is not on the allow-list",
            subject,
        )
    elif not verify_payload(vector, vector.signature, vector.signer_hotkey):
        result.add("bad_signature", "error", "the signature does not verify", subject)

    if not vector.entries:
        result.add("empty", "error", "the vector contains no entries", subject)
        return result

    total = sum(entry.weight for entry in vector.entries)
    if abs(total - 1.0) > 1e-6:
        result.add("not_normalised", "error", f"weights sum to {total:.8f}", subject)

    uids = [entry.uid for entry in vector.entries]
    if len(set(uids)) != len(uids):
        result.add("duplicate_uid", "error", f"repeated UIDs in {uids}", subject)

    paid = [
        entry
        for entry in vector.entries
        if entry.role != "burn" and entry.uid != C.BURN_UID and entry.weight > 0
    ]

    # A reference holding the throne means no miner has beaten an off-the-shelf
    # merge yet. Paying for that would be paying the operator.
    for entry in paid:
        if is_reference(entry.hotkey or ""):
            result.add(
                "reference_paid",
                "error",
                f"UID {entry.uid} is a permanent reference and must not earn emission",
                subject,
            )

    if vector.mode == C.MODE_WINNER_TAKE_ALL and len(paid) > 1:
        result.add(
            "multiple_recipients",
            "error",
            f"winner-take-all mode paid {len(paid)} recipients",
            subject,
        )

    # The champion must be someone a report actually crowned.
    if vector.champion_hotkey:
        crowned = {
            report.miner_hotkey
            for report in reports
            if report.verdict == "dethrone" and report.miner_hotkey
        }
        held = {
            report.miner_hotkey
            for report in reports
            if report.verdict in ("held", "dethrone") and report.miner_hotkey
        }
        if crowned and vector.champion_hotkey not in crowned | held:
            result.add(
                "unsupported_champion",
                "error",
                f"{vector.champion_hotkey[:16]}… is paid as champion but no published "
                "report crowns or sustains it",
                subject,
            )

    if vector.mode == C.MODE_GRADED_CONTRIBUTION:
        # A graded payment is a claim that the recipient cleared every hard gate
        # and earned a grade. Both halves are checkable from the published
        # reports, and neither is checked by the chain — so a mode that pays more
        # people needs a rule that says who may be paid, or "graded" becomes a
        # licence to pay anyone.
        qualified = {
            report.miner_hotkey: report
            for report in reports
            if report.gates_passed and report.miner_hotkey
        }
        for entry in paid:
            if entry.role in ("burn", "queued"):
                # The queue tail is deregistration protection, not payment for a
                # result, so it is not required to have a passing report.
                continue
            if not entry.hotkey or not qualified:
                continue
            if entry.hotkey not in qualified:
                result.add(
                    "unqualified_recipient",
                    "error",
                    f"{entry.hotkey[:16]}… is paid a graded share but no published report "
                    "shows it clearing every hard gate",
                    subject,
                )
                continue
            report = qualified[entry.hotkey]
            if entry.role == "contributor" and report.contribution.get("contribution", 0.0) <= 0.0:
                result.add(
                    "ungraded_contributor",
                    "error",
                    f"{entry.hotkey[:16]}… is paid as a contributor but its report records "
                    "no contribution grade",
                    subject,
                )

    if vector.mode == C.MODE_GRADED_TOP3:
        qualified = {
            report.miner_hotkey for report in reports if report.gates_passed and report.miner_hotkey
        }
        for entry in paid:
            if entry.hotkey and qualified and entry.hotkey not in qualified:
                result.add(
                    "unqualified_recipient",
                    "error",
                    f"{entry.hotkey[:16]}… is paid but no published report shows it "
                    "clearing every gate",
                    subject,
                )

    return result


def audit_window(
    reports: list[EvaluationReport],
    vector: WeightVector | None,
    *,
    trusted_signers: set[str] | None = None,
    expected_snapshot: str | None = None,
    expected_base_revision: str | None = None,
) -> AuditResult:
    """Verify every report in a window and the vector derived from them."""
    result = AuditResult()

    for report in reports:
        verify_report(
            report,
            trusted_signers=trusted_signers,
            expected_snapshot=expected_snapshot,
            expected_base_revision=expected_base_revision,
            result=result,
        )

    crowned = [r for r in reports if r.verdict == "dethrone"]
    if len(crowned) > 1:
        result.add(
            "multiple_dethrones",
            "warning",
            f"{len(crowned)} reports claim a dethrone in this window: "
            f"{[r.candidate_id[:12] for r in crowned]}. Only the last can be the "
            "standing champion.",
        )

    if vector is not None:
        verify_weight_vector(vector, reports, trusted_signers=trusted_signers, result=result)

    return result
