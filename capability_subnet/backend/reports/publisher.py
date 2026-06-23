"""Signed evaluation reports.

A report is the engine's complete account of one evaluation: what was measured,
against what, under which gates, with which statistics, and what was decided. It
is signed with the operator's hotkey and published.

This is what makes a centralised evaluation engine auditable rather than merely
convenient. Anyone can take the stream of reports and re-derive the weight vector
independently; anyone with the published recipe and the pool can rebuild the
artifact and confirm its digest; and nothing the engine publishes can later be
disowned, because it is signed.

What a report deliberately does *not* contain is the hidden instances. It carries
the seeds' window identifier and the aggregate outcomes, not the problems
themselves, because publishing those would end the arena.
"""

from __future__ import annotations

import logging
from pathlib import Path

from capability_subnet import __spec_version__
from capability_subnet.backend.evaluation import EvaluationOutput
from capability_subnet.common.hashing import canonical_json_str, sha256_bytes
from capability_subnet.common.schemas import (
    ComparatorOutcome,
    EvaluationReport,
    GateVerdict,
)
from capability_subnet.common.signing import sign_in_place

log = logging.getLogger(__name__)


def build_report(
    output: EvaluationOutput,
    *,
    window_id: int,
    block: int,
    workflow_id: str,
    base_revision: str,
    source_snapshot_sha256: str,
    evaluator_image_digest: str,
    miner_hotkey: str = "",
    miner_uid: int | None = None,
    recipe_sha256: str | None = None,
    baseline_scores: dict[str, float] | None = None,
    strongest_reference_id: str = "",
    strongest_reference_score: float = 0.0,
    comparator: ComparatorOutcome | None = None,
    verdict: str = "held",
    verdict_reason: str = "",
    extra_gates: list[GateVerdict] | None = None,
) -> EvaluationReport:
    """Assemble the report for one evaluation."""
    return EvaluationReport(
        workflow_id=workflow_id,
        window_id=window_id,
        evaluated_at_block=block,
        miner_hotkey=miner_hotkey,
        miner_uid=miner_uid,
        candidate_id=output.candidate_id,
        recipe_sha256=recipe_sha256,
        artifact_sha256=output.artifact_sha256,
        base_revision=base_revision,
        source_snapshot_sha256=source_snapshot_sha256,
        evaluator_image_digest=evaluator_image_digest,
        spec_version=__spec_version__,
        hard_gates=list(output.gate_verdicts) + list(extra_gates or []),
        scores=output.scores,
        resources=output.resources,
        baseline_scores=dict(baseline_scores or {}),
        strongest_reference_id=strongest_reference_id,
        strongest_reference_score=strongest_reference_score,
        comparator=comparator,
        verdict=verdict,  # type: ignore[arg-type]
        verdict_reason=verdict_reason,
    )


class ReportPublisher:
    """Signs reports and writes them where validators can read them."""

    def __init__(
        self,
        directory: str | Path,
        keypair=None,
        *,
        signer_hotkey: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keypair = keypair
        self.signer_hotkey = signer_hotkey

        if keypair is None:
            log.warning(
                "no signing key configured; published reports will be unsigned and "
                "validators enforcing an operator allow-list will refuse them"
            )

    def publish(self, report: EvaluationReport) -> str:
        """Sign, store and return the report's digest.

        The digest is taken over the *unsigned* canonical bytes, so it identifies
        the content rather than the signature. Re-signing after a key rotation
        therefore does not change a report's identity, and a champion record that
        points at a report stays valid across one.
        """
        if self.keypair is not None:
            sign_in_place(self.keypair, report)
        elif self.signer_hotkey:
            report.signer_hotkey = self.signer_hotkey

        digest = sha256_bytes(report.signable_bytes())
        target = self._path_for(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            canonical_json_str(report.model_dump(mode="json", exclude_none=True)),
            encoding="utf-8",
        )

        log.info(
            "published report %s for %s: %s (%s)",
            digest[:19],
            report.candidate_id,
            report.verdict,
            report.verdict_reason or "no reason recorded",
        )
        return digest

    def _path_for(self, digest: str) -> Path:
        short = digest.split(":", 1)[-1]
        return self.directory / short[:2] / f"{short}.json"

    def load(self, digest: str) -> EvaluationReport | None:
        path = self._path_for(digest)
        if not path.is_file():
            return None
        return EvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))


def compatibility_record(output: EvaluationOutput, recipe) -> dict:
    """The row appended to the compatibility history for one evaluation.

    Over many windows this table answers the questions the network exists to
    answer: which adapters transfer positively together, which conflict, which
    layer depths need a domain specialist, when trimming beats dropping, and how
    much rank a workflow actually needs. One row on its own says very little,
    which is exactly why it is recorded on every evaluation rather than only on
    the interesting ones.
    """
    stats = output.build.result.stats if output.build is not None else None

    return {
        "candidate_id": output.candidate_id,
        "selected_adapters": sorted(recipe.selected_adapters) if recipe else [],
        "combination_type": recipe.merge.combination_type if recipe else None,
        "density": recipe.merge.density if recipe else None,
        "output_rank": recipe.compression.output_rank if recipe else None,
        "svd_clamp_quantile": recipe.compression.svd_clamp_quantile if recipe else None,
        "global_weights": dict(recipe.global_weights) if recipe else {},
        "layer_group_overrides": (
            {group: dict(values) for group, values in recipe.layer_group_overrides.items()}
            if recipe
            else {}
        ),
        "end_to_end": output.scores.end_to_end,
        "stage_balance": output.scores.stage_balance,
        "ood": output.scores.ood,
        "retention": output.scores.retention,
        "qualified_score": output.scores.qualified_score,
        "per_stage_means": dict(output.scores.per_stage_means),
        "adapter_mb": output.resources.adapter_mb,
        "peak_vram_gb": output.resources.peak_vram_gb,
        "p95_workflow_seconds": output.resources.p95_workflow_seconds,
        "mean_retained_energy": stats.mean_retained_energy if stats else None,
        "min_retained_energy": stats.min_retained_energy if stats else None,
        "contribution_by_group": stats.contribution_by_group if stats else {},
        "failed_gates": [v.name for v in output.gate_verdicts if not v.passed],
    }
