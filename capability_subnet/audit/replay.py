"""Re-scoring a closed window from its disclosure.

The strongest check anyone outside the engine can run, and it needs no GPU.

Hidden instances are drawn fresh every window and never reused, so once a window
closes its instances have no value as a secret and considerable value as
evidence. The engine publishes their seeds together with the traces it scored.
An auditor then regenerates each instance from its seed — instance generation is
a pure function of the seed, so this reproduces the exact problem the candidate
faced — and re-runs the deterministic scorer over the published trace.

If the engine's arithmetic was honest, every stage score matches. If it was not,
the disagreement is specific: this instance, this stage, this number against that
one.

What this cannot check is whether the trace is a faithful record of what the
model actually did. A determined operator could publish a fabricated trace that
scores exactly as claimed. What it does mean is that fabrication has to be
carried all the way down to per-turn tool calls that are internally consistent
with a workflow the auditor can regenerate — which is a great deal harder than
adjusting an aggregate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from capability_subnet.audit.verify import SCORE_TOLERANCE, AuditResult
from capability_subnet.common.schemas import InstanceResult, StageResult, WindowDisclosure
from capability_subnet.common.trace import ExecutionTrace, SqlSubmission, ToolCall
from capability_subnet.workflows import get_workflow

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ReplayOutcome:
    """What re-scoring one window produced."""

    checked: int = 0
    agreed: int = 0
    disagreed: list[str] = field(default_factory=list)
    unscorable: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.disagreed

    def summary(self) -> str:
        parts = [f"{self.agreed}/{self.checked} instances re-scored to the same result"]
        if self.disagreed:
            parts.append(f"{len(self.disagreed)} disagreed")
        if self.unscorable:
            parts.append(f"{len(self.unscorable)} could not be re-scored")
        return ", ".join(parts)


def trace_from_dict(payload: dict) -> ExecutionTrace:
    """Rebuild a trace from its published form.

    Only the fields the scorer reads are restored. Anything else in the payload
    is ignored rather than trusted, so a disclosure cannot smuggle a field the
    scorer would treat as authoritative.
    """
    trace = ExecutionTrace(
        instance_id=payload.get("instance_id", ""),
        instance_seed=int(payload.get("instance_seed", 0)),
        split=payload.get("split", "hidden"),
    )
    trace.stop_reason = payload.get("stop_reason", "submitted")
    trace.turns_used = int(payload.get("turns_used", 0))
    trace.input_tokens = int(payload.get("input_tokens", 0))
    trace.output_tokens = int(payload.get("output_tokens", 0))
    trace.wall_seconds = float(payload.get("wall_seconds", 0.0))
    trace.final_payload = payload.get("final_payload")
    trace.harness_error = payload.get("harness_error")

    trace.manual_queries = list(payload.get("manual_queries") or [])
    trace.manual_sections_read = list(payload.get("manual_sections_read") or [])
    trace.diagnostic_code_submissions = list(payload.get("diagnostic_code_submissions") or [])
    trace.diagnostic_tool_results = list(payload.get("diagnostic_tool_results") or [])
    trace.inventory_final_state = dict(payload.get("inventory_final_state") or {})
    trace.inventory_attempts = list(payload.get("inventory_attempts") or [])
    trace.safety_verdicts = list(payload.get("safety_verdicts") or [])

    trace.calls = [
        ToolCall(
            turn=int(call.get("turn", 0)),
            name=call.get("name", ""),
            arguments=call.get("arguments") or {},
            result=call.get("result"),
            ok=bool(call.get("ok", True)),
            error=call.get("error"),
        )
        for call in (payload.get("calls") or [])
    ]
    trace.sql_submissions = [
        SqlSubmission(
            statement=item.get("statement", ""),
            columns=tuple(item.get("columns") or ()),
            rows=tuple(tuple(row) for row in (item.get("rows") or ())),
            error=item.get("error"),
        )
        for item in (payload.get("sql_submissions") or [])
    ]
    return trace


def _compare(claimed: InstanceResult, rescored: InstanceResult) -> list[str]:
    """Differences between what the engine claimed and what re-scoring gives."""
    problems: list[str] = []

    if claimed.end_to_end_success != rescored.end_to_end_success:
        problems.append(
            f"end-to-end success claimed {claimed.end_to_end_success}, "
            f"re-scored {rescored.end_to_end_success}"
        )
    if claimed.final_state_correct != rescored.final_state_correct:
        problems.append(
            f"final state claimed {claimed.final_state_correct}, "
            f"re-scored {rescored.final_state_correct}"
        )
    if claimed.critical_unsafe_actions != rescored.critical_unsafe_actions:
        problems.append(
            f"critical unsafe actions claimed {claimed.critical_unsafe_actions}, "
            f"re-scored {rescored.critical_unsafe_actions}"
        )

    for stage, rescored_stage in rescored.stages.items():
        claimed_stage = claimed.stages.get(stage)
        if claimed_stage is None:
            problems.append(f"{stage}: not present in the claimed result")
            continue
        if abs(claimed_stage.score - rescored_stage.score) > SCORE_TOLERANCE:
            problems.append(
                f"{stage}: claimed {claimed_stage.score:.4f}, re-scored {rescored_stage.score:.4f}"
            )

    return problems


def check_draw_is_bound(disclosure: WindowDisclosure, result: AuditResult) -> None:
    """Whether this window's instance draw can be shown not to have been chosen.

    Replaying a window proves the published instances match the published seeds.
    It says nothing about where the *seeds* came from — and seeds derive from a
    root only the operator holds. An operator free to try roots until the draw
    suited a candidate they had already seen would pass every other check here.

    So two fields have to be present and stable: the beacon this window's draw was
    bound to, which is a block hash the operator does not choose, and the
    commitment to the seed root, which must be identical in every window of a
    deployment. Their absence is not proof of anything, which is why it is a
    warning — but a run of disclosures whose commitment moves is an operator
    changing the draw.
    """
    if not disclosure.root_commitment:
        result.add(
            "unbound_seed_root",
            "warning",
            "this window publishes no commitment to the operator's seed root, so "
            "the instance draw cannot be shown to be the same root as other windows",
            f"window {disclosure.window_id}",
        )
    if not disclosure.beacon:
        result.add(
            "unbound_draw",
            "warning",
            "this window's draw is not bound to a public beacon, so it rests "
            "entirely on the operator's secret root and cannot be shown to be "
            "unchosen",
            f"window {disclosure.window_id}",
        )


def commitments_agree(disclosures: list[WindowDisclosure]) -> tuple[bool, str]:
    """Whether a run of windows all derive from one seed root.

    The check a single disclosure cannot make. One commitment proves nothing; the
    same commitment across every window a deployment has published is what says
    the operator has not been re-rolling the draw.
    """
    seen = {d.root_commitment for d in disclosures if d.root_commitment}
    if not seen:
        return False, "no window published a seed-root commitment"
    if len(seen) > 1:
        return False, (
            f"{len(seen)} different seed-root commitments across "
            f"{len(disclosures)} windows; the draw was re-rooted"
        )
    missing = [d.window_id for d in disclosures if not d.root_commitment]
    if missing:
        return False, f"windows {missing} published no commitment"
    return True, f"one seed root across {len(disclosures)} windows"


def replay_disclosure(
    disclosure: WindowDisclosure,
    *,
    result: AuditResult | None = None,
) -> tuple[ReplayOutcome, AuditResult]:
    """Regenerate each disclosed instance and re-score its published trace."""
    result = result or AuditResult()
    outcome = ReplayOutcome()
    workflow = get_workflow(disclosure.workflow_id)

    check_draw_is_bound(disclosure, result)

    disclosed_seeds = set(disclosure.hidden_seeds) | set(disclosure.ood_seeds)

    for entry in disclosure.instances:
        outcome.checked += 1
        subject = f"{entry.candidate_id[:12]}/{entry.instance_id}"

        if disclosed_seeds and entry.instance_seed not in disclosed_seeds:
            result.add(
                "undisclosed_instance",
                "error",
                f"instance {entry.instance_id} carries seed {entry.instance_seed}, which "
                "is not among the seeds this window says it drew",
                subject,
            )

        try:
            instance = workflow.generate_instance(entry.instance_seed, split=entry.split)
        except Exception as exc:  # noqa: BLE001
            outcome.unscorable.append(entry.instance_id)
            result.add(
                "regeneration_failed",
                "error",
                f"could not regenerate the instance from seed {entry.instance_seed}: {exc}",
                subject,
            )
            continue

        if instance.instance_id != entry.instance_id:
            result.add(
                "instance_id_mismatch",
                "error",
                f"seed {entry.instance_seed} generates {instance.instance_id}, but the "
                f"disclosure labels it {entry.instance_id}",
                subject,
            )

        trace = trace_from_dict(entry.trace)
        if not trace.is_scorable:
            outcome.unscorable.append(entry.instance_id)
            continue

        try:
            score = workflow.score_instance(instance, trace)
        except Exception as exc:  # noqa: BLE001
            outcome.unscorable.append(entry.instance_id)
            result.add(
                "rescoring_failed",
                "error",
                f"the published trace could not be scored against the regenerated instance: {exc}",
                subject,
            )
            continue

        rescored = InstanceResult(
            instance_id=instance.instance_id,
            instance_seed=instance.seed,
            split=entry.split,  # type: ignore[arg-type]
            stages={
                name: StageResult(
                    stage=name, score=item.score, passed=item.passed, detail=item.detail
                )
                for name, item in score.stages.items()
            },
            final_state_correct=score.final_state_correct,
            end_to_end_success=score.end_to_end_success,
            critical_unsafe_actions=score.critical_unsafe_actions,
        )

        problems = _compare(entry.claimed_result, rescored)
        if problems:
            outcome.disagreed.append(entry.instance_id)
            result.add(
                "score_disagreement",
                "error",
                "re-scoring the published trace against the regenerated instance "
                "disagrees with the published result: " + "; ".join(problems[:4]),
                subject,
            )
        else:
            outcome.agreed += 1

    if outcome.checked == 0:
        result.add(
            "nothing_disclosed",
            "warning",
            f"window {disclosure.window_id} discloses no instances, so none of its "
            "scoring can be independently checked",
        )

    return outcome, result


def verify_disclosure(
    disclosure: WindowDisclosure,
    *,
    trusted_signers: set[str] | None = None,
) -> tuple[ReplayOutcome, AuditResult]:
    """Check the disclosure's attribution, then re-score everything in it."""
    from capability_subnet.common.signing import verify_payload

    result = AuditResult()
    subject = f"window {disclosure.window_id}"

    if not disclosure.signature or not disclosure.signer_hotkey:
        result.add("unsigned", "error", "the disclosure carries no signature", subject)
    elif trusted_signers is not None and disclosure.signer_hotkey not in trusted_signers:
        result.add(
            "untrusted_signer",
            "error",
            f"signed by {disclosure.signer_hotkey}, which is not on the allow-list",
            subject,
        )
    elif not verify_payload(disclosure, disclosure.signature, disclosure.signer_hotkey):
        result.add("bad_signature", "error", "the signature does not verify", subject)

    return replay_disclosure(disclosure, result=result)
