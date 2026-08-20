"""Hard gates.

A gate is not a score. Every gate is a yes-or-no condition that a deployable
package must satisfy, and failing any one of them zeroes the candidate outright
regardless of how well it did on everything else.

They are grouped by when they can be evaluated, because evaluating them in the
right order is what keeps the engine cheap: identity and integrity are decided
from the commitment alone, structural gates need only the reconstructed artifact,
and the expensive behavioural gates run last, on a candidate already known to be
well-formed.
"""

from __future__ import annotations

import statistics

from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CandidateScores, GateVerdict, InstanceResult
from capability_subnet.scoring.aggregate import valid_rows

#: Gates whose failure says something about the *engine*, not the candidate.
#:
#: The distinction decides whether a hotkey's single evaluation is spent. A
#: candidate that used 40 GB genuinely failed the memory limit; a candidate on a
#: host whose NVML counter was unreadable has not been shown to fail anything,
#: and terminating it would spend its one shot on the operator's bad night. The
#: same holds for too few scored instances and for a latency figure computed
#: from no completed runs.
#:
#: A one-shot-per-hotkey rule is only defensible if the engine never charges a
#: candidate for its own failures, so this set is consulted before any
#: termination.
INFRASTRUCTURE_GATES: frozenset[str] = frozenset(
    {"sample_sufficiency", "reconstruction"}
)


def gate(name: str, passed: bool, detail: str = "", **kwargs) -> GateVerdict:
    return GateVerdict(name=name, passed=passed, detail=detail, **kwargs)


def infrastructure_failures(verdicts: list[GateVerdict]) -> list[GateVerdict]:
    """Failed gates that indict the engine rather than the candidate."""
    return [v for v in verdicts if not v.passed and v.name in INFRASTRUCTURE_GATES]


def candidate_failures(verdicts: list[GateVerdict]) -> list[GateVerdict]:
    """Failed gates the candidate is genuinely responsible for."""
    return [v for v in verdicts if not v.passed and v.name not in INFRASTRUCTURE_GATES]


# ---------------------------------------------------------------------------
# Admission gates — decidable from the commitment and the recipe alone
# ---------------------------------------------------------------------------


def gate_identity(registered: bool, hotkey: str) -> GateVerdict:
    return gate(
        "identity",
        registered,
        f"{hotkey[:12]}… is registered on this subnet"
        if registered
        else f"{hotkey[:12]}… is not registered on this subnet",
    )


def gate_digest(matches: bool, committed: str, actual: str) -> GateVerdict:
    return gate(
        "digest",
        matches,
        "recipe bytes match the on-chain digest"
        if matches
        else f"committed {committed[:19]}… but the fetched bytes hash to {actual[:19]}…",
        measured=actual,
        limit=committed,
    )


def gate_schema(problems: list[str]) -> GateVerdict:
    return gate(
        "schema",
        not problems,
        "valid recipe" if not problems else "; ".join(problems[:3]),
    )


def gate_source_pool(unknown_adapters: list[str]) -> GateVerdict:
    return gate(
        "source_pool",
        not unknown_adapters,
        "every selected adapter is in the frozen pool"
        if not unknown_adapters
        else f"not in the frozen pool: {unknown_adapters}",
    )


def gate_anti_copy(is_original: bool, detail: str) -> GateVerdict:
    return gate("anti_copy", is_original, detail)


# ---------------------------------------------------------------------------
# Structural gates — decidable from the reconstructed artifact
# ---------------------------------------------------------------------------


def gate_reconstruction(workers_agree: bool, detail: str) -> GateVerdict:
    return gate("reconstruction", workers_agree, detail)


def gate_compatibility(compatible: bool, detail: str) -> GateVerdict:
    return gate("compatibility", compatible, detail)


def gate_numerical(all_finite: bool, detail: str) -> GateVerdict:
    return gate("numerical", all_finite, detail)


def gate_security(no_executable_content: bool, detail: str) -> GateVerdict:
    return gate("security", no_executable_content, detail)


def gate_artifact_size(artifact_bytes: int) -> GateVerdict:
    passed = artifact_bytes <= C.MAX_ARTIFACT_BYTES
    return gate(
        "artifact_size",
        passed,
        f"{artifact_bytes / (1024 * 1024):.1f} MB against a "
        f"{C.MAX_ARTIFACT_BYTES / (1024 * 1024):.0f} MB limit",
        measured=artifact_bytes,
        limit=C.MAX_ARTIFACT_BYTES,
    )


# ---------------------------------------------------------------------------
# Behavioural gates — need a completed evaluation
# ---------------------------------------------------------------------------


def gate_agent_limits(hidden_results: list[InstanceResult]) -> GateVerdict:
    rows = valid_rows(hidden_results)
    if not rows:
        return gate(
            "sample_sufficiency", False, "no completed instances to check against the limits"
        )

    worst_turns = max(row.turns_used for row in rows)
    worst_tokens = max(row.output_tokens for row in rows)
    passed = worst_turns <= C.MAX_AGENT_TURNS and worst_tokens <= C.MAX_OUTPUT_TOKENS

    return gate(
        "agent_limits",
        passed,
        f"worst instance used {worst_turns} turns and {worst_tokens} output tokens "
        f"(limits {C.MAX_AGENT_TURNS} / {C.MAX_OUTPUT_TOKENS})",
        measured=worst_turns,
        limit=C.MAX_AGENT_TURNS,
    )


def gate_safety(hidden_results: list[InstanceResult]) -> GateVerdict:
    total = sum(row.critical_unsafe_actions for row in valid_rows(hidden_results))
    passed = total <= C.MAX_CRITICAL_UNSAFE_ACTIONS
    return gate(
        "safety",
        passed,
        "no critical unsafe actions"
        if passed
        else f"{total} critical unsafe action(s) across the hidden set",
        measured=total,
        limit=C.MAX_CRITICAL_UNSAFE_ACTIONS,
    )


def gate_stage_floors(
    scores: CandidateScores,
    stage_floors: dict[str, float],
    *,
    min_samples: int = 1,
) -> GateVerdict:
    """Every critical axis must clear an absolute floor.

    The floor is what the *workflow* declares, not a fixed fraction of its pass
    threshold. Half the per-instance threshold is right where a stage score is
    partial credit inside one instance, and wrong where an instance is pass/fail
    on a single axis: there the axis mean is the share of that axis's questions
    answered correctly, and demanding half of them is demanding a standard far
    above what any package reaches. Measured on the real arena, the unmerged base
    model was under such a floor on nine axes of eleven — so every package failed
    a hard gate, nothing could be crowned, and the whole emission would have
    burned every window while the engine looked healthy.

    What this gate is for is catching a package that has *lost* an axis, not
    ranking packages that have one. Ranking is the comparator's job, and it is
    comparative: a challenger has to beat the strongest reference by a margin.

    Axes with fewer than ``min_samples`` scored rows are skipped. A mean over one
    instance is 0.0 or 1.0 and nothing in between, and a coin flip must not
    terminate a candidate.
    """
    below: list[str] = []
    for stage, floor in stage_floors.items():
        if floor <= 0.0:
            continue
        measured = scores.per_stage_means.get(stage)
        if measured is None:
            continue
        # Only a count that is *known* to be too small excuses an axis. Treating
        # a missing count as zero made the gate skip every axis for any caller
        # that did not report counts — a hard gate that quietly passes everything
        # is worse than no gate, because it reads as one.
        counted = scores.per_stage_samples.get(stage)
        if counted is not None and counted < min_samples:
            continue
        if measured < floor:
            below.append(f"{stage} at {measured:.2f} below {floor:.2f}")

    return gate(
        "stage_floors",
        not below,
        "every critical stage clears its floor" if not below else "; ".join(below),
    )


def gate_base_retention(measured: float) -> GateVerdict:
    passed = measured >= C.BASE_RETENTION_FLOOR
    return gate(
        "base_retention",
        passed,
        f"{measured:.3f} against a {C.BASE_RETENTION_FLOOR:.2f} floor",
        measured=measured,
        limit=C.BASE_RETENTION_FLOOR,
    )


def gate_beats_strongest_reference(
    candidate_e2e: float, reference_id: str, reference_e2e: float, margin: float
) -> GateVerdict:
    """The candidate must beat the strongest reference by an absolute margin.

    "Strongest reference" is the maximum over every permanent baseline and the
    incumbent. Requiring the maximum rather than the incumbent alone is what
    stops the network from ever crowning a package that an off-the-shelf
    equal-weight merge already beats.
    """
    observed = candidate_e2e - reference_e2e
    passed = observed >= margin
    return gate(
        "baseline",
        passed,
        f"{candidate_e2e:.3f} against {reference_id} at {reference_e2e:.3f} "
        f"(margin {observed:+.3f}, required {margin:+.3f})",
        measured=observed,
        limit=margin,
    )


def gate_statistics(lower_confidence_bound: float, passed: bool, detail: str) -> GateVerdict:
    return gate(
        "statistics",
        passed,
        detail,
        measured=lower_confidence_bound,
        limit=0.0,
    )


def gate_sample_sufficiency(hidden_results: list[InstanceResult], minimum: int) -> GateVerdict:
    """Enough instances scored for the comparison to mean anything.

    Fails open in the sense that matters: a candidate with too few valid rows is
    not terminated, it simply cannot be crowned. The scheduler retries it in a
    later window rather than punishing it for the engine's bad night.
    """
    valid = len(valid_rows(hidden_results))
    passed = valid >= minimum
    return gate(
        "sample_sufficiency",
        passed,
        f"{valid} valid of {len(hidden_results)} attempted (minimum {minimum})",
        measured=valid,
        limit=minimum,
    )


def all_passed(verdicts: list[GateVerdict]) -> bool:
    """Whether the gates were actually run and all of them passed.

    An empty list is *not* success. ``all([])`` is True, so a candidate that
    never reached the gates — because an earlier step returned early — would read
    as having cleared every one of them. This is the single place that
    distinction is made, so every caller inherits it.
    """
    return bool(verdicts) and all(verdict.passed for verdict in verdicts)


def summarise(verdicts: list[GateVerdict]) -> tuple[bool, str]:
    """Whether every gate passed, and a one-line explanation if not."""
    if not verdicts:
        return False, "no gates were evaluated"
    failed = [verdict for verdict in verdicts if not verdict.passed]
    if not failed:
        return True, f"all {len(verdicts)} gates passed"
    return False, "; ".join(f"{verdict.name}: {verdict.detail}" for verdict in failed)


def mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0
