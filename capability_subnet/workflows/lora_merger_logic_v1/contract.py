"""The published contract miners build against."""

from __future__ import annotations

from typing import Any

from capability_subnet import __spec_version__
from capability_subnet.common import constants as C
from capability_subnet.workflows.lora_merger_logic_v1 import sources as S
from capability_subnet.workflows.lora_merger_logic_v1.runner import MAX_OUTPUT_TOKENS
from capability_subnet.workflows.shared_contract import (
    base_model_contract,
    hard_gates_contract,
    incentive_contract,
    qualified_scoring_contract,
    ranking_contract,
    recipe_contract,
    resolve_snapshot,
    source_pool_contract,
    windows_contract,
)


def build_contract(snapshot=None, seed_root_commitment: str = "") -> dict[str, Any]:
    """Everything a miner needs to know about how a package is judged here,
    including the limitations.

    Args:
        snapshot: the frozen pool. The engine passes its own so the published
            contract describes the pool actually being evaluated against; the CLI
            passes nothing and the shipped one is loaded.
    """
    snapshot = resolve_snapshot(snapshot)

    return {
        "contract_version": 1,
        "spec_version": __spec_version__,
        "workflow_id": "lora_merger_logic_v1",
        "base_model": base_model_contract(snapshot),
        "source_pool": source_pool_contract(snapshot),
        "recipe": recipe_contract(snapshot),
        "title": "LoRA Merger Logic — single-turn reasoning and execution-verified code",
        "corpus": {
            "sources": [
                {
                    "name": source.name,
                    "dataset": source.repo,
                    "revision": source.revision,
                    "scoring": "exact match" if source.kind == "logic" else "execution",
                    "detail": source.detail,
                }
                for source in S.SOURCES
            ],
            "code_fraction": S.CODE_FRACTION,
            "difficulty_column": S.DIFFICULTY_COLUMN,
            "difficulty_band": list(S.DIFFICULTY_BAND),
            "code_difficulties": sorted(S.CODE_DIFFICULTIES),
            "task_families": [*S.LOGIC_FAMILIES, S.CODE_FAMILY],
            "note": (
                "Both corpora are public and pinned. The hidden seed protects which "
                "items a window draws, not the items themselves. Roughly 3,200 logic "
                "items carry the difficulty labels selection needs and roughly 2,900 "
                "code problems are admitted, so corpus size is not the mitigation: "
                "stratified selection, the executed quarter of each window, and a "
                "general-capability probe scored outside this corpus are."
            ),
        },
        "harness": {
            "turns": 1,
            "tools": [],
            "max_output_tokens": 1024,
            "temperature": C.SANDBOX_TEMPERATURE,
            "top_p": C.SANDBOX_TOP_P,
            "enable_thinking": C.SANDBOX_ENABLE_THINKING,
            "note": (
                "One question, one answer, no tools and no state. Sampling is greedy "
                "and seeded per instance, and the reasoning channel is disabled. The "
                "logic corpus difficulty labels were measured at pass@16 with sampling, "
                "so expect absolute scores well below the band."
            ),
        },
        "scoring": {
            "method": (
                "exact match after whitespace and quote normalisation for logic items; "
                "the submitted program is executed against stdin/stdout cases for code "
                "items, and must pass every case"
            ),
            "extraction": "the wrapper the prompt requested, then the whole reply",
            "axes": [*S.LOGIC_FAMILIES, S.CODE_FAMILY, "format_compliance"],
            "note": (
                "No language model judges any part of a result. Format compliance is "
                "its own axis rather than part of correctness, because a package that "
                "is right but can no longer follow an output instruction has failed "
                "differently from one that is wrong. Note that families whose answer is "
                "itself a bracketed list are compliant by construction, so the axis is "
                "informative mainly on the scalar families and on code."
            ),
        },
        "stages": {
            "order": list(S.STAGES),
            "thresholds": dict(S.STAGE_THRESHOLDS),
            # What terminates a submission, as opposed to what scores it. Only
            # format compliance carries a floor here; capability is judged
            # against the references, not against an absolute bar.
            "floors": dict(S.STAGE_FLOORS),
            # Every axis is critical here; a package that abandoned a family has
            # not solved the arena.
            "critical_axes": list(S.STAGES),
        },
        "limits": {
            "max_selected_adapters": C.MAX_SELECTED_ADAPTERS,
            "min_selected_adapters": C.MIN_SELECTED_ADAPTERS,
            "max_cases_per_problem": S.MAX_CASES_PER_PROBLEM,
            "note": (
                "A submission must combine at least two adapters; a single adapter is "
                "a reference, not a candidate. Code problems are scored on at most the "
                "first 32 test cases."
            ),
        },
        "hard_gates": hard_gates_contract(
            # This workflow's own budgets, not the protocol's defaults: one
            # question, one answer, a thousand tokens to give it in.
            max_output_tokens=MAX_OUTPUT_TOKENS,
            max_turns=1,
            stage_floors=dict(S.STAGE_FLOORS),
        ),
        "qualified_score": qualified_scoring_contract(),
        "champion_challenge": ranking_contract(),
        "windows": windows_contract(seed_root_commitment),
        "incentive": incentive_contract(),
    }
