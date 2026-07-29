"""The published contract miners build against."""

from __future__ import annotations

from typing import Any

from capability_subnet.common import constants as C
from capability_subnet.workflows.lora_merger_logic_v1 import dataset


def build_contract() -> dict[str, Any]:
    """Everything a miner needs to know about how a package is judged here.

    Including the parts that are limitations. A contract that published only the
    favourable properties would be marketing; the two paragraphs below on corpus
    visibility and difficulty calibration are exactly what a miner needs in order
    to decide whether to compete.
    """
    return {
        "workflow_id": "lora_merger_logic_v1",
        "title": "LoRA Merger Logic — single-turn exact-match reasoning",
        "corpus": {
            "dataset": dataset.DATASET,
            "revision": dataset.REVISION,
            "difficulty_column": dataset.DIFFICULTY_COLUMN,
            "difficulty_band": list(dataset.BAND),
            "task_families": list(dataset.TASK_FAMILIES),
            "note": (
                "The corpus is public and pinned. What the hidden seed protects is "
                "which items a window draws, not the items themselves — a weaker "
                "anti-overfitting property than a generated workflow, and the price "
                "of a corpus whose difficulty is already measured."
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
                "and seeded per instance, and the model's separate reasoning channel "
                "is disabled — so the difficulty labels in the corpus, which were "
                "measured at pass@16 with sampling, overstate what this harness "
                "observes. Expect materially lower absolute scores than the band."
            ),
        },
        "scoring": {
            "method": "exact match after whitespace and quote normalisation",
            "extraction": "the wrapper the prompt requested, then the whole reply",
            "axes": list(dataset.TASK_FAMILIES) + ["format_compliance"],
            "note": (
                "No language model judges any part of a result. Format compliance is "
                "its own axis rather than part of correctness: a package that is right "
                "but can no longer follow an output instruction has failed differently "
                "from one that is wrong, and losing compliance is the most common way "
                "an over-aggressive merge goes bad."
            ),
        },
        "limits": {
            "max_selected_adapters": C.MAX_SELECTED_ADAPTERS,
            "min_selected_adapters": C.MIN_SELECTED_ADAPTERS,
            "note": (
                "A submission must combine at least two adapters. A single adapter is "
                "a reference, not a candidate."
            ),
        },
    }
