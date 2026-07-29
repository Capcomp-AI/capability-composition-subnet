"""LoRA Merger Logic — a single-turn, exactly-scored reasoning arena.

Ten task families from a pinned public corpus, each item stating its own output
format and carrying an exact answer. One question, one answer, string
comparison. No agent loop, no tool services, no language model judging anything.

Why this shape, next to the V1 maintenance workflow:

* **It discriminates for this model class.** Items are selected on the corpus's
  own measured pass rate for a 4B model, so a window contains problems an
  8B-class package can neither trivially solve nor never solve. The V1 workflow
  had no such calibration and its scripted oracle needs ten of its twelve turns,
  leaving a real model no room for a single wasted step.
* **The axes are genuinely independent.** Ten families that do not depend on one
  another, so per-axis dominance and stage balance measure breadth rather than
  ten views of one dependent chain.
* **It is cheap.** One turn instead of twelve, which is what makes a
  statistically honest sample size affordable at all.

What it gives up is stated in ``dataset.py``: the items are public, so the
secret seed protects only which of them a window draws.
"""

from __future__ import annotations

from capability_subnet.workflows.lora_merger_logic_v1.contract import build_contract
from capability_subnet.workflows.lora_merger_logic_v1.instance import (
    LogicInstance,
    generate_instance,
)
from capability_subnet.workflows.lora_merger_logic_v1.runner import run_instance
from capability_subnet.workflows.lora_merger_logic_v1.scoring import score_instance
from capability_subnet.workflows.lora_merger_logic_v1.sources import (
    STAGE_THRESHOLDS,
    STAGES,
)

WORKFLOW_ID = "lora_merger_logic_v1"
WORKFLOW_TITLE = "LoRA Merger Logic — single-turn reasoning and execution-verified code"

#: Every axis is critical: a package that abandoned one family has not solved the
#: arena, and the stage-balance term is a geometric mean for the same reason.
CRITICAL_AXES: tuple[str, ...] = STAGES

#: No tools. The whole point of this arena is that the package answers alone.
TOOL_SCHEMAS: list[dict] = []

__all__ = [
    "CRITICAL_AXES",
    "STAGES",
    "STAGE_THRESHOLDS",
    "TOOL_SCHEMAS",
    "WORKFLOW_ID",
    "WORKFLOW_TITLE",
    "LogicInstance",
    "build_contract",
    "generate_instance",
    "run_instance",
    "score_instance",
]
