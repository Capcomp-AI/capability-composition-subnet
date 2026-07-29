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
from capability_subnet.workflows.lora_merger_logic_v1.dataset import TASK_FAMILIES
from capability_subnet.workflows.lora_merger_logic_v1.instance import (
    LogicInstance,
    generate_instance,
)
from capability_subnet.workflows.lora_merger_logic_v1.runner import run_instance
from capability_subnet.workflows.lora_merger_logic_v1.scoring import score_instance

WORKFLOW_ID = "lora_merger_logic_v1"
WORKFLOW_TITLE = "LoRA Merger Logic — single-turn exact-match reasoning"

#: Each task family is an axis, plus format compliance.
#:
#: Compliance is separated on purpose. A merged package losing the ability to
#: follow an output instruction is the most common way merging goes wrong — the
#: first run of this corpus had a linear merge answering terse instructions with
#: prose and burning five times the tokens to do it — and folding that into
#: correctness would report it as "wrong" rather than "no longer obedient".
STAGES: tuple[str, ...] = (*TASK_FAMILIES, "format_compliance")
CRITICAL_AXES: tuple[str, ...] = STAGES

#: An instance exercises one family, so an axis is scored only on the instances
#: that belong to it. The threshold is exactness: a partially right answer to a
#: puzzle is a wrong answer.
STAGE_THRESHOLDS: dict[str, float] = {stage: 1.0 for stage in STAGES}

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
