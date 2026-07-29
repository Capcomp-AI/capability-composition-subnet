"""One problem, reconstructible from its seed alone."""

from __future__ import annotations

from dataclasses import dataclass

from capability_subnet.workflows.lora_merger_logic_v1.dataset import select


@dataclass(frozen=True, slots=True)
class LogicInstance:
    """A single question with an exact expected answer."""

    instance_id: str
    seed: int
    split: str
    task: str
    question: str
    answer: str
    difficulty: float


def generate_instance(seed: int, *, split: str = "hidden") -> LogicInstance:
    """The instance this seed denotes.

    Deterministic in the seed and the pinned revision, which is what lets an
    auditor replay a closed window: the published seed regenerates the exact
    problem the candidate was given.

    ``split`` names which draw an instance came from and does not change the
    problem. Out-of-distribution here means a different region of the same
    corpus rather than a mutation, because the corpus is fixed — a limitation
    worth knowing when reading an OOD score on this workflow.
    """
    item = select(seed)
    return LogicInstance(
        instance_id=f"{split}-{seed:012d}",
        seed=seed,
        split=split,
        task=item.task,
        question=item.question,
        answer=item.answer,
        difficulty=item.difficulty,
    )
