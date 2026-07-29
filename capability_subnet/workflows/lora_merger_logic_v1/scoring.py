"""Exact-match scoring for the pinned logic corpus.

Every item states the wrapper it wants its answer in — a fenced block,
``\\boxed{}``, or double square brackets — and carries the exact answer. So
scoring is a string comparison, and no model judges anything. That is the same
guarantee the V1 workflow gets from executing SQL and running hidden tests, and
it is the property that makes a published score checkable by someone who is not
the operator.

Extraction tries each wrapper and then the whole reply. Falling back to the
whole reply is deliberate: a package that produced the right answer without the
requested fence has demonstrated the capability, and format compliance is
measured as its own axis rather than folded into correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE = re.compile(r"```(?:python|json)?\s*(.*?)```", re.S)
_BOXED = re.compile(r"\\boxed\{(.*?)\}", re.S)
_BRACKETS = re.compile(r"\[\[(.*?)\]\]", re.S)


def normalise(text: str) -> str:
    """Whitespace- and quote-insensitive comparison form.

    Deliberately forgiving about layout and quote style, because the corpus is
    inconsistent about both between families, and deliberately unforgiving about
    everything else.
    """
    text = text.strip().strip("`").strip()
    return re.sub(r"\s+", "", text).replace("'", '"').casefold()


def candidate_spans(reply: str) -> list[str]:
    """Every plausible answer span in a reply, best guess first."""
    if not reply:
        return []
    spans: list[str] = []
    for pattern in (_FENCE, _BOXED, _BRACKETS):
        found = pattern.findall(reply)
        if found:
            # The answer is stated last; earlier matches are usually the
            # worked example the prompt itself supplied.
            spans.append(found[-1])
    spans.append(reply)
    return spans


def answered_correctly(reply: str | None, answer: str) -> bool:
    target = normalise(answer)
    if not target or reply is None:
        return False
    for span in candidate_spans(reply):
        candidate = normalise(span)
        if candidate == target or candidate.strip('"[]') == target.strip('"[]'):
            return True
    return False


def used_requested_format(reply: str | None) -> bool:
    """Whether the reply wrapped its answer the way the prompt asked.

    Scored separately from correctness. A package that is right but cannot
    follow an output instruction is a different failure from one that is wrong,
    and a merged package losing format compliance is the single most common way
    merging goes bad — worth its own axis rather than hidden inside a pass/fail.
    """
    if not reply:
        return False
    return bool(_FENCE.search(reply) or _BOXED.search(reply) or _BRACKETS.search(reply))


@dataclass(frozen=True, slots=True)
class StageOutcome:
    score: float
    passed: bool
    detail: str = ""


def score_instance(instance, trace) -> dict[str, StageOutcome]:
    """Score one answered instance into its per-axis outcomes."""
    reply = trace.reply
    correct = answered_correctly(reply, instance.answer)
    formatted = used_requested_format(reply)

    return {
        instance.task: StageOutcome(
            float(correct), correct, "exact match" if correct else "did not match"
        ),
        "format_compliance": StageOutcome(
            float(formatted), formatted, "used the requested wrapper" if formatted else "did not"
        ),
    }
