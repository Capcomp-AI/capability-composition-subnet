"""Ranking submissions when the highest score wins.

The highest measured score ranks first, and nothing else enters the order. In
particular commit time does not: a submission is never advanced over another
because it was committed earlier. Two miners are compared on what they scored,
not on when they arrived.

The champion crown is protected on its own terms — a challenger has to beat the
strongest reference (the incumbent included) by an absolute end-to-end margin,
so a near-identical copy cannot take the throne it copied whatever its rank.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Submission:
    """One evaluated, gate-clearing submission competing for a rank."""

    uid: int
    hotkey: str
    score: float
    #: Block the commitment was made at. Retained because callers record it and
    #: audits read it, but it no longer affects the order — ranking is by score
    #: alone and gives no advantage to an earlier commitment.
    first_block: int
    #: What the window could resolve when this was measured. Kept for ``tied``
    #: and for audit, but no longer used to order submissions.
    resolvable: float = 0.0


def rank(submissions: list[Submission]) -> list[Submission]:
    """Order submissions best first, by measured score alone.

    Highest ``score`` ranks first. Commit time is not consulted: a lower-scoring
    submission is never placed above a higher-scoring one because it committed
    earlier. An exact score tie — which two distinct artifacts effectively never
    produce — falls back to ``uid`` only so the order is stable and reproducible,
    not to grant any submitter an advantage.
    """
    return sorted(submissions, key=lambda s: (-s.score, s.uid))


def tied(a: Submission, b: Submission) -> bool:
    """Whether two submissions are closer than the evidence can separate."""
    return abs(a.score - b.score) < max(a.resolvable, b.resolvable)
