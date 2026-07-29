"""Ranking submissions when the highest score wins.

A leaderboard is a simpler contract than champion-challenge and it removes a
real failure: under the old rule a challenger had to clear the strongest
permanent reference by an absolute margin, so a network whose best merge never
beat the base model would burn its emission indefinitely while producing
perfectly good comparative information. Paying the best merge on the board is a
decision the operator is entitled to make — the product is the best composition
anyone has found, not proof that composition was worth attempting.

It opens one thing champion-challenge closed, though, and it has to be closed
again here.

Under a margin rule, a copy of the leader could not displace the leader: an
identical package scores identically and identical is not a margin. Under a
leaderboard, an identical package *ties* — and since no two evaluations of two
distinct artifacts land on exactly the same number, a copy with one coefficient
nudged will sometimes score a hair higher and take the top slot on nothing but
sampling noise. Recipes are public, so this is not hypothetical: read the
leader's recipe, perturb it, resubmit.

The fix reuses the machinery the comparator already needs. Two scores closer
together than the window can resolve are not evidence that one package is
better; they are one measurement repeated. So submissions within the minimum
detectable effect of each other are ranked as **tied**, and ties resolve to the
earliest commitment. A copier commits later by construction and therefore can
never displace what it copied — it has to be *measurably* better, which is the
same bar the margin rule enforced, expressed in the units the evidence actually
supports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Submission:
    """One evaluated, gate-clearing submission competing for a rank."""

    uid: int
    hotkey: str
    score: float
    #: Block the commitment was made at. The tie-break, and the reason a copy
    #: cannot take a slot from what it copied.
    first_block: int
    #: What the window could resolve when this was measured. Two submissions
    #: closer than this are one measurement repeated, not two results.
    resolvable: float = 0.0


def rank(submissions: list[Submission]) -> list[Submission]:
    """Order submissions best first, treating unresolvable gaps as ties.

    Sorting on the raw score alone would let a difference smaller than the
    evidence supports decide who gets paid — which is the same error as
    demanding a margin the sample size cannot demonstrate, made in the opposite
    direction.

    Submissions are sorted by score, then cut into runs of consecutive entries
    no further apart than the window can resolve. Each run is one equivalence
    class — the measurements do not separate its members — and inside it the
    earliest commitment ranks first.

    Grouping rather than pairwise swapping, because indistinguishability is not
    transitive and a single pass gets it wrong: with three submissions each
    within tolerance of the next, one swap leaves a later entry ahead of an
    earlier one it cannot be shown to beat, which is precisely the hole this
    exists to close.

    The chaining that grouping implies is a real cost and worth stating: a long
    ladder of small gaps becomes one class, so a submission can be ranked equal
    to another it would beat in a direct comparison. That errs toward commit
    priority, which is the safe direction — it can delay a genuine improvement
    by a window, where the alternative hands slots to copies.
    """
    ordered = sorted(submissions, key=lambda s: (-s.score, s.first_block, s.uid))

    ranked: list[Submission] = []
    run: list[Submission] = []
    for entry in ordered:
        if run:
            gap = run[-1].score - entry.score
            tolerance = max(run[-1].resolvable, entry.resolvable)
            if gap >= tolerance:
                ranked.extend(sorted(run, key=lambda s: (s.first_block, s.uid)))
                run = []
        run.append(entry)
    if run:
        ranked.extend(sorted(run, key=lambda s: (s.first_block, s.uid)))

    for before, after in zip(ordered, ranked, strict=True):
        if before.uid != after.uid:
            log.info(
                "commit order decided a tie: %s ranks where score alone put %s, "
                "because the gap is inside what this window could resolve",
                after.hotkey[:12],
                before.hotkey[:12],
            )
            break

    return ranked


def tied(a: Submission, b: Submission) -> bool:
    """Whether two submissions are closer than the evidence can separate."""
    return abs(a.score - b.score) < max(a.resolvable, b.resolvable)
