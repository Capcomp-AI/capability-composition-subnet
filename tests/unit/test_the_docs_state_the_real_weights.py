"""The scoring weights the docs publish must be the ones the code applies.

They drifted, and silently: the weights were changed to an experimental split
and changed back, and four documents kept the experimental figures. A miner
reading them would have optimised for stage balance at 95% when it carries 15%,
and the published grade table summed to 145%.

Nothing else pins these — the docs are prose, and prose does not fail a build.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from capability_subnet.common import constants as C

DOCS = Path(__file__).resolve().parents[2]
SOURCES = [DOCS / "README.md", *sorted((DOCS / "docs").glob("*.md"))]
BLOB = "\n".join(p.read_text(encoding="utf-8") for p in SOURCES)


def as_percent(weight: float) -> str:
    return f"{weight * 100:g}%"


@pytest.mark.parametrize(
    "axis,weight", sorted(C.QUALIFIED_SCORE_WEIGHTS.items(), key=lambda kv: -kv[1])
)
def test_every_quality_weight_is_published(axis, weight):
    assert as_percent(weight) in BLOB, (
        f"{axis} carries {as_percent(weight)} and no document says so"
    )


@pytest.mark.parametrize(
    "term,weight",
    [
        ("quality", C.CONTRIBUTION_WEIGHT_QUALITY),
        ("improvement", C.CONTRIBUTION_WEIGHT_IMPROVEMENT),
        ("cost", C.CONTRIBUTION_WEIGHT_COST),
    ],
)
def test_every_grade_weight_is_published(term, weight):
    assert as_percent(weight) in BLOB, (
        f"the {term} term carries {as_percent(weight)} and no document says so"
    )


def test_no_document_states_a_weight_the_code_does_not_use():
    """The specific way this broke: figures left behind by a reverted change."""
    live = {as_percent(w) for w in C.QUALIFIED_SCORE_WEIGHTS.values()}
    live |= {
        as_percent(C.CONTRIBUTION_WEIGHT_QUALITY),
        as_percent(C.CONTRIBUTION_WEIGHT_IMPROVEMENT),
        as_percent(C.CONTRIBUTION_WEIGHT_COST),
    }
    # Percentages that appeared in a scoring table under a reverted weighting.
    for stale in ("3.75%", "1.25%", "0.0375", "0.0125"):
        assert stale not in BLOB, f"{stale} is a weight the code has never applied"

    # And the grade table must not sum to something other than one.
    for source in SOURCES:
        text = source.read_text(encoding="utf-8")
        for table in re.findall(r"\| Term \| Weight \|.*?(?=\n\n)", text, re.S):
            percents = [float(m) for m in re.findall(r"\|\s*(\d+(?:\.\d+)?)%\s*\|", table)]
            if len(percents) >= 3:
                assert abs(sum(percents) - 100.0) < 0.01, (
                    f"{source.name}: a grade table sums to {sum(percents)}%"
                )
