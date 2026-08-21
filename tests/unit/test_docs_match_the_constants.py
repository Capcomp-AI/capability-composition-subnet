"""The published numbers are the protocol's numbers.

Documentation drifts silently. A miner reading that a recipe may name twelve
adapters writes one that names twelve, and finds out it cannot when the engine
rejects it — the doc was wrong for long enough that seven submissions in one
run carried eleven. These check the statements most likely to be read as
authoritative against the constants they are meant to describe.

Only statements in the present tense are checked. A line recording what a value
*used to be* is history, and rewriting history each time a constant moves is how
the reason for a change gets lost.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from capability_subnet.common import constants as C

REPO = Path(__file__).resolve().parents[2]

#: Phrasings that mark a line as a record of the past rather than a claim about
#: the present. Kept narrow on purpose: "was" and "no longer" are explicit,
#: where a looser rule would hide a genuinely stale sentence.
HISTORICAL = re.compile(
    r"\b(was|were|used to|no longer|previously|until|earlier|retired|removed|"
    r"predate|reverted|now\))\b",
    re.IGNORECASE,
)

#: (what the constant says, pattern that only a contradicting statement matches)
CHECKS = [
    ("selected_adapters is 2-%d" % C.MAX_SELECTED_ADAPTERS,
     r"2 (and|to) 12|at most 12 adapter|maxItems\"?:? ?12"),
    ("retention floor is %.2f" % C.BASE_RETENTION_FLOOR,
     r"0\.98 retention|retention floor of 0\.98|0\.98 on a held-out"),
    ("end-to-end margin is %.2f" % C.DEFAULT_END_TO_END_MARGIN,
     r"end_to_end_margin[`\s]*(to|:)\s*0\.0[26]\b|margin of 0\.0[26]\b"),
    ("hidden instances are %d" % C.DEFAULT_HIDDEN_INSTANCES,
     r"hidden_instances: (?!1350\b)\d+"),
    ("the serving reservation is %.0f GiB" % C.SERVING_RESERVED_GIB,
     r"\b(fixed )?\*{0,2}20 GiB\*{0,2}\b|0\.4168"),
    ("the served context is %d" % C.SERVING_MAX_MODEL_LEN,
     r"max_model_len: (?!8192\b)\d+"),
    ("serving is batched at %d" % C.SANDBOX_BATCH_CONCURRENCY,
     r"max-num-seqs 1\b|one sequence at a time"),
    ("the validator floor is %d cards" % C.MIN_VALIDATOR_CARDS,
     r"\*\*8 × RTX 5090 \(32 GB\)\*\*, one candidate"),
    ("the only reference is the base model",
     r"single_adapter_rotation` \| `|best single adapter, the standard merges"),
    ("peak VRAM is neither gated nor scored", r"peak_vram|peak-VRAM gate"),
]

DOC_GLOBS = ("*.md", "*.yml", "*.yaml")


def _doc_lines() -> list[tuple[str, int, str]]:
    """Every line of every tracked document, with its origin.

    Tracked files only, via git: an untracked scratch file or a vendored copy is
    not something the network publishes, and scanning one would report a problem
    nobody can act on.
    """
    listed = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    files = [
        REPO / name
        for name in listed.stdout.split()
        if name.endswith((".md", ".yml", ".yaml"))
    ]
    assert files, "no documentation files found; every check would pass vacuously"

    lines: list[tuple[str, int, str]] = []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lines.append((str(path.relative_to(REPO)), number, line))
    return lines


@pytest.mark.parametrize("label,pattern", CHECKS, ids=[c[0] for c in CHECKS])
def test_no_document_contradicts_the_constant(label, pattern):
    rule = re.compile(pattern)
    offenders = [
        f"{name}:{number}: {line.strip()[:100]}"
        for name, number, line in _doc_lines()
        if rule.search(line) and not HISTORICAL.search(line)
    ]
    assert not offenders, f"documentation contradicts that {label}:\n" + "\n".join(offenders)
