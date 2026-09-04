"""The published numbers are the protocol's numbers.

Documentation drifts silently. A miner reading that a recipe may name twelve
adapters writes one that names twelve, and finds out it cannot when the engine
rejects it - the doc was wrong for long enough that seven submissions in one
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
    (
        f"selected_adapters is 2-{C.MAX_SELECTED_ADAPTERS}",
        r"2 (and|to) 12|at most 12 adapter|maxItems\"?:? ?12",
    ),
    (
        f"retention floor is {C.BASE_RETENTION_FLOOR:.2f}",
        r"0\.98 retention|retention floor of 0\.98|0\.98 on a held-out",
    ),
    (
        f"end-to-end margin is {C.DEFAULT_END_TO_END_MARGIN}",
        # Any margin that is not the configured one. Built from the constant so
        # the check follows it: written out by hand, the pattern goes on naming
        # whatever the value used to be and flags the correct number instead.
        rf"end_to_end_margin[`\s]*(to|:)\s*(?!{C.DEFAULT_END_TO_END_MARGIN}\b)0\.\d+"
        rf"|margin of (?!{C.DEFAULT_END_TO_END_MARGIN}\b)0\.\d+",
    ),
    (
        f"hidden instances are {C.DEFAULT_HIDDEN_INSTANCES}",
        rf"hidden_instances: (?!{C.DEFAULT_HIDDEN_INSTANCES}\b)\d+",
    ),
    (
        f"the anti-copy window is {C.COPY_LOOKBACK_RUNS} runs",
        # The rule used to reach over all of history, and the sentences saying
        # so read as reassurance rather than as a specification - which is
        # exactly how they would survive a change to the constant. Anything
        # claiming the check covers everything ever admitted now contradicts it.
        r"against every submission already admitted"
        r"|every submission ever (admitted|made)"
        r"|compares you to every submission"
        r"|anti-copy check against all",
    ),
    (
        f"the serving reservation is {C.SERVING_RESERVED_GIB:.0f} GiB",
        r"\b(fixed )?\*{0,2}20 GiB\*{0,2}\b|0\.4168",
    ),
    (f"the served context is {C.SERVING_MAX_MODEL_LEN}", r"max_model_len: (?!8192\b)\d+"),
    (
        f"serving is batched at {C.SANDBOX_BATCH_CONCURRENCY}",
        r"max-num-seqs 1\b|one sequence at a time",
    ),
    (
        f"the validator floor is {C.MIN_VALIDATOR_CARDS} cards",
        r"\*\*8 × RTX 5090 \(32 GB\)\*\*, one candidate",
    ),
    (
        "the only reference is the base model",
        r"single_adapter_rotation` \| `|best single adapter, the standard merges",
    ),
    # The lookbehind is the point: a sentence saying there is *no* peak-VRAM
    # gate is the documentation agreeing with the constant, and flagging it made
    # this check fail on a correct doc. HISTORICAL cannot cover it - that is
    # deliberately past-tense only, and widening it to catch present-tense
    # denials would hide sentences this suite exists to find.
    (
        "peak VRAM is neither gated nor scored",
        r"peak_vram|(?<!no )peak-VRAM gate",
    ),
    (
        f"a run is {C.DEFAULT_RUN_BLOCKS} blocks, one day",
        r"run_blocks:? ?`?21600|~?72 ?h(ours|-hour)?\b|3-day run|three-day run",
    ),
    (
        f"weights lag the measurement by {C.WEIGHT_LAG_RUNS} run",
        r"weights? (are|is) set in the (same|measuring) run",
    ),
    (
        f"an on-chain recipe is at most {C.MAX_ONCHAIN_RECIPE_BYTES:,} canonical bytes",
        # Any other figure offered as *the* on-chain recipe limit. A miner who
        # builds to a stale number finds out when the seal will not fit, which
        # is after they have already decided what to submit.
        rf"on-chain (size )?limit is (?!{C.MAX_ONCHAIN_RECIPE_BYTES:,}\b)[\d,]+"
        rf"|at most \*{{0,2}}(?!{C.MAX_ONCHAIN_RECIPE_BYTES:,}\b)[\d,]+ canonical bytes",
    ),
    (
        f"the epoch commitment budget is {C.MAX_EPOCH_COMMIT_BYTES:,} bytes",
        rf"[\d,]+ bytes of commitments per epoch(?<!{C.MAX_EPOCH_COMMIT_BYTES:,} bytes of "
        rf"commitments per epoch)",
    ),
    (
        "recipes are submitted on chain, not to a service",
        # Inverted from what it checked for a day. The expensive stale claim is
        # now the opposite one: a miner told to POST a recipe loses runs in
        # silence, because there is no service to refuse them and the chain
        # never heard from them.
        #
        # Written without inline (?i) mid-pattern. Python 3.11 rejects a global
        # flag anywhere but position zero, and 3.10 accepts it - so the version
        # this file was written on was the only one where it compiled.
        r"(?:POST|post) (?:your |the )?recipe to|capcomp submit --confirm"
        r"|submissions? (?:API|service) is (?:the|where)",
    ),
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
        REPO / name for name in listed.stdout.split() if name.endswith((".md", ".yml", ".yaml"))
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
