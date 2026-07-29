"""The corpora this arena draws from, and what each one buys.

Two pinned sources, because they are not interchangeable:

* **logic** — 81,566 puzzles across ten families, each carrying a measured pass
  rate for a small model. The calibration is what makes item selection
  defensible rather than arbitrary.
* **code** — competitive-programming problems with stdin/stdout test cases,
  scored by **running the candidate's program**. A stronger guarantee than
  string comparison: exact match asks whether the answer looks right, execution
  asks whether the code works.

A third source, ``affine-lgc-xlarge``, was measured and deliberately **not**
used. It advertises 1,081,566 rows against this corpus's 81,566, and a larger
pool would be a real defence — the corpora are public, so a miner can read every
item, and scale is the only honest mitigation for that. But the difficulty column
is populated on 8% of its rows, all inside its first shard, and banding that
shard yields exactly the 3,193 usable items this corpus already yields: its
labelled subset *is* this corpus. The extra million rows cannot be banded, and an
unbanded item is one every package fails or every package passes. Preferring it
bought four shards of download and no additional problems.

That leaves the memorisation exposure stated plainly rather than argued away: a
window draws from ~3,193 logic items and ~3,920 code problems, both public. The
defence is not corpus size. It is that recipes are scored on held-out draws per
window, retention is probed separately, and closed windows are re-scorable from
published traces.

Each source is pinned to a revision. A score that cannot say what it was measured
on is not reproducible, and the arena — its harness, its axes, its scoring, its
contract — belongs to this subnet whatever the problems are sourced from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    """One pinned upstream corpus."""

    name: str
    repo: str
    revision: str
    kind: str  # "logic" (exact match) or "code" (execution)
    detail: str


#: Exact-match logic puzzles with a measured difficulty column.
LOGIC = Source(
    name="logic",
    repo="AffineFoundation/affine-lgc",
    revision="19765edac477",
    kind="logic",
    detail="81,566 puzzles, ten families, per-item measured pass rate",
)

#: Execution-verified programs.
CODE = Source(
    name="code",
    repo="AffineFoundation/rl-python",
    revision="0cc711b1f059",
    kind="code",
    detail="competitive programming with stdin/stdout cases, scored by execution",
)

SOURCES: tuple[Source, ...] = (LOGIC, CODE)

#: Column carrying a measured pass rate for a 4B model on the logic corpora.
#:
#: Read the label carefully: it is pass@16 with sampling, while this engine
#: scores pass@1, greedy, with the reasoning channel disabled. The band selects
#: for *discrimination*, and absolute scores land far below it.
DIFFICULTY_COLUMN = "avg@16_qwen3_4b_instruct_2507"
DIFFICULTY_BAND = (0.20, 0.80)

#: Problem difficulties admitted from the code corpus. The hardest tier is
#: excluded: a family every package fails carries no information, and the first
#: measured run of this pool put a merged package near 0.10 on the logic corpus.
CODE_DIFFICULTIES: frozenset[str] = frozenset({"introductory", "interview"})

#: Logic families a window draws from, in a fixed order so two engines given the
#: same seed draw the same instance.
LOGIC_FAMILIES: tuple[str, ...] = (
    "word_sorting",
    "goods_exchange",
    "object_counting",
    "cipher",
    "word_sorting_mistake",
    "zebra_puzzle",
    "web_of_lies",
    "arrow_maze",
    "time_sequence",
    "boolean_expressions",
)

#: The axis execution-verified programs are scored on.
CODE_FAMILY = "code_execution"

#: Every axis a window can score, in a fixed order.
#:
#: Defined here rather than in the package's ``__init__`` because the contract
#: needs it too, and importing the package from its own contract module would be
#: circular. One definition, two readers.
#:
#: Compliance is separated on purpose. A merged package losing the ability to
#: follow an output instruction is the most common way merging goes wrong — the
#: first run of this corpus had a linear merge answering terse instructions with
#: prose and burning five times the tokens to do it — and folding that into
#: correctness would report it as "wrong" rather than "no longer obedient".
STAGES: tuple[str, ...] = (*LOGIC_FAMILIES, CODE_FAMILY, "format_compliance")

#: An instance exercises one axis, so an axis is scored only on the instances that
#: belong to it. The threshold is exactness: a partially right answer to a puzzle
#: is a wrong answer, and a program that fails one case has not solved the problem.
STAGE_THRESHOLDS: dict[str, float] = {stage: 1.0 for stage in STAGES}


#: Cases retained per code problem, as a deterministic prefix.
#:
#: The corpus is unbounded here in a way that matters: the median problem carries
#: 9 cases but the worst carries 565, and at a 20-second ceiling per case one
#: pathological instance would occupy a validator for three hours against a
#: budget of fifteen seconds. Capping bounds that at a level well past the median,
#: and the cap is applied at load time so the retained cases travel with the
#: instance — an auditor replaying a closed window scores the same cases rather
#: than reconstructing which prefix was used.
MAX_CASES_PER_PROBLEM = 32

#: Share of a window's instances drawn from the code corpus.
#:
#: Execution is the stronger signal, so it is deliberately not a tenth of the
#: board. It is also slower — every instance runs a subprocess per test case —
#: so it is not the whole board either.
CODE_FRACTION = 0.25
