"""The pinned corpora this arena draws from.

* **logic** — 81,566 puzzles across ten families, each carrying a measured pass
  rate for a small model. That calibration is what item selection uses.
* **code** — competitive-programming problems with stdin/stdout test cases,
  scored by running the candidate's program. Exact match asks whether an answer
  looks right; execution asks whether the code works.

Both are public, so a run draws from ~3,193 logic items and ~2,944 code
problems that a miner can read. The defence is not corpus size: recipes are
scored on held-out draws per run, retention is probed separately, and closed
runs are re-scorable from published traces.

Each source is pinned to a revision, so a score can say what it was measured on.
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
#: excluded: an axis every package fails carries no information.
CODE_DIFFICULTIES: frozenset[str] = frozenset({"introductory", "interview"})

#: Logic families a run draws from, in a fixed order so two engines given the
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

#: Every axis a run can score, in a fixed order. Defined here rather than in
#: the package's ``__init__`` so the contract module can read it without a
#: circular import.
#:
#: Format compliance is its own axis. Losing the ability to follow an output
#: instruction is a different failure from answering wrongly, and folding it into
#: correctness would report it as "wrong" rather than "no longer obedient".
STAGES: tuple[str, ...] = (*LOGIC_FAMILIES, CODE_FAMILY, "format_compliance")

#: An instance exercises one axis, so an axis is scored only on the instances that
#: belong to it. The threshold is exactness: a partially right answer to a puzzle
#: is a wrong answer, and a program that fails one case has not solved the problem.
STAGE_THRESHOLDS: dict[str, float] = {stage: 1.0 for stage in STAGES}

#: Absolute floor per axis, below which a package is terminated outright.
#:
#: Only format compliance carries one. It is the axis a working package always
#: clears — wrapping the answer the way the prompt asked costs it nothing — so
#: falling under it means the package no longer produces usable output at all.
#: The equal-weight linear merge of twelve adapters scores 0.000 here: it emits
#: repeated fragments and whitespace, and there is no sense in ranking it.
#:
#: The capability axes carry no absolute floor, and that is a deliberate
#: statement rather than an omission. These are hard problems: the unmerged base
#: model answers under a fifth of them, and a floor set anywhere near "half of
#: this axis" is above what anything on this arena reaches, so it terminates
#: every package including the best one. Capability is judged comparatively
#: instead — a challenger must beat the strongest reference by a margin and
#: dominate it on the axes that matter — which is a real contest between miners
#: rather than a bar none of them can reach.
STAGE_FLOORS: dict[str, float] = {stage: 0.0 for stage in STAGES} | {"format_compliance": 0.5}


#: Cases retained per code problem, as a deterministic prefix.
#:
#: Bounds execution cost: the median problem carries 9 cases and the worst carries
#: 565, each a subprocess with a 20-second ceiling. Applied at load time so the
#: retained cases travel with the instance and a replay scores the same prefix.
MAX_CASES_PER_PROBLEM = 32

#: Share of a run's instances drawn from the code corpus. Execution is the
#: stronger signal and also the slower one, being a subprocess per test case.
CODE_FRACTION = 0.25
