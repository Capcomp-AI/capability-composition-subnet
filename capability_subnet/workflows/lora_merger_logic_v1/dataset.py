"""Loading the pinned corpora, and how a seed selects from them.

A generated workflow is a pure function of its seed, so an auditor regenerates
the exact problem from the seed alone. A fixed corpus offers less: the *items* are
public, and what the secret seed protects is only which of them a window draws.

Three things narrow that gap. Selection is stratified, so a window cannot be
dominated by whichever family a miner studied. A quarter of every window is scored
by **executing** the candidate's program, where recognising a memorised problem
does not help unless the code runs. And general capability is probed separately,
so a package tuned onto this corpus at the cost of everything else is caught by
the retention floor rather than the arena.

The audit path is intact either way: selection is deterministic in the seed and
every revision is pinned, so a replay draws exactly the items the candidate faced
and those items cannot change underneath.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from functools import lru_cache

from capability_subnet.workflows.lora_merger_logic_v1 import sources as S

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TestCase:
    """One stdin/stdout pair a submitted program must satisfy."""

    stdin: str
    expected_stdout: str


@dataclass(frozen=True, slots=True)
class CorpusItem:
    item_id: str
    family: str
    question: str
    #: Exact expected answer for a logic item; empty for a code item.
    answer: str = ""
    #: Test cases for a code item; empty for a logic item.
    cases: tuple[TestCase, ...] = ()
    difficulty: float = 0.0
    source: str = ""

    @property
    def is_code(self) -> bool:
        return bool(self.cases)


@lru_cache(maxsize=2)
def _logic_shards(source: S.Source) -> tuple[str, ...]:
    from huggingface_hub import HfApi, hf_hub_download

    files = [
        f
        for f in HfApi().list_repo_files(source.repo, repo_type="dataset", revision=source.revision)
        if f.startswith("data/") and f.endswith(".parquet")
    ]
    if not files:
        raise OSError(f"{source.repo}@{source.revision} exposes no parquet shards")
    return tuple(
        hf_hub_download(source.repo, f, repo_type="dataset", revision=source.revision)
        for f in sorted(files)
    )


@lru_cache(maxsize=1)
def load_logic() -> dict[str, tuple[CorpusItem, ...]]:
    """Banded logic items, grouped by family and sorted for determinism."""
    import pandas as pd

    source = S.LOGIC
    grouped: dict[str, list[CorpusItem]] = {family: [] for family in S.LOGIC_FAMILIES}

    for shard_number, shard in enumerate(_logic_shards(source)):
        frame = pd.read_parquet(shard)
        banded = frame[
            (frame[S.DIFFICULTY_COLUMN] >= S.DIFFICULTY_BAND[0])
            & (frame[S.DIFFICULTY_COLUMN] <= S.DIFFICULTY_BAND[1])
        ]
        for index, row in banded.iterrows():
            info = json.loads(row["info"])
            family = info.get("task")
            if family not in grouped:
                continue
            answer = str(json.loads(info["game_data_str"]).get("answer") or "").strip()
            if not answer:
                # Some families state the answer only as solver state. Without an
                # exact expected value there is nothing to compare, and a scorer
                # that guessed would be the model-judge this design avoids.
                continue
            grouped[family].append(
                CorpusItem(
                    # The shard number belongs in the id: the row index restarts
                    # at zero in every shard, so family+index alone collides the
                    # moment the source has more than one.
                    item_id=f"{family}-{shard_number}-{index}",
                    family=family,
                    question=row["question"],
                    answer=answer,
                    difficulty=float(row[S.DIFFICULTY_COLUMN]),
                    source=source.name,
                )
            )

    corpus = {
        family: tuple(sorted(items, key=lambda i: i.item_id))
        for family, items in grouped.items()
        if items
    }
    log.info(
        "logic corpus %s: %d items across %d families",
        source.repo,
        sum(len(v) for v in corpus.values()),
        len(corpus),
    )
    return corpus


@lru_cache(maxsize=1)
def load_code() -> tuple[CorpusItem, ...]:
    """Execution-verified programs, sorted for determinism.

    Only the first shards are read. The corpus runs to tens of thousands of
    problems and a window samples a few hundred; loading all of it would cost
    minutes of every engine start for items no window will draw.
    """
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f
        for f in HfApi().list_repo_files(S.CODE.repo, repo_type="dataset", revision=S.CODE.revision)
        if f.endswith(".json")
    )[:8]

    def as_mapping(value) -> dict:
        """Some rows carry these fields as JSON text rather than as objects.

        Tolerated rather than assumed away: the corpus is upstream data and a
        loader that crashed on a shape variation would take the arena down for a
        reason that has nothing to do with the protocol.
        """
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                try:
                    import ast

                    parsed = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    items: list[CorpusItem] = []
    for name in files:
        path = hf_hub_download(S.CODE.repo, name, repo_type="dataset", revision=S.CODE.revision)
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows:
            if row.get("task_type") != "verifiable_code":
                continue
            if as_mapping(row.get("metadata")).get("difficulty") not in S.CODE_DIFFICULTIES:
                continue
            raw_cases = as_mapping(row.get("verification_info")).get("test_cases") or []
            cases = tuple(
                TestCase(str(c.get("input", "")), str(c.get("output", "")))
                for c in (as_mapping(x) for x in raw_cases)
                if c.get("type") == "stdin_stdout"
            )[: S.MAX_CASES_PER_PROBLEM]
            if not cases or not _has_a_hidden_case(row["prompt"], cases):
                continue
            items.append(
                CorpusItem(
                    item_id=f"code-{row.get('problem_id') or row.get('in_source_id')}",
                    family=S.CODE_FAMILY,
                    question=row["prompt"],
                    cases=cases,
                    source=S.CODE.name,
                )
            )

    log.info("code corpus %s: %d problems", S.CODE.repo, len(items))
    return tuple(sorted(items, key=lambda i: i.item_id))


def _has_a_hidden_case(prompt: str, cases: tuple[TestCase, ...]) -> bool:
    """Whether solving this problem requires more than reading the prompt.

    Competitive-programming statements print a worked example, and the corpus
    usually keeps that example as the first test case — harmless when other cases
    follow, because passing requires all of them.

    It is not harmless when it is the *only* case. Then the expected output is
    printed in the question, and a program that ignores its input and prints that
    constant passes. Measured on the admitted pool, roughly a quarter of the code
    problems were in that state: free marks for any package including the base
    model, on the axis whose whole claim is that execution is the stronger signal.

    A problem is admitted only if at least one retained case cannot be answered
    from the statement.
    """
    return any(
        case.expected_stdout.strip() and case.expected_stdout.strip() not in prompt
        for case in cases
    )


def select(seed: int) -> CorpusItem:
    """The item this seed denotes.

    The seed decides the corpus first, then the family, then the item. Deciding
    the corpus first is what makes the code share a stable fraction of a window
    rather than an accident of how many logic families happen to be loaded.
    """
    rng = random.Random(seed)

    if rng.random() < S.CODE_FRACTION:
        code = load_code()
        if code:
            return code[rng.randrange(len(code))]
        log.warning("code corpus is empty; this instance falls back to logic")

    corpus = load_logic()
    families = tuple(f for f in S.LOGIC_FAMILIES if f in corpus)
    if not families:
        raise RuntimeError("no logic families are available")
    family = families[rng.randrange(len(families))]
    items = corpus[family]
    return items[rng.randrange(len(items))]
