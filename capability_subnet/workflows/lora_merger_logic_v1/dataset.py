"""The pinned corpus, and how a seed selects from it.

Instance generation in the V1 workflow is a pure function of the seed — an
auditor regenerates the exact problem a candidate faced from the seed alone.
A fixed corpus cannot offer quite that, and the difference is worth stating
rather than glossing: the *items* are public, so a miner can see every one of
them, and what the secret seed protects is only which items a window draws.

That is a materially weaker anti-overfitting guarantee than a generator, and it
is the price of a corpus whose difficulty is already calibrated. It is bought
back in two ways: the corpus is large enough that memorising it is not the
cheap attack it sounds like, and selection is stratified across task families
so a window cannot be dominated by whichever family a miner happened to tune on.

Everything else the audit design needs is intact. Selection is deterministic in
the seed, so an auditor replaying a closed window draws exactly the items the
candidate saw, and the revision is pinned so those items cannot change under it.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)

#: Upstream corpus this arena draws its problems from.
#:
#: A data source, recorded the way every adapter in the registry records its
#: `source_repo`: it is the identifier the file is fetched by, and a score that
#: cannot say what it was measured on is not reproducible. The arena itself —
#: the harness, the axes, the scoring, the contract, what counts as an answer —
#: belongs to this subnet.
DATASET = "AffineFoundation/affine-lgc"
#: Pinned, for the same reason the base model and every adapter are pinned: a
#: score has to mean the same thing next month.
REVISION = "19765edac477"
PARQUET = "data/train-00000-of-00001.parquet"

#: Column carrying a measured pass rate for a 4B model, used to select items
#: that discriminate rather than items that are uniformly trivial or hopeless.
DIFFICULTY_COLUMN = "avg@16_qwen3_4b_instruct_2507"

#: Difficulty band. Chosen on the corpus's own measurement rather than a guess.
#:
#: Note what this label is and is not. It is pass@16 with sampling; this engine
#: scores pass@1, greedy, with the model's reasoning channel disabled. Those are
#: very different regimes, and the band therefore overstates what the engine
#: will observe — a 0.2-0.8 band measured at pass@16 produced roughly 0.10 at
#: pass@1 in the first run. It still selects for *discrimination*, which is what
#: it is for.
BAND = (0.20, 0.80)

#: Families a window draws from, largest first. Fixed rather than discovered, so
#: two engines on the same seed draw the same instances.
TASK_FAMILIES: tuple[str, ...] = (
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


@dataclass(frozen=True, slots=True)
class CorpusItem:
    item_id: str
    task: str
    question: str
    answer: str
    difficulty: float


@lru_cache(maxsize=1)
def load_corpus() -> dict[str, tuple[CorpusItem, ...]]:
    """The banded corpus, grouped by task family and sorted for determinism.

    Cached because it is tens of megabytes and every window reads it. Sorted by
    item id inside each family so the ordering a seed indexes into does not
    depend on how the parquet happened to be written.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=REVISION)
    frame = pd.read_parquet(path)
    banded = frame[(frame[DIFFICULTY_COLUMN] >= BAND[0]) & (frame[DIFFICULTY_COLUMN] <= BAND[1])]

    grouped: dict[str, list[CorpusItem]] = {family: [] for family in TASK_FAMILIES}
    for index, row in banded.iterrows():
        info = json.loads(row["info"])
        family = info.get("task")
        if family not in grouped:
            continue
        answer = str(json.loads(info["game_data_str"]).get("answer") or "").strip()
        if not answer:
            # Some families state the answer only as a solver state. Without an
            # exact expected value there is nothing to compare against, and a
            # scorer that guessed would be the model-judge this design avoids.
            continue
        grouped[family].append(
            CorpusItem(
                f"{family}-{index}", family, row["question"], answer, float(row[DIFFICULTY_COLUMN])
            )
        )

    corpus = {
        family: tuple(sorted(items, key=lambda i: i.item_id))
        for family, items in grouped.items()
        if items
    }
    log.info(
        "logic corpus: %d items across %d families",
        sum(len(v) for v in corpus.values()),
        len(corpus),
    )
    return corpus


def select(seed: int) -> CorpusItem:
    """The item this seed selects. A pure function of the seed and the pin.

    Stratified: the seed picks a family first and an item within it second, so a
    window's draw is spread across families rather than landing wherever the
    corpus happens to be dense.
    """
    corpus = load_corpus()
    families = tuple(family for family in TASK_FAMILIES if family in corpus)
    if not families:
        raise RuntimeError(f"{DATASET}@{REVISION} yielded no usable items")

    rng = random.Random(seed)
    family = families[rng.randrange(len(families))]
    items = corpus[family]
    return items[rng.randrange(len(items))]
