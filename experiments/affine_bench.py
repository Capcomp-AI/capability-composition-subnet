"""Single adapters versus merged adapters, per task, on Affine's logic corpus.

The question this answers is the one the subnet exists to ask: does composing
adapters beat picking the best single one, and beat the standard merges. The
German maintenance workflow could not answer it — its oracle needs ten of twelve
turns and the pool has no German or SQL adapter, so a null result there would
say more about calibration than about composition.

This corpus does not have that problem. Every item carries
`avg@16_qwen3_4b_instruct_2507`, a measured pass rate for a 4B model, so items
can be selected where a model of this class actually discriminates. Every item
states its own output format and carries an exact answer, so scoring is string
comparison rather than judgement.

Design:

* **Paired.** Every package sees the identical item set, so differences are not
  a difference in draw. Same property the engine's comparator relies on.
* **Stratified and seeded.** Items are drawn per task family from a fixed seed,
  so one family cannot dominate and the sample is reproducible.
* **Banded.** Only items a 4B model solves between 20% and 80% of the time. Too
  easy and every package ties; too hard and every package scores zero.
* **Exact.** The answer is extracted from the wrapper the prompt itself demanded
  and compared literally. No model judges anything.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
import pandas as pd
from huggingface_hub import hf_hub_download

DATASET = "AffineFoundation/affine-lgc"
REVISION = "19765edac477"          # pinned, like every other input to this subnet
SAMPLE_SEED = 20260729
BAND = (0.20, 0.80)
DIFFICULTY = "avg@16_qwen3_4b_instruct_2507"


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    task: str
    question: str
    answer: str
    difficulty: float


def load_items(per_task: int, max_tasks: int) -> list[Item]:
    """Draw a stratified, banded, reproducible sample."""
    path = hf_hub_download(
        DATASET, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION
    )
    frame = pd.read_parquet(path)
    band = frame[(frame[DIFFICULTY] >= BAND[0]) & (frame[DIFFICULTY] <= BAND[1])]

    by_task: dict[str, list[Item]] = defaultdict(list)
    for index, row in band.iterrows():
        info = json.loads(row["info"])
        game = json.loads(info["game_data_str"])
        answer = str(game.get("answer") or "").strip()
        if not answer:
            continue
        by_task[info.get("task", "unknown")].append(
            Item(f"{info.get('task')}-{index}", info.get("task", "unknown"),
                 row["question"], answer, float(row[DIFFICULTY]))
        )

    # Largest families first so the sample is dominated by well-populated tasks,
    # then a fixed shuffle inside each so the choice is not "whatever came first".
    rng = random.Random(SAMPLE_SEED)
    chosen: list[Item] = []
    for _task, items in sorted(by_task.items(), key=lambda kv: -len(kv[1]))[:max_tasks]:
        pool = sorted(items, key=lambda i: i.item_id)
        rng.shuffle(pool)
        chosen.extend(pool[:per_task])
    chosen.sort(key=lambda i: i.item_id)
    return chosen


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------
# Each prompt states the wrapper it wants: a fenced block, \boxed{}, or double
# square brackets. Extraction tries them in order and falls back to the whole
# reply, so a package that answered correctly without the wrapper is not marked
# wrong for a formatting slip — the comparison is about capability, and format
# compliance is measured separately.

_FENCE = re.compile(r"```(?:python|json)?\s*(.*?)```", re.S)
_BOXED = re.compile(r"\\boxed\{(.*?)\}", re.S)
_BRACKETS = re.compile(r"\[\[(.*?)\]\]", re.S)


def _normalise(text: str) -> str:
    """Whitespace-insensitive, quote-insensitive comparison form."""
    text = text.strip().strip("`").strip()
    text = re.sub(r"\s+", "", text)
    return text.replace("'", '"').casefold()


def extract(reply: str) -> list[str]:
    """Every plausible answer span in a reply, best guess first."""
    if not reply:
        return []
    spans: list[str] = []
    for pattern in (_FENCE, _BOXED, _BRACKETS):
        found = pattern.findall(reply)
        if found:
            spans.append(found[-1])  # the answer is stated last
    spans.append(reply)
    return spans


def is_correct(reply: str, answer: str) -> bool:
    target = _normalise(answer)
    if not target:
        return False
    for span in extract(reply):
        candidate = _normalise(span)
        if candidate == target:
            return True
        # `[[x]]` and `\boxed{x}` answers are often quoted inside their wrapper.
        if candidate.strip('"[]') == target.strip('"[]'):
            return True
    return False


# --------------------------------------------------------------------------
# Running one package
# --------------------------------------------------------------------------


@dataclass
class PackageResult:
    label: str
    kind: str
    correct: int = 0
    attempted: int = 0
    failed_requests: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    per_task: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    @property
    def score(self) -> float:
        return self.correct / self.attempted if self.attempted else 0.0


async def _ask(client: httpx.AsyncClient, url: str, model: str, item: Item,
               semaphore: asyncio.Semaphore, max_tokens: int) -> tuple[Item, str | None, int]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": item.question}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": abs(hash(item.item_id)) % (2**31),
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.post(f"{url}/v1/chat/completions", json=body, timeout=600.0)
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]["message"]
                usage = payload.get("usage") or {}
                return item, choice.get("content") or "", int(usage.get("completion_tokens", 0))
            except Exception:
                if attempt == 2:
                    return item, None, 0
                await asyncio.sleep(2.0 * (attempt + 1))
    return item, None, 0


async def run_package(url: str, model: str, label: str, kind: str, items: list[Item],
                      concurrency: int, max_tokens: int) -> PackageResult:
    result = PackageResult(label=label, kind=kind)
    started = time.time()
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        answers = await asyncio.gather(
            *(_ask(client, url, model, item, semaphore, max_tokens) for item in items)
        )
    for item, reply, tokens in answers:
        if reply is None:
            result.failed_requests += 1
            continue
        result.attempted += 1
        result.output_tokens += tokens
        ok = is_correct(reply, item.answer)
        result.correct += ok
        counts = result.per_task[item.task]
        counts[0] += ok
        counts[1] += 1
    result.seconds = time.time() - started
    return result


def main() -> int:
    from capability_subnet.backend.executor.serving import ManagedVllmServer

    base_model, pool_dir, packages_dir, out_path, vllm_python = sys.argv[1:6]
    per_task = int(sys.argv[6]) if len(sys.argv) > 6 else 25
    gpu = int(sys.argv[7]) if len(sys.argv) > 7 else 1

    import pathlib

    items = load_items(per_task=per_task, max_tasks=10)
    families = sorted({i.task for i in items})
    print(f"{len(items)} items across {len(families)} task families", flush=True)
    print(f"  {', '.join(families)}\n", flush=True)

    manifest = json.loads((pathlib.Path(packages_dir) / "manifest.json").read_text())
    targets: list[tuple[str, str | None, str]] = [("base_model", None, "reference")]
    targets += [(f"single:{p.name}", str(p), "single_adapter")
                for p in sorted(pathlib.Path(pool_dir).iterdir()) if p.is_dir()]
    targets += [(name, manifest[name]["path"], "merge") for name in sorted(manifest)]

    out = pathlib.Path(out_path)
    results = json.loads(out.read_text()) if out.exists() else {}

    for label, adapter, kind in targets:
        if label in results:
            print(f"  {label:34s} (done)", flush=True)
            continue
        server = ManagedVllmServer(
            base_model_path=base_model, model_name="candidate", port=8100 + gpu,
            gpu_index=gpu, python_executable=vllm_python, gpu_memory_utilization=0.92,
            startup_timeout=1500.0,
            extra_args=("--kv-cache-dtype", "fp8", "--max-num-seqs", "32"),
        )
        try:
            with server.serve(adapter) as handle:
                res = asyncio.run(run_package(
                    handle.base_url, handle.model_name, label, kind, items,
                    concurrency=32, max_tokens=1024))
        except Exception as exc:  # noqa: BLE001 - one bad package must not end the run
            results[label] = {"error": str(exc)[:300], "kind": kind}
            print(f"  {label:34s} FAILED: {str(exc)[:110]}", flush=True)
            out.write_text(json.dumps(results, indent=1))
            continue

        results[label] = {
            "kind": kind, "score": round(res.score, 4), "correct": res.correct,
            "attempted": res.attempted, "failed_requests": res.failed_requests,
            "output_tokens": res.output_tokens, "minutes": round(res.seconds / 60, 1),
            "per_task": {t: c for t, c in sorted(res.per_task.items())},
        }
        print(f"  {label:34s} {res.score:.3f}  ({res.correct}/{res.attempted})  "
              f"{res.output_tokens:>7,} tok  {res.seconds/60:.1f} min", flush=True)
        out.write_text(json.dumps(results, indent=1))

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
