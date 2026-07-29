"""Single adapters versus merges, per task. The comparison the subnet exists to make."""

from __future__ import annotations

import json
import math
import pathlib
import sys

results = json.loads(pathlib.Path(sys.argv[1]).read_text())
rows = {k: v for k, v in results.items() if "score" in v}
if not rows:
    raise SystemExit("no scored packages yet")

base = rows.get("base_model")
singles = {k: v for k, v in rows.items() if v["kind"] == "single_adapter"}
merges = {k: v for k, v in rows.items() if v["kind"] == "merge"}
tasks = sorted(base["per_task"]) if base else []


def wilson(correct: int, total: int) -> tuple[float, float]:
    """95% Wilson interval — honest about 25-item cells, unlike a normal approximation."""
    if total == 0:
        return 0.0, 0.0
    z, p, n = 1.96, correct / total, total
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


print("=" * 78)
print("OVERALL".center(78))
print("=" * 78)
print(f"{'package':34s} {'score':>7s} {'95% CI':>16s} {'tokens':>10s}  kind")
for label, row in sorted(rows.items(), key=lambda kv: -kv[1]["score"]):
    lo, hi = wilson(row["correct"], row["attempted"])
    mark = "  <-- base" if label == "base_model" else ""
    print(f"{label:34s} {row['score']:7.3f} [{lo:.3f}, {hi:.3f}] {row['output_tokens']:10,}  "
          f"{row['kind']}{mark}")

best_single = max(singles.items(), key=lambda kv: kv[1]["score"], default=(None, None))
best_merge = max(merges.items(), key=lambda kv: kv[1]["score"], default=(None, None))

print()
print("=" * 78)
print("THE QUESTION: does composition beat the best single adapter?".center(78))
print("=" * 78)
if base:
    print(f"  base model            {base['score']:.3f}")
if best_single[0]:
    print(f"  best single adapter   {best_single[1]['score']:.3f}   ({best_single[0]})")
if best_merge[0]:
    print(f"  best merge            {best_merge[1]['score']:.3f}   ({best_merge[0]})")
if best_single[0] and best_merge[0]:
    delta = best_merge[1]["score"] - best_single[1]["score"]
    verdict = "COMPOSITION WINS" if delta > 0 else "COMPOSITION LOSES"
    print(f"\n  {verdict}: {delta:+.3f} on {base['attempted']} paired items")

print()
print("=" * 78)
print("PER TASK — best single vs best merge".center(78))
print("=" * 78)
print(f"{'task':24s} {'base':>6s} {'best single':>26s} {'best merge':>24s}   verdict")
wins = losses = ties = 0
for task in tasks:
    b = base["per_task"][task]
    bs = max(((k, v["per_task"][task]) for k, v in singles.items() if task in v["per_task"]),
             key=lambda kv: kv[1][0] / max(1, kv[1][1]), default=(None, [0, 0]))
    bm = max(((k, v["per_task"][task]) for k, v in merges.items() if task in v["per_task"]),
             key=lambda kv: kv[1][0] / max(1, kv[1][1]), default=(None, [0, 0]))
    bs_score = bs[1][0] / max(1, bs[1][1])
    bm_score = bm[1][0] / max(1, bm[1][1])
    verdict = "merge" if bm_score > bs_score else ("single" if bs_score > bm_score else "tie")
    wins += verdict == "merge"
    losses += verdict == "single"
    ties += verdict == "tie"
    print(f"{task:24s} {b[0]/max(1,b[1]):6.2f} "
          f"{bs_score:6.2f} {(bs[0] or '')[:18]:>19s} "
          f"{bm_score:6.2f} {(bm[0] or '')[:16]:>17s}   {verdict}")

print(f"\n  merge better on {wins} task(s), single better on {losses}, tied on {ties}")

lifted = [t for t in tasks
          if base["per_task"][t][0] == 0
          and any(v["per_task"].get(t, [0, 1])[0] > 0 for v in merges.values())]
if lifted:
    print(f"\n  composition produced a capability the base model has none of: {lifted}")
