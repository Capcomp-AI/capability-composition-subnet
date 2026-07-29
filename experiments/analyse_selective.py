"""The hypothesis, tested directly.

"LoRA A, B, C are each good at A, B, C. Merged A+B+C should beat A on tasks B
and C, beat B on tasks A and C, and so on."

So the comparison is not merge-versus-the-best-adapter. It is merge versus *each
constituent*, on the tasks that constituent is not the specialist for. A merge
earns its place if committing to it beats committing to any one of the adapters
inside it — which is the choice a deployer actually faces, since routing to the
right specialist per request is a different product with different costs.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

MEMBERS = [
    "action-planner-v1",
    "code-generation-v1",
    "constrained-selection-v1",
    "creative-writing-v1",
    "legal-citation-v1",
]

results = json.loads(pathlib.Path(sys.argv[1]).read_text())
rows = {k: v for k, v in results.items() if "score" in v}
base = rows["base_model"]
tasks = sorted(base["per_task"])


def sc(row, task):
    got, total = row["per_task"].get(task, [0, 1])
    return got / max(1, total)


def wilson(correct, total):
    if not total:
        return 0.0, 0.0
    z, p, n = 1.96, correct / total, total
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


for merge_name in ("selective_ties", "selective_linear"):
    if merge_name not in rows:
        print(f"({merge_name} not scored yet)\n")
        continue
    merge = rows[merge_name]
    lo, hi = wilson(merge["correct"], merge["attempted"])
    bl, bh = wilson(base["correct"], base["attempted"])

    print("=" * 78)
    print(f"{merge_name}  —  merge of {len(MEMBERS)} individually-certified adapters".center(78))
    print("=" * 78)
    print(
        f"  merge      {merge['score']:.3f}  [{lo:.3f}, {hi:.3f}]   {merge['output_tokens']:>7,} tokens"
    )
    print(f"  base       {base['score']:.3f}  [{bl:.3f}, {bh:.3f}]")

    print("\n  versus each constituent, on that adapter's OWN task vs its AWAY tasks:")
    print(f"  {'constituent':26s} {'home task':22s} {'home':>14s} {'away':>14s}")
    away_wins = home_wins = 0
    for member in MEMBERS:
        key = f"single:{member}"
        if key not in rows:
            continue
        single = rows[key]
        lifts = {t: sc(single, t) - sc(base, t) for t in tasks}
        home = max(lifts, key=lifts.get)
        away = [t for t in tasks if t != home]

        m_home, s_home = sc(merge, home), sc(single, home)
        m_away = sum(sc(merge, t) for t in away) / len(away)
        s_away = sum(sc(single, t) for t in away) / len(away)
        home_wins += m_home >= s_home
        away_wins += m_away > s_away
        print(
            f"  {member:26s} {home:22s} "
            f"{m_home:5.2f} vs {s_home:5.2f}  {m_away:5.3f} vs {s_away:5.3f}"
            f"   {'AWAY-WIN' if m_away > s_away else ''}"
        )

    print(f"\n  merge broader on {away_wins}/{len(MEMBERS)} constituents (away tasks)")
    print(f"  merge holds its own on {home_wins}/{len(MEMBERS)} home tasks")

    verdict = (
        "SUPPORTS the hypothesis"
        if away_wins > len(MEMBERS) / 2
        else "does NOT support the hypothesis"
    )
    print(f"\n  -> {verdict}")

    print("\n  per task:")
    print(f"  {'task':24s} {'base':>6s} {'merge':>7s} {'best member':>13s}  {'':>10s}")
    for t in tasks:
        best = max(
            ((m, sc(rows[f"single:{m}"], t)) for m in MEMBERS if f"single:{m}" in rows),
            key=lambda kv: kv[1],
            default=("", 0.0),
        )
        flag = ""
        if sc(merge, t) > best[1]:
            flag = "MERGE BEATS EVERY MEMBER"
        elif sc(base, t) == 0 and sc(merge, t) > 0:
            flag = "lifted off a zero floor"
        print(f"  {t:24s} {sc(base, t):6.2f} {sc(merge, t):7.2f} {best[1]:13.2f}  {flag}")
    print()
