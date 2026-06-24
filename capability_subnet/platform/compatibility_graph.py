"""The compatibility graph.

Every evaluation appends one row: which adapters were selected, with which
coefficients at which depth, under which merge method and rank, and what that
package achieved on each capability axis. One row says almost nothing. Thousands
of them, across many windows and many independent miners, answer questions that
no single experiment can:

* which adapter pairs reinforce each other and which interfere,
* which capabilities need a specialist at which depth,
* when trimming beats dropping, and at what density,
* how much rank a workflow actually needs before quality falls off,
* which adapters turn out to be unnecessary.

The analyses here are deliberately simple — co-selection frequency, conditional
means, marginal contribution. Anything more elaborate would be reading structure
into what is, at the start of a network's life, a small and heavily confounded
sample. These are the summaries that stay honest as the sample grows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PairStatistics:
    """How two adapters behave when selected together."""

    left: str
    right: str
    together: int = 0
    together_score: float = 0.0
    left_only: int = 0
    left_only_score: float = 0.0
    right_only: int = 0
    right_only_score: float = 0.0

    @property
    def mean_together(self) -> float:
        return self.together_score / self.together if self.together else 0.0

    @property
    def mean_apart(self) -> float:
        total = self.left_only + self.right_only
        if not total:
            return 0.0
        return (self.left_only_score + self.right_only_score) / total

    @property
    def interaction(self) -> float:
        """Difference between the pair together and either alone.

        Positive means the two transfer well together, negative means they
        interfere. It is a raw difference, not a controlled effect: recipes that
        select both adapters differ in other ways too, and with a small sample
        this number is a hint, not a conclusion.
        """
        return self.mean_together - self.mean_apart

    @property
    def confident(self) -> bool:
        """Whether both arms have enough observations to be worth reading."""
        return self.together >= 5 and (self.left_only + self.right_only) >= 5


@dataclass(slots=True)
class AdapterStatistics:
    """How one adapter behaves across every recipe that selected it."""

    adapter_id: str
    selections: int = 0
    total_score: float = 0.0
    absent: int = 0
    absent_score: float = 0.0
    stage_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    coefficient_sum: float = 0.0

    @property
    def mean_when_selected(self) -> float:
        return self.total_score / self.selections if self.selections else 0.0

    @property
    def mean_when_absent(self) -> float:
        return self.absent_score / self.absent if self.absent else 0.0

    @property
    def marginal_contribution(self) -> float:
        return self.mean_when_selected - self.mean_when_absent

    @property
    def mean_coefficient(self) -> float:
        return self.coefficient_sum / self.selections if self.selections else 0.0


@dataclass(slots=True)
class GraphSummary:
    """The whole analysis over a set of recorded evaluations."""

    records: int = 0
    adapters: dict[str, AdapterStatistics] = field(default_factory=dict)
    pairs: dict[tuple[str, str], PairStatistics] = field(default_factory=dict)
    by_method: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    by_rank: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    by_density: dict[float, list[float]] = field(default_factory=lambda: defaultdict(list))

    def best_pairs(self, limit: int = 10) -> list[PairStatistics]:
        confident = [pair for pair in self.pairs.values() if pair.confident]
        return sorted(confident, key=lambda pair: -pair.interaction)[:limit]

    def worst_pairs(self, limit: int = 10) -> list[PairStatistics]:
        confident = [pair for pair in self.pairs.values() if pair.confident]
        return sorted(confident, key=lambda pair: pair.interaction)[:limit]

    def ranked_adapters(self) -> list[AdapterStatistics]:
        return sorted(
            self.adapters.values(), key=lambda stats: -stats.marginal_contribution
        )

    def method_means(self) -> dict[str, float]:
        return {
            method: sum(scores) / len(scores)
            for method, scores in sorted(self.by_method.items())
            if scores
        }

    def rank_means(self) -> dict[int, float]:
        return {
            rank: sum(scores) / len(scores)
            for rank, scores in sorted(self.by_rank.items())
            if scores
        }

    def density_means(self) -> dict[float, float]:
        return {
            density: sum(scores) / len(scores)
            for density, scores in sorted(self.by_density.items())
            if scores
        }


def build_graph(
    records: Iterable[dict[str, Any]], *, metric: str = "qualified_score"
) -> GraphSummary:
    """Fold recorded evaluations into a summary.

    Args:
        records: rows as stored by the engine's compatibility history.
        metric: which score to analyse. ``qualified_score`` blends quality and
            efficiency; ``end_to_end`` isolates completion, which is usually what
            you want when asking whether two adapters work together.
    """
    summary = GraphSummary()
    rows = [row for row in records if metric in row and row.get("selected_adapters")]
    summary.records = len(rows)

    universe: set[str] = set()
    for row in rows:
        universe.update(row["selected_adapters"])

    for row in rows:
        score = float(row.get(metric) or 0.0)
        selected = set(row["selected_adapters"])
        weights = row.get("global_weights") or {}

        for adapter in universe:
            stats = summary.adapters.setdefault(adapter, AdapterStatistics(adapter))
            if adapter in selected:
                stats.selections += 1
                stats.total_score += score
                stats.coefficient_sum += float(weights.get(adapter, 1.0))
                for stage, value in (row.get("per_stage_means") or {}).items():
                    stats.stage_totals[stage] += float(value)
            else:
                stats.absent += 1
                stats.absent_score += score

        ordered = sorted(universe)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                pair = summary.pairs.setdefault(
                    (left, right), PairStatistics(left, right)
                )
                in_left = left in selected
                in_right = right in selected
                if in_left and in_right:
                    pair.together += 1
                    pair.together_score += score
                elif in_left:
                    pair.left_only += 1
                    pair.left_only_score += score
                elif in_right:
                    pair.right_only += 1
                    pair.right_only_score += score

        method = row.get("combination_type")
        if method:
            summary.by_method[method].append(score)

        rank = row.get("output_rank")
        if rank:
            summary.by_rank[int(rank)].append(score)

        density = row.get("density")
        if density is not None:
            summary.by_density[round(float(density), 2)].append(score)

    return summary


def rank_efficiency_frontier(summary: GraphSummary) -> list[tuple[int, float, float]]:
    """How much quality each output rank buys, and what it costs in size.

    Returns:
        ``(rank, mean_score, relative_size)`` sorted by rank. Size is relative to
        the largest rank observed, so the trade-off reads without knowing the
        base model's dimensions.
    """
    means = summary.rank_means()
    if not means:
        return []

    largest = max(means)
    return [(rank, score, rank / largest) for rank, score in sorted(means.items())]


def redundant_adapters(summary: GraphSummary, *, threshold: float = 0.005) -> list[str]:
    """Adapters whose presence does not measurably change the outcome.

    A near-zero marginal contribution is evidence of redundancy, not proof: an
    adapter can look useless because every recipe that selected it also selected
    a better one. The threshold is deliberately tight, and the result is a list
    to investigate rather than a list to remove.
    """
    return sorted(
        stats.adapter_id
        for stats in summary.adapters.values()
        if stats.selections >= 5
        and stats.absent >= 5
        and abs(stats.marginal_contribution) < threshold
    )


def render_summary(summary: GraphSummary) -> str:
    """A readable report."""
    if not summary.records:
        return "No evaluations recorded yet."

    lines = [f"compatibility history over {summary.records} evaluations", ""]

    lines.append("marginal contribution per adapter:")
    for stats in summary.ranked_adapters():
        if stats.selections < 3:
            continue
        lines.append(
            f"  {stats.adapter_id:<34} {stats.marginal_contribution:+.4f} "
            f"(selected {stats.selections}×, mean coefficient {stats.mean_coefficient:.2f})"
        )

    best = summary.best_pairs(5)
    if best:
        lines.extend(["", "pairs that transfer positively:"])
        for pair in best:
            lines.append(
                f"  {pair.left} + {pair.right}: {pair.interaction:+.4f} "
                f"({pair.together} together)"
            )

    worst = summary.worst_pairs(5)
    if worst:
        lines.extend(["", "pairs that appear to interfere:"])
        for pair in worst:
            lines.append(
                f"  {pair.left} + {pair.right}: {pair.interaction:+.4f} "
                f"({pair.together} together)"
            )

    methods = summary.method_means()
    if methods:
        lines.extend(["", "mean score by merge method:"])
        for method, value in sorted(methods.items(), key=lambda item: -item[1]):
            lines.append(f"  {method:<24} {value:.4f}")

    frontier = rank_efficiency_frontier(summary)
    if frontier:
        lines.extend(["", "quality against output rank:"])
        for rank, score, relative in frontier:
            lines.append(f"  rank {rank:>4}  {score:.4f}  ({relative:.0%} of the largest)")

    redundant = redundant_adapters(summary)
    if redundant:
        lines.extend(
            ["", "adapters with no measurable effect (worth investigating): "]
        )
        lines.append("  " + ", ".join(redundant))

    return "\n".join(lines)


def geometric_mean(values: list[float]) -> float:
    """Geometric mean, with a floor so one zero does not erase the rest."""
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(1e-6, value)) for value in values) / len(values))
