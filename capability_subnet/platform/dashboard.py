"""The public dashboard.

Renders the engine's published state — champion, queue, recent reports,
compatibility history — as a single self-contained HTML page. No external
requests, no build step, no framework: the file can be copied anywhere, opened
from disk, and read without a server.

Everything shown here comes from the same store the read-only API serves, and
nothing is shown that the API does not already publish. The dashboard is a view,
not a second source of truth.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capability_subnet import __version__
from capability_subnet.backend.store import Store
from capability_subnet.platform.compatibility_graph import build_graph

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 2rem 1.5rem; max-width: 1100px; margin-inline: auto;
  background: Canvas; color: CanvasText;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2.25rem 0 .75rem; padding-bottom: .35rem;
     border-bottom: 1px solid color-mix(in srgb, CanvasText 15%, transparent); }
.sub { opacity: .65; margin: 0 0 1.5rem; font-size: .9rem; }
.cards { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
.card { border: 1px solid color-mix(in srgb, CanvasText 15%, transparent);
        border-radius: 10px; padding: .85rem 1rem; }
.card .label { font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; opacity: .6; }
.card .value { font-size: 1.35rem; font-weight: 600; margin-top: .2rem;
               font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; }
th, td { text-align: left; padding: .45rem .7rem; white-space: nowrap;
         border-bottom: 1px solid color-mix(in srgb, CanvasText 10%, transparent); }
th { font-weight: 600; opacity: .75; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code, .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .84em; }
.tag { display: inline-block; padding: .1rem .45rem; border-radius: 5px; font-size: .76rem;
       border: 1px solid currentColor; }
.tag.ok { color: #1a7f47; } .tag.bad { color: #b3261e; } .tag.idle { opacity: .6; }
@media (prefers-color-scheme: dark) { .tag.ok { color: #6ee7a8; } .tag.bad { color: #ff9d94; } }
.empty { opacity: .6; font-style: italic; }
footer { margin-top: 3rem; font-size: .8rem; opacity: .6; }
"""

_VERDICT_CLASS = {
    "dethrone": "ok",
    "held": "idle",
    "terminated": "bad",
    "rejected": "bad",
    "reference": "idle",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _short(digest: str | None, length: int = 12) -> str:
    if not digest:
        return "—"
    return digest.split(":", 1)[-1][:length]


def _card(label: str, value: Any) -> str:
    return f'<div class="card"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div></div>'


def _table(headers: list[str], rows: list[list[str]], *, empty: str) -> str:
    if not rows:
        return f'<p class="empty">{_e(empty)}</p>'
    head = "".join(f"<th>{_e(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render(store: Store, *, workflow_id: str, generated_at: str | None = None) -> str:
    """Build the dashboard page."""
    champion = store.get_champion()
    queue = store.list_queue()
    reports = store.list_reports(limit=25)
    weights = store.latest_weights()
    window_id = store.latest_window_id()
    stamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cards = [
        _card("workflow", workflow_id),
        _card("window", window_id if window_id is not None else "—"),
        _card(
            "champion",
            f"{champion.hotkey[:10]}…" if champion and champion.hotkey else "throne empty",
        ),
        _card("queued challengers", sum(1 for e in queue if e.status == "queued")),
        _card("terminated", sum(1 for e in queue if e.status == "terminated")),
    ]

    champion_rows: list[list[str]] = []
    if champion is not None:
        scores = champion.last_scores
        champion_rows.append(
            [
                f'<td class="mono">{_e(champion.candidate_id)}</td>',
                f'<td class="mono">{_e(_short(champion.recipe_sha256))}</td>',
                f'<td class="mono">{_e(_short(champion.artifact_sha256))}</td>',
                f'<td class="num">{champion.crowned_at_window}</td>',
                f'<td class="num">{scores.end_to_end:.3f}</td>' if scores else "<td>—</td>",
                f'<td class="num">{scores.qualified_score:.3f}</td>' if scores else "<td>—</td>",
            ]
        )

    queue_rows = [
        [
            f'<td class="mono">{_e(entry.hotkey[:16])}…</td>',
            f'<td class="num">{entry.uid}</td>',
            f'<td class="mono">{_e(_short(entry.recipe_sha256))}</td>',
            f'<td class="num">{entry.first_block}</td>',
            f'<td><span class="tag {_VERDICT_CLASS.get(entry.status, "idle")}">'
            f"{_e(entry.status)}</span></td>",
            f"<td>{_e(entry.status_reason[:90])}</td>",
        ]
        for entry in sorted(queue, key=lambda e: e.first_block)
    ]

    report_rows = [
        [
            f'<td class="mono">{_e(_short(digest))}</td>',
            f'<td class="mono">{_e(report.candidate_id[:22])}</td>',
            f'<td class="num">{report.window_id}</td>',
            f'<td class="num">{report.scores.end_to_end:.3f}</td>',
            f'<td class="num">{report.scores.qualified_score:.3f}</td>',
            f'<td><span class="tag {_VERDICT_CLASS.get(report.verdict, "idle")}">'
            f"{_e(report.verdict)}</span></td>",
            f"<td>{_e(report.verdict_reason[:90])}</td>",
        ]
        for digest, report in reports
    ]

    weight_rows = (
        [
            [
                f'<td class="num">{entry.uid}</td>',
                f'<td class="mono">{_e(entry.hotkey[:16] or "—")}</td>',
                f"<td>{_e(entry.role)}</td>",
                f'<td class="num">{entry.weight:.4f}</td>',
            ]
            for entry in weights.entries
        ]
        if weights
        else []
    )

    graph = build_graph(store.load_compatibility(limit=2000), metric="end_to_end")
    adapter_rows = [
        [
            f'<td class="mono">{_e(stats.adapter_id)}</td>',
            f'<td class="num">{stats.selections}</td>',
            f'<td class="num">{stats.mean_when_selected:.3f}</td>',
            f'<td class="num">{stats.marginal_contribution:+.4f}</td>',
        ]
        for stats in graph.ranked_adapters()
        if stats.selections >= 3
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capability Composition Subnet</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>Capability Composition Subnet</h1>
<p class="sub">Continuous champion-challenge evaluation · {_e(workflow_id)} · generated {
        _e(stamp)
    }</p>

<div class="cards">{"".join(cards)}</div>

<h2>Champion</h2>
{
        _table(
            ["candidate", "recipe", "artifact", "crowned in window", "end-to-end", "qualified"],
            champion_rows,
            empty="The throne is empty. No package has cleared the reference baselines yet, so the workflow share is burned.",
        )
    }

<h2>Latest weight vector</h2>
{
        _table(
            ["uid", "hotkey", "role", "weight"],
            weight_rows,
            empty="No weight vector has been published yet.",
        )
    }

<h2>Queue</h2>
{
        _table(
            ["hotkey", "uid", "recipe", "commit block", "status", "reason"],
            queue_rows,
            empty="No submissions have been admitted.",
        )
    }

<h2>Recent evaluations</h2>
{
        _table(
            ["report", "candidate", "window", "end-to-end", "qualified", "verdict", "reason"],
            report_rows,
            empty="No evaluations have been published.",
        )
    }

<h2>Adapter contribution</h2>
{
        _table(
            ["adapter", "times selected", "mean end-to-end", "marginal contribution"],
            adapter_rows,
            empty="Not enough evaluations yet to say anything about individual adapters.",
        )
    }

<footer>
Version {_e(__version__)}. Every number here is derived from published, signed
evaluation reports; nothing is shown that the read-only API does not serve.
</footer>
</body>
</html>
"""


def write(store: Store, destination: str | Path, *, workflow_id: str) -> Path:
    """Render and write the dashboard."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(store, workflow_id=workflow_id), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    from capability_subnet.backend.settings import load_settings

    parser = argparse.ArgumentParser(
        prog="capability-dashboard",
        description="Render the public dashboard from engine state.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="dashboard.html")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    store = Store(settings.database_path)
    try:
        target = write(store, args.out, workflow_id=settings.workflow_id)
    finally:
        store.close()

    print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
