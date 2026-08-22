"""
Search effort by algorithm/heuristic.

Merges what would otherwise be 3 separate bar charts (expanded nodes,
elapsed time, frontier size) plus operations_done into ONE file, using a
dropdown to switch the metric — they all share the same chart shape
(one bar per config, averaged across levels), so a filter is a natural fit.

Only successful runs are averaged in, since effort metrics from failed/aborted
searches aren't comparable (e.g. a search that errored out early looks
artificially "cheap").
"""

from __future__ import annotations
import sys
from pathlib import Path

import plotly.graph_objects as go

from .common import load_data, successful, config_order, config_color

METRICS = {
    "expanded_nodes": "Expanded Nodes (avg)",
    "elapsed_seconds": "Elapsed Time (s, avg)",
    "frontier_nodes": "Frontier Size (avg)",
    "operations_done": "Operations Done (avg)",
}


def build(df) -> go.Figure:
    ok = successful(df)
    configs = config_order(ok)

    means = ok.groupby("config")[list(METRICS.keys())].mean().reindex(configs)
    counts = ok.groupby("config").size().reindex(configs).fillna(0).astype(int)

    fig = go.Figure()

    metric_keys = list(METRICS.keys())
    for i, metric in enumerate(metric_keys):
        fig.add_trace(
            go.Bar(
                x=configs,
                y=means[metric],
                marker_color=[config_color(c) for c in configs],
                customdata=counts.values,
                hovertemplate="%{x}<br>" + METRICS[metric] + ": %{y:.2f}<br>n=%{customdata}<extra></extra>",
                visible=(i == 0),
                name=METRICS[metric],
            )
        )

    buttons = []
    for i, metric in enumerate(metric_keys):
        visible = [j == i for j in range(len(metric_keys))]
        buttons.append(
            dict(
                label=METRICS[metric],
                method="update",
                args=[{"visible": visible}, {"yaxis": {"title": METRICS[metric]}}],
            )
        )

    fig.update_layout(
        title="Search Effort by Algorithm / Heuristic (average over solved levels)",
        xaxis_title="Configuration",
        yaxis_title=METRICS[metric_keys[0]],
        updatemenus=[dict(active=0, buttons=buttons, x=1.0, xanchor="right", y=1.15, yanchor="top")],
        template="plotly_white",
        bargap=0.3,
    )
    return fig


def main():
    csv_path = sys.argv[1]
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    fig = build(df)
    fig.write_html(outdir / "search_effort.html", include_plotlyjs="cdn")
    print(f"Wrote {outdir / 'search_effort.html'}")


if __name__ == "__main__":
    main()