"""
Effort vs. quality scatter.

Merges two related scatter plots into one file via a dropdown on the Y axis:
  - Expanded Nodes vs Solution Cost   (how much search buys you a better solution)
  - Expanded Nodes vs Frontier Size   (memory footprint vs work done)

X axis is always expanded_nodes (log scale, since BFS/DFS can dwarf A*/Greedy).
One trace per config so you get a legend you can click to isolate algorithms.
"""

from __future__ import annotations
import sys
from pathlib import Path

import plotly.graph_objects as go

from .common import load_data, successful, config_order, config_color

Y_METRICS = {
    "cost": "Solution Cost",
    "frontier_nodes": "Frontier Size (nodes)",
}


def build(df) -> go.Figure:
    ok = successful(df)
    configs = config_order(ok)

    fig = go.Figure()

    # traces are grouped in blocks: [config0_cost, config1_cost, ..., config0_frontier, config1_frontier, ...]
    n = len(configs)
    for y_key in Y_METRICS:
        for cfg in configs:
            sub = ok[ok["config"] == cfg]
            fig.add_trace(
                go.Scatter(
                    x=sub["expanded_nodes"],
                    y=sub[y_key],
                    mode="markers",
                    name=cfg,
                    legendgroup=cfg,
                    marker=dict(color=config_color(cfg), size=8, opacity=0.75),
                    text=sub["level"],
                    hovertemplate="Level: %{text}<br>Expanded: %{x}<br>"
                    + Y_METRICS[y_key]
                    + ": %{y}<extra>"
                    + cfg
                    + "</extra>",
                    visible=(y_key == "cost"),
                )
            )

    y_keys = list(Y_METRICS.keys())
    buttons = []
    for i, y_key in enumerate(y_keys):
        visible = [False] * (2 * n)
        visible[i * n : (i + 1) * n] = [True] * n
        buttons.append(
            dict(
                label=Y_METRICS[y_key],
                method="update",
                args=[{"visible": visible}, {"yaxis": {"title": Y_METRICS[y_key]}}],
            )
        )

    fig.update_layout(
        title="Search Effort vs. Solution Quality",
        xaxis_title="Expanded Nodes",
        xaxis_type="log",
        yaxis_title=Y_METRICS[y_keys[0]],
        updatemenus=[dict(active=0, buttons=buttons, x=1.0, xanchor="right", y=1.15, yanchor="top")],
        template="plotly_white",
        legend_title="Configuration",
    )
    return fig


def main():
    csv_path = sys.argv[1]
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    fig = build(df)
    fig.write_html(outdir / "effort_vs_quality.html", include_plotlyjs="cdn")
    print(f"Wrote {outdir / 'effort_vs_quality.html'}")


if __name__ == "__main__":
    main()