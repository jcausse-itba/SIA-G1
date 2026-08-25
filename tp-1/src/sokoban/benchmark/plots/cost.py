"""
Cost optimality.

BFS explores level-by-level so its solution cost is optimal (assuming uniform
move cost). This chart shows, per level, how each configuration's solution
cost compares to that optimal baseline — this is usually the punchline for
Greedy: faster, but at what quality cost?

Since showing every level in one bar chart would be unreadable, levels are
selected via a dropdown (one pre-built trace pair per level: bars + a dashed
baseline line at the BFS-optimal cost).
"""

from __future__ import annotations
import sys
from pathlib import Path

import plotly.graph_objects as go

from .common import load_data, successful, config_order, config_color


def build(df) -> go.Figure:
    ok = successful(df)
    configs = config_order(ok)
    levels = sorted(ok["level"].unique())

    fig = go.Figure()
    n_traces_per_level = 2  # [bar, baseline line]

    for i, level in enumerate(levels):
        sub = ok[ok["level"] == level].set_index("config").reindex(configs)
        bfs_cost = sub.loc["bfs", "cost"] if "bfs" in sub.index else None

        fig.add_trace(
            go.Bar(
                x=configs,
                y=sub["cost"],
                texttemplate="%{y}",
                textposition="auto",
                marker_color=[config_color(c) for c in configs],
                hovertemplate="%{x}<br>Cost: %{y}<extra></extra>",
                visible=(i == 0),
                name=level,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=configs,
                y=[bfs_cost] * len(configs) if bfs_cost is not None else [None] * len(configs),
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="BFS-optimal cost",
                hovertemplate="BFS-optimal: %{y}<extra></extra>",
                visible=(i == 0),
                showlegend=(i == 0),
            )
        )

    total = len(levels) * n_traces_per_level
    buttons = []
    for i, level in enumerate(levels):
        visible = [False] * total
        visible[i * n_traces_per_level] = True
        visible[i * n_traces_per_level + 1] = True
        buttons.append(
            dict(
                label=level,
                method="update",
                args=[{"visible": visible}, {"title": f"Solution Cost vs. BFS-Optimal — {level}"}],
            )
        )

    fig.update_layout(
        title=f"Solution Cost vs. BFS-Optimal — {levels[0] if levels else ''}",
        xaxis_title="Configuration",
        yaxis_title="Solution Cost",
        updatemenus=[dict(active=0, buttons=buttons, x=1.0, xanchor="right", y=1.15, yanchor="top", direction="down")],
        template="plotly_white",
    )
    return fig


def main():
    csv_path = sys.argv[1]
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    fig = build(df)
    fig.write_html(outdir / "cost_optimality.html", include_plotlyjs="cdn")
    print(f"Wrote {outdir / 'cost_optimality.html'}")


if __name__ == "__main__":
    main()