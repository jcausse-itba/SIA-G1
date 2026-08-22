"""
Heuristic quality.

Box plot of expanded_nodes across levels, grouped by heuristic — the classic
way to show which heuristic prunes the search space best. Since BOTH astar
and greedy take heuristics in this project, a dropdown lets you flip between
"which algorithm's heuristics am I comparing" instead of needing two files.
"""

from __future__ import annotations
import sys
from pathlib import Path

import plotly.graph_objects as go

from .common import load_data, successful

ALGOS = ["astar", "greedy"]
HEURISTIC_COLORS = {
    "min_goal_distance": "#1f77b4",
    "player_distance": "#ff7f0e",
}


def build(df) -> go.Figure:
    ok = successful(df)
    ok = ok[ok["heuristic"] != ""]  # only heuristic-driven algorithms

    fig = go.Figure()

    heuristics_by_algo = {
        algo: sorted(ok.loc[ok["algorithm"] == algo, "heuristic"].unique()) for algo in ALGOS
    }

    trace_index_ranges = {}
    idx = 0
    for algo in ALGOS:
        start = idx
        for h in heuristics_by_algo[algo]:
            sub = ok[(ok["algorithm"] == algo) & (ok["heuristic"] == h)]
            fig.add_trace(
                go.Box(
                    y=sub["expanded_nodes"],
                    name=h,
                    marker_color=HEURISTIC_COLORS.get(h, "#999999"),
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=-1.8,
                    text=sub["level"],
                    hovertemplate="Level: %{text}<br>Expanded: %{y}<extra>" + h + "</extra>",
                    visible=(algo == ALGOS[0]),
                )
            )
            idx += 1
        trace_index_ranges[algo] = (start, idx)

    buttons = []
    total = idx
    for algo in ALGOS:
        start, end = trace_index_ranges[algo]
        visible = [start <= i < end for i in range(total)]
        buttons.append(
            dict(
                label=algo.upper(),
                method="update",
                args=[{"visible": visible}, {"title": f"Heuristic Quality — {algo.upper()} (expanded nodes)"}],
            )
        )

    fig.update_layout(
        title=f"Heuristic Quality — {ALGOS[0].upper()} (expanded nodes)",
        yaxis_title="Expanded Nodes",
        xaxis_title="Heuristic",
        updatemenus=[dict(active=0, buttons=buttons, x=1.0, xanchor="right", y=1.15, yanchor="top")],
        template="plotly_white",
    )
    return fig


def main():
    csv_path = sys.argv[1]
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    fig = build(df)
    fig.write_html(outdir / "heuristic_quality.html", include_plotlyjs="cdn")
    print(f"Wrote {outdir / 'heuristic_quality.html'}")


if __name__ == "__main__":
    main()