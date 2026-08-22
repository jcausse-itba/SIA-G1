"""
Scalability: how effort grows as levels get harder.

There's no explicit "difficulty" column, so we derive one: each level's
BFS solution cost (BFS is uninformed but cost-optimal, so it's a fair proxy
for how "big" the search space is). Levels are sorted by that proxy and used
as the x-axis category order — this is usually the most convincing chart in
a report, since it's where uninformed search visibly falls over.

Falls back to DFS cost, then the minimum cost seen for that level, if BFS
didn't succeed on a given level.

Merges expanded_nodes / elapsed_seconds / frontier_nodes into one file via a
metric dropdown, same pattern as plot_search_effort.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

import plotly.graph_objects as go

from .common import load_data, successful, config_order, config_color

METRICS = {
    "expanded_nodes": "Expanded Nodes",
    "elapsed_seconds": "Elapsed Time (s)",
    "frontier_nodes": "Frontier Size (nodes)",
}


def difficulty_order(df) -> list[str]:
    ok = successful(df)

    def proxy_for_level(sub):
        bfs_rows = sub[sub["algorithm"] == "bfs"]
        if not bfs_rows.empty:
            return bfs_rows["cost"].iloc[0]
        dfs_rows = sub[sub["algorithm"] == "dfs"]
        if not dfs_rows.empty:
            return dfs_rows["cost"].iloc[0]
        return sub["cost"].min()

    proxies = ok.groupby("level").apply(proxy_for_level).dropna().sort_values()
    return list(proxies.index)


def build(df) -> go.Figure:
    ok = successful(df)
    configs = config_order(ok)
    levels = difficulty_order(df)
    ok = ok[ok["level"].isin(levels)]

    fig = go.Figure()
    metric_keys = list(METRICS.keys())

    for i, metric in enumerate(metric_keys):
        for cfg in configs:
            sub = ok[ok["config"] == cfg].set_index("level").reindex(levels)
            fig.add_trace(
                go.Scatter(
                    x=levels,
                    y=sub[metric],
                    mode="lines+markers",
                    name=cfg,
                    legendgroup=cfg,
                    line=dict(color=config_color(cfg)),
                    connectgaps=True,
                    hovertemplate="Level: %{x}<br>" + METRICS[metric] + ": %{y}<extra>" + cfg + "</extra>",
                    visible=(i == 0),
                )
            )

    n = len(configs)
    buttons = []
    for i, metric in enumerate(metric_keys):
        visible = [False] * (len(metric_keys) * n)
        visible[i * n : (i + 1) * n] = [True] * n
        buttons.append(
            dict(
                label=METRICS[metric],
                method="update",
                args=[{"visible": visible}, {"yaxis": {"title": METRICS[metric]}}],
            )
        )

    fig.update_layout(
        title="Scalability — effort vs. level difficulty (levels sorted by BFS-optimal cost)",
        xaxis_title="Level (increasing difficulty →)",
        yaxis_title=METRICS[metric_keys[0]],
        yaxis_type="log",
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
    fig.write_html(outdir / "scalability.html", include_plotlyjs="cdn")
    print(f"Wrote {outdir / 'scalability.html'}")


if __name__ == "__main__":
    main()