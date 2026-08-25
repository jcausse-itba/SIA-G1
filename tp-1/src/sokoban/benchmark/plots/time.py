"""
Execution time.

This chart shows, per level, the execution time ("elapsed_seconds") for each
configuration.

Levels are selected via a dropdown menu.
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

    for i, level in enumerate(levels):
        sub = ok[ok["level"] == level].set_index("config").reindex(configs)

        fig.add_trace(
            go.Bar(
                x=configs,
                y=sub["elapsed_seconds"],
                marker_color=[config_color(c) for c in configs],
                hovertemplate="%{x}<br>Time: %{y:.3f}s<extra></extra>",
                visible=(i == 0),
                name=level,
                showlegend=False,
            )
        )

    buttons = []
    for i, level in enumerate(levels):
        visible = [False] * len(levels)
        visible[i] = True
        buttons.append(
            dict(
                label=level,
                method="update",
                args=[{"visible": visible}, {"title": f"Execution Time — {level}"}],
            )
        )

    fig.update_layout(
        title=f"Execution Time — {levels[0] if levels else ''}",
        xaxis_title="Configuration",
        yaxis_title="Elapsed Time (s)",
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
    fig.write_html(outdir / "time.html", include_plotlyjs="cdn")
    print(f"Wrote {outdir / 'time.html'}")


if __name__ == "__main__":
    main()