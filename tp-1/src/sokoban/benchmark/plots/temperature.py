"""
Temperature map — per-level, per-configuration cost.

One row per algorithm+heuristic config, one column per level. Color encodes
solution COST on a log scale:
  BLACK             = cost is missing or 0 for that (config, level) cell —
                       covers real failures, unrun configs, or zero-cost cases.
  GREEN -> RED      = a positive numeric cost was recorded; green is low (good)
                       cost, transitioning to red as cost magnitude grows. The
                       scale is log-based since cost can span orders of magnitude.

Level names are zero-padded on their trailing number before sorting/display,
so "2.level" < "10.level" alphabetically instead of "10.level" < "2.level".
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .common import load_data


_TRAILING_NUMBER = re.compile(r"(\d+)(?!.*\d)")  # last run of digits in the string


def _pad_width(names: list[str]) -> int:
    widths = [len(m.group(1)) for name in names if (m := _TRAILING_NUMBER.search(name))]
    return max(widths) if widths else 0


def _zero_pad(name: str, width: int) -> str:
    m = _TRAILING_NUMBER.search(name)
    if not m:
        return name
    return name[: m.start()] + m.group(1).zfill(width) + name[m.end() :]


def build(df) -> go.Figure:
    configs = sorted(df["config"].unique())

    raw_levels = df["level"].unique().tolist()
    width = _pad_width(raw_levels)
    padded = {lvl: _zero_pad(lvl, width) for lvl in raw_levels}
    levels = sorted(raw_levels, key=lambda lvl: padded[lvl])  # sort by padded form
    display_labels = [padded[lvl] for lvl in levels]

    cost_grid = df.pivot_table(index="config", columns="level", values="cost", aggfunc="first")
    cost_grid = cost_grid.reindex(index=configs, columns=levels)
    cost_vals = cost_grid.values.astype(float)
    
    # Identify zero/missing vs positive cost
    is_zero_or_nan = np.isnan(cost_vals) | (cost_vals <= 0)
    has_positive_cost = ~is_zero_or_nan

    fig = go.Figure()

    # Layer 1: Solid black background wherever cost is missing or 0
    zero_z = np.where(is_zero_or_nan, 0.0, np.nan)
    fig.add_trace(
        go.Heatmap(
            z=zero_z,
            x=display_labels,
            y=configs,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#ffffff"], [1, "#ffffff"]],
            showscale=False,
            hovertemplate="Level: %{x}<br>Config: %{y}<br>Cost: 0 or not recorded<extra></extra>",
        )
    )

    # Layer 2: Green (low cost) -> Yellow -> Red (high cost) for positive costs
    if has_positive_cost.any():
        pos_costs = cost_vals[has_positive_cost]
        min_cost = np.min(pos_costs)
        max_cost = np.max(pos_costs)

        log_vals = np.where(has_positive_cost, np.log10(cost_vals), np.nan)
        zmin = np.log10(min_cost)
        zmax = np.log10(max(max_cost, min_cost * 10))

        # Colorbar ticks
        tick_actual = np.unique(np.round(np.geomspace(max(min_cost, 1), max(max_cost, min_cost + 1), num=6)).astype(int))
        tick_actual = tick_actual[tick_actual > 0]
        tick_log = np.log10(tick_actual)

        fig.add_trace(
            go.Heatmap(
                z=log_vals,
                x=display_labels,
                y=configs,
                zmin=zmin,
                zmax=zmax,
                colorscale=[
                    [0.0, "#1a9850"],   # Green (Low cost / Best)
                    [0.5, "#ffffbf"],   # Yellow (Medium cost)
                    [1.0, "#d73027"],   # Red (High cost / Worst)
                ],
                colorbar=dict(title="Cost", tickvals=tick_log, ticktext=[str(v) for v in tick_actual]),
                customdata=cost_vals,
                hovertemplate="Level: %{x}<br>Config: %{y}<br>Cost: %{customdata}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Temperature Map — Solution Cost by Configuration (log scale, black = 0 / missing cost)",
        xaxis=dict(
            title="Level",
            type="category",
            categoryorder="array",
            categoryarray=display_labels,
        ),
        yaxis=dict(
            title="Configuration",
            type="category",
            categoryorder="array",
            categoryarray=configs,
            autorange="reversed",  # alphabetically first config at the top
        ),
        template="plotly_white",
        height=max(300, 60 + 40 * len(configs)),
    )
    return fig


def main():
    csv_path = sys.argv[1]
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    fig = build(df)
    fig.write_html(outdir / "temperature_map.html", include_plotlyjs="cdn")
    print(f"Wrote {outdir / 'temperature_map.html'}")


if __name__ == "__main__":
    main()