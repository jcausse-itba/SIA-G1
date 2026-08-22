"""
Effective Branching Factor (EBF).

EBF normalizes expanded_nodes by solution depth, so a heuristic isn't unfairly
penalized just because it was tested on a deeper level. This is the standard
metric used in AI courses to compare heuristic quality independent of problem size.

Solves for b* in:  N + 1 = sum_{i=0}^{d} b*^i     (N = expanded_nodes, d = solution_length)
via bisection (no scipy dependency needed).
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .common import load_data, successful, config_order, config_color


def _reaches_target(b: float, depth: int, target: float, max_steps: int = 100_000) -> bool:
    """
    True if sum_{i=0}^{depth} b**i >= target.

    Walks the geometric sum term-by-term and stops as soon as it crosses
    `target`, instead of computing b**depth directly — for large depth/b
    that power can exceed float's max (~1.8e308) and raise OverflowError
    even though we only ever need to know whether the sum crosses target,
    not its exact value.
    """
    total = 0.0
    term = 1.0
    steps = min(depth, max_steps) + 1
    for _ in range(steps):
        total += term
        if total >= target:
            return True
        term *= b
    return total >= target


def effective_branching_factor(n_expanded: float, depth: float, tol: float = 1e-4, max_iter: int = 200) -> float:
    if pd.isna(n_expanded) or pd.isna(depth) or depth <= 0 or n_expanded <= 0:
        return float("nan")

    target = n_expanded + 1  # total nodes including root
    depth = int(depth)

    lo, hi = 1.0 + 1e-6, 50.0
    if _reaches_target(lo, depth, target):
        return 1.0
    if not _reaches_target(hi, depth, target):
        return float("nan")  # branching factor absurdly high; treat as unbounded

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if _reaches_target(mid, depth, target):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def build(df) -> go.Figure:
    ok = successful(df).copy()
    ok["ebf"] = ok.apply(
        lambda r: effective_branching_factor(r["expanded_nodes"], r["solution_length"]), axis=1
    )
    ok = ok.dropna(subset=["ebf"])
    configs = config_order(ok)

    fig = go.Figure()
    for cfg in configs:
        sub = ok[ok["config"] == cfg]
        fig.add_trace(
            go.Box(
                y=sub["ebf"],
                name=cfg,
                marker_color=config_color(cfg),
                boxpoints="all",
                jitter=0.4,
                pointpos=-1.8,
                text=sub["level"],
                hovertemplate="Level: %{text}<br>EBF: %{y:.3f}<extra>" + cfg + "</extra>",
            )
        )

    fig.update_layout(
        title="Effective Branching Factor by Configuration (lower = better-informed search)",
        yaxis_title="Effective Branching Factor (b*)",
        xaxis_title="Configuration",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def main():
    csv_path = sys.argv[1]
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    fig = build(df)
    fig.write_html(outdir / "effective_branching_factor.html", include_plotlyjs="cdn")
    print(f"Wrote {outdir / 'effective_branching_factor.html'}")


if __name__ == "__main__":
    main()