"""
Shared helpers for loading and labeling the benchmark CSV
(level, algorithm, heuristic, success, cost, expanded_nodes, frontier_nodes,
 elapsed_seconds, operations_done, solution_length, solution, error)

Every plot_*.py script imports this instead of re-implementing loading logic,
so all charts stay consistent (same config labels, same colors, same ordering).
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

NUMERIC_COLUMNS = [
    "cost",
    "expanded_nodes",
    "frontier_nodes",
    "elapsed_seconds",
    "operations_done",
    "solution_length",
]

# Grouped color palette based on algorithm family & state collapse status
COLOR_MAP = {
    "astar - collapsed": "#aec7e8",  # Light Blue
    "astar": "#1f77b4",              # Dark Blue
    "bfs - collapsed": "#98df8a",    # Light Green
    "bfs": "#2ca02c",                # Dark Green
    "dfs - collapsed": "#dbdb8d",    # Light Yellow-Olive
    "dfs": "#bcbd22",                # Dark Yellow-Olive
    "greedy - collapsed": "#ff9896", # Light Red / Coral
    "greedy": "#d62728",             # Dark Red
}


def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["success"] = df["success"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df["heuristic"] = df["heuristic"].fillna("").astype(str)
    df["algorithm"] = df["algorithm"].astype(str)
    df["level"] = df["level"].astype(str)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["config"] = df.apply(
        lambda r: r["algorithm"] if not r["heuristic"] else f"{r['algorithm']} ({r['heuristic']})",
        axis=1,
    )

    return df


def successful(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where a solution was actually found. Most effort/quality metrics only make sense here."""
    return df[df["success"]].copy()


def config_order(df: pd.DataFrame) -> list[str]:
    """Returns unique configurations sorted alphabetically."""
    return sorted(df["config"].unique())


def config_color(config: str) -> str:
    """Assigns visual color based on algorithm family prefix."""
    for key, color in COLOR_MAP.items():
        if config.startswith(key):
            return color
    return "#7f7f7f"
