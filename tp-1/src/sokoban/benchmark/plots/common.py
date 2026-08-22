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

# Fixed order / color so every chart uses the same visual language for a given config.
CONFIG_ORDER = [
    "bfs",
    "dfs",
    "astar (min_goal_distance)",
    "astar (player_distance)",
    "greedy (min_goal_distance)",
    "greedy (player_distance)",
]

CONFIG_COLORS = {
    "bfs": "#7f7f7f",
    "dfs": "#bcbd22",
    "astar (min_goal_distance)": "#1f77b4",
    "astar (player_distance)": "#17becf",
    "greedy (min_goal_distance)": "#d62728",
    "greedy (player_distance)": "#ff7f0e",
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
    present = set(df["config"].unique())
    ordered = [c for c in CONFIG_ORDER if c in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


def config_color(config: str) -> str:
    return CONFIG_COLORS.get(config, "#999999")