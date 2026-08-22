"""
Batch-runs every search algorithm (and, for A*/Greedy, every heuristic) over
every .level file found in a directory, and writes the results to a CSV.

Usage:
    python -m yourpackage.benchmark -d levels/ -o results.csv
    python -m yourpackage.benchmark -d levels/ -o results.csv --recursive
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from ..algorithm.a_star import AStar
from ..algorithm.bfs import BFS
from ..algorithm.greedy import GreedyBFS
from ..algorithm.dfs import DFS
from ..algorithm.heuristics import (
    matching_with_player_heuristic,
    min_goal_distance_heuristic,
    unique_goal_matching_heuristic,
)
from ..parser.loader import load_level

# Algorithms that don't take a heuristic — run once per level, heuristic col left empty.
ALGORITHMS_NO_HEURISTIC = {
    "bfs": BFS(),
    "dfs": DFS()
}

# Algorithms that do take a heuristic — run once per (algorithm, heuristic) pair.
ALGORITHMS_WITH_HEURISTIC = {
    "astar": AStar(),
    "greedy": GreedyBFS(),
}

HEURISTIC_REGISTRY = {
    "min_goal_distance": min_goal_distance_heuristic,
    "player_distance": matching_with_player_heuristic,
    # "unique_min_goal_distance": unique_goal_matching_heuristic,
    # "none": lambda board: 0.0,
}

CSV_FIELDS = [
    "level",
    "algorithm",
    "heuristic",
    "success",
    "cost",
    "expanded_nodes",
    "frontier_nodes",
    "elapsed_seconds",
    "operations_done",
    "solution_length",
    "solution",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every algorithm/heuristic combo on every level in a directory and export to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--dir",
        type=Path,
        required=True,
        help="Directory containing .level files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for .level files recursively",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.level",
        help="Glob pattern used to find level files",
    )
    return parser.parse_args()


def find_level_files(directory: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(directory.rglob(pattern))
    return sorted(directory.glob(pattern))


def run_single(algorithm_name, algorithm, heuristic_name, heuristic_fn, level_path: Path) -> dict:
    """Load the level fresh and run one (algorithm, heuristic) combo on it."""
    row = {
        "level": level_path.name,
        "algorithm": algorithm_name,
        "heuristic": heuristic_name or "",
        "success": False,
        "cost": "",
        "expanded_nodes": "",
        "frontier_nodes": "",
        "elapsed_seconds": "",
        "operations_done": "",
        "solution_length": "",
        "solution": "",
        "error": "",
    }

    try:
        # Reload the board fresh for every run in case search mutates state.
        board = load_level(level_path)

        start = time.perf_counter()
        result = algorithm.search(board, heuristic_fn)
        wall_time = time.perf_counter() - start

        row["success"] = result.success
        row["cost"] = result.cost
        row["expanded_nodes"] = result.expanded_nodes
        row["frontier_nodes"] = result.frontier_nodes
        row["elapsed_seconds"] = getattr(result, "elapsed_seconds", wall_time)
        row["operations_done"] = result.operations_done
        row["solution_length"] = len(result.solution) if result.solution else 0
        row["solution"] = "".join(str(d) for d in result.solution) if result.solution else ""

    except Exception as e:  # noqa: BLE001 - we want to capture *any* failure and keep going
        row["error"] = f"{type(e).__name__}: {e}"
        row["success"] = False
        # Uncomment for full tracebacks while debugging:
        # traceback.print_exc()

    return row


def main() -> None:
    args = parse_args()

    if not args.dir.exists() or not args.dir.is_dir():
        print(f"Error: '{args.dir}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    level_files = find_level_files(args.dir, args.pattern, args.recursive)
    if not level_files:
        print(f"No files matching '{args.pattern}' found in '{args.dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(level_files)} level file(s). Writing results to '{args.output}'.\n")

    total_runs = len(level_files) * (
        len(ALGORITHMS_NO_HEURISTIC)
        + len(ALGORITHMS_WITH_HEURISTIC) * len(HEURISTIC_REGISTRY)
    )
    done = 0

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for level_path in level_files:
            # Algorithms without a heuristic (bfs)
            for algo_name, algo in ALGORITHMS_NO_HEURISTIC.items():
                done += 1
                print(f"[{done}/{total_runs}] {level_path.name} | {algo_name}")
                row = run_single(algo_name, algo, "", None, level_path)
                writer.writerow(row)
                f.flush()

            # Algorithms with a heuristic (astar, greedy)
            for algo_name, algo in ALGORITHMS_WITH_HEURISTIC.items():
                for heuristic_name, heuristic_fn in HEURISTIC_REGISTRY.items():
                    done += 1
                    print(f"[{done}/{total_runs}] {level_path.name} | {algo_name} | {heuristic_name}")
                    row = run_single(algo_name, algo, heuristic_name, heuristic_fn, level_path)
                    writer.writerow(row)
                    f.flush()

    print(f"\nDone. Results written to '{args.output}'.")


if __name__ == "__main__":
    main()