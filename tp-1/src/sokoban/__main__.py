import argparse
from pathlib import Path
import sys

from .algorithm.a_star import AStar
from .loader import load_level
from .visualizer import SokobanVisualizer

ALGORITHM_REGISTRY = {
    # "dfs": DFS(),
    # "bfs": BFS(),
    "astar": AStar(),
    # "greedy": GreedySearch(),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a Sokoban level using search algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p", "--path",
        type=Path,
        required=True,
        help="Path to the .level file",
    )

    parser.add_argument(
        "-a", "--algorithm",
        type=str.lower,
        choices=list(ALGORITHM_REGISTRY.keys()),
        default="astar",
        help="Search algorithm to use",
    )

    parser.add_argument(
        "-g", "--gif",
        type=Path,
        default=None,
        help="Optional path to save the solution GIF. If omitted, no GIF is generated.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not args.path.exists():
        print(f"Error: Level file not found at '{args.path}'", file=sys.stderr)
        sys.exit(1)

    try:
        board = load_level(args.path)
    except Exception as e:
        print(f"Error parsing level file: {e}", file=sys.stderr)
        sys.exit(1)

    algorithm_solver = ALGORITHM_REGISTRY[args.algorithm]
    print(f"Running {args.algorithm.upper()} on level: {args.path.name}...\n")

    result = algorithm_solver.search(board)

    print(result)
    
    if args.gif:
        print(f"\nGenerating solution GIF at: {args.gif}")
        viz = SokobanVisualizer()
        viz.create_solution_gif(board, result.solution, str(args.gif))

if __name__ == "__main__":
    main()