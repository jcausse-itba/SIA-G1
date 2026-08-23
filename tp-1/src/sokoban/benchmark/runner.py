import time
from pathlib import Path

from ..parser.loader import load_level


def run_single(algorithm_name, algorithm, heuristic_name, heuristic_fn, level_path: Path) -> dict:
    """Load the level fresh and run one (algorithm, heuristic) combo on it."""
    print(f"-> {level_path.name}   {algorithm_name}   {heuristic_name or ""}")
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