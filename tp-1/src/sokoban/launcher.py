"""Lightweight tkinter GUI launcher for the Sokoban solver.

Replaces the CLI (argparse) interface with a graphical form
that exposes every option from ``__main__.py``:
  * Level file selection (file-browser dialog)
  * Algorithm choice (dropdown)
  * Heuristic choice (dropdown)
  * Optional GIF output path (file-browser dialog)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── Registries (mirrors of __main__.py) ─────────────────────────────────────
# Imported lazily inside _run_solver() to avoid circular-import issues when
# this module is imported directly.

ALGORITHM_KEYS: list[str] = ["dfs", "bfs", "astar", "greedy"]
HEURISTIC_KEYS: list[str] = [
    "none",
    "min_goal_distance",
    "unique_min_goal_distance",
    "player_distance",
    "hungarian",
]

# ── Descriptions shown in the GUI ───────────────────────────────────────────

ALGORITHM_DESCRIPTIONS: dict[str, str] = {
    "dfs": "DFS – Depth-First Search",
    "bfs": "BFS – Breadth-First Search",
    "astar": "A* – A-Star Search",
    "greedy": "Greedy – Greedy Best-First Search",
}

HEURISTIC_DESCRIPTIONS: dict[str, str] = {
    "none": "None (no heuristic)",
    "min_goal_distance": "Min Goal Distance",
    "unique_min_goal_distance": "Unique Min Goal Distance",
    "player_distance": "Player Distance (matching + player)",
    "hungarian": "Hungarian (optimal matching)",
}

# Reverse look-ups: description → key
_ALG_DESC_TO_KEY: dict[str, str] = {v: k for k, v in ALGORITHM_DESCRIPTIONS.items()}
_HEUR_DESC_TO_KEY: dict[str, str] = {v: k for k, v in HEURISTIC_DESCRIPTIONS.items()}


# ── Solver logic (runs in a background thread) ──────────────────────────────

def _run_solver(
    level_path: Path,
    algorithm_key: str,
    heuristic_key: str,
    gif_path: Path | None,
    *,
    on_done: Callable[[str], None],
    on_error: Callable[[str], None],
) -> None:
    """Execute the Sokoban solver in the calling thread.

    This is intentionally identical in behaviour to ``__main__.main()`` but
    accepts explicit arguments instead of reading ``sys.argv``.
    """
    try:
        # Late imports to avoid circular references and keep module load fast.
        from .algorithm.a_star import AStar
        from .algorithm.bfs import BFS
        from .algorithm.dfs import DFS
        from .algorithm.greedy import GreedyBFS
        from .algorithm.heuristics import (
            hungarian_matching_heuristic,
            matching_with_player_heuristic,
            min_goal_distance_heuristic,
            unique_goal_matching_heuristic,
        )
        from .parser.loader import load_level
        from .visualizer import SokobanVisualizer

        algorithm_registry = {
            "dfs": DFS(),
            "bfs": BFS(),
            "astar": AStar(),
            "greedy": GreedyBFS(),
        }
        heuristic_registry = {
            "min_goal_distance": min_goal_distance_heuristic,
            "unique_min_goal_distance": unique_goal_matching_heuristic,
            "player_distance": matching_with_player_heuristic,
            "hungarian": hungarian_matching_heuristic,
            "none": lambda board: 0.0,
        }

        board = load_level(level_path)
        algorithm_solver = algorithm_registry[algorithm_key]
        heuristic = heuristic_registry[heuristic_key]

        result = algorithm_solver.search(board, heuristic)

        gif_msg = ""
        if gif_path is not None:
            viz = SokobanVisualizer()
            viz.create_solution_gif(board, result.solution, str(gif_path))
            gif_msg = f"\n\nGIF saved to:\n{gif_path}"

        on_done(
            f"Solver finished!\n\n{result}{gif_msg}"
        )
    except Exception as exc:
        on_error(str(exc))


# ── GUI ─────────────────────────────────────────────────────────────────────

class SokobanLauncher(tk.Tk):
    """Main application window for the Sokoban solver launcher."""

    _WINDOW_TITLE = "Sokoban Solver"
    _PAD = 10

    def __init__(self) -> None:
        super().__init__()
        self.title(self._WINDOW_TITLE)
        self.resizable(False, False)

        # ── Variables ────────────────────────────────────────────────────
        self._level_var = tk.StringVar()
        self._alg_var = tk.StringVar(
            value=ALGORITHM_DESCRIPTIONS["astar"]  # default: astar
        )
        self._heur_var = tk.StringVar(
            value=HEURISTIC_DESCRIPTIONS["none"]  # default: none
        )
        self._gif_var = tk.StringVar()

        self._build_ui()
        self._center_window()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = self._PAD

        # Title label
        title_label = ttk.Label(
            self,
            text="Sokoban Solver Launcher",
            font=("Segoe UI", 14, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(pad, pad * 2))

        # ── Level file ───────────────────────────────────────────────────
        ttk.Label(self, text="Level file:").grid(
            row=1, column=0, sticky="w", padx=pad, pady=4
        )
        level_entry = ttk.Entry(self, textvariable=self._level_var, width=48)
        level_entry.grid(row=1, column=1, padx=(0, 4), pady=4)
        ttk.Button(self, text="Browse…", command=self._browse_level).grid(
            row=1, column=2, padx=(0, pad), pady=4
        )

        # ── Algorithm ────────────────────────────────────────────────────
        ttk.Label(self, text="Algorithm:").grid(
            row=2, column=0, sticky="w", padx=pad, pady=4
        )
        alg_combo = ttk.Combobox(
            self,
            textvariable=self._alg_var,
            values=list(ALGORITHM_DESCRIPTIONS.values()),
            state="readonly",
            width=45,
        )
        alg_combo.grid(row=2, column=1, columnspan=2, padx=(0, pad), pady=4, sticky="w")

        # ── Heuristic ───────────────────────────────────────────────────
        ttk.Label(self, text="Heuristic:").grid(
            row=3, column=0, sticky="w", padx=pad, pady=4
        )
        heur_combo = ttk.Combobox(
            self,
            textvariable=self._heur_var,
            values=list(HEURISTIC_DESCRIPTIONS.values()),
            state="readonly",
            width=45,
        )
        heur_combo.grid(row=3, column=1, columnspan=2, padx=(0, pad), pady=4, sticky="w")

        # ── GIF output (optional) ───────────────────────────────────────
        ttk.Label(self, text="Save GIF to:").grid(
            row=4, column=0, sticky="w", padx=pad, pady=4
        )
        gif_entry = ttk.Entry(self, textvariable=self._gif_var, width=48)
        gif_entry.grid(row=4, column=1, padx=(0, 4), pady=4)
        ttk.Button(self, text="Browse…", command=self._browse_gif).grid(
            row=4, column=2, padx=(0, pad), pady=4
        )

        # ── Run button ──────────────────────────────────────────────────
        self._run_btn = ttk.Button(
            self, text="▶  Run Solver", command=self._on_run
        )
        self._run_btn.grid(
            row=5, column=0, columnspan=3, pady=(pad * 2, pad), ipady=4
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _center_window(self) -> None:
        """Center the window on the screen after widgets have been laid out."""
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    def _browse_level(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a Sokoban level file",
            filetypes=[("Level files", "*.level"), ("All files", "*.*")],
        )
        if path:
            self._level_var.set(path)

    def _browse_gif(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save solution GIF as…",
            defaultextension=".gif",
            filetypes=[("GIF image", "*.gif"), ("All files", "*.*")],
        )
        if path:
            self._gif_var.set(path)

    # ── Run logic ────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        # Validate level path
        level_path_str = self._level_var.get().strip()
        if not level_path_str:
            messagebox.showwarning("Missing field", "Please select a level file.")
            return
        level_path = Path(level_path_str)
        if not level_path.exists():
            messagebox.showerror(
                "File not found", f"Level file not found:\n{level_path}"
            )
            return

        algorithm_key = _ALG_DESC_TO_KEY[self._alg_var.get()]
        heuristic_key = _HEUR_DESC_TO_KEY[self._heur_var.get()]

        gif_path_str = self._gif_var.get().strip()
        gif_path: Path | None = Path(gif_path_str) if gif_path_str else None

        # Disable the button while the solver runs
        self._run_btn.configure(state="disabled", text="⏳  Solving…")

        def on_done(msg: str) -> None:
            self.after(0, lambda: self._solver_done(msg))

        def on_error(msg: str) -> None:
            self.after(0, lambda: self._solver_error(msg))

        thread = threading.Thread(
            target=_run_solver,
            args=(level_path, algorithm_key, heuristic_key, gif_path),
            kwargs={"on_done": on_done, "on_error": on_error},
            daemon=True,
        )
        thread.start()

    def _solver_done(self, msg: str) -> None:
        self._run_btn.configure(state="normal", text="▶  Run Solver")
        messagebox.showinfo("Result", msg)

    def _solver_error(self, msg: str) -> None:
        self._run_btn.configure(state="normal", text="▶  Run Solver")
        messagebox.showerror("Error", f"Solver failed:\n\n{msg}")


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Launch the Sokoban solver GUI."""
    app = SokobanLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
