"""
Entry point for the `plots` package.

Usage:
    python -m plots --csv results.csv --outdir plots_output/
    python -m plots -c results.csv -o plots_output/
"""

from __future__ import annotations
import argparse
from pathlib import Path

from .common import load_data
from . import cost, ebf, effort_vs_quality, heuristic_quality, scalability, search_effort

# (module, output filename)
MODULES = [
    (search_effort, "search_effort.html"),
    (effort_vs_quality, "effort_vs_quality.html"),
    (heuristic_quality, "heuristic_quality.html"),
    (ebf, "effective_branching_factor.html"),
    (scalability, "scalability.html"),
    (cost, "cost_optimality.html"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate every Sokoban benchmark plot from a results CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--csv",
        type=Path,
        required=True,
        help="Path to the benchmark results CSV",
    )
    parser.add_argument(
        "-o", "--outdir",
        type=Path,
        default=Path("plots_output"),
        help="Directory to write the generated HTML plots to",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise SystemExit(f"Error: CSV file not found at '{args.csv}'")

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)

    for module, filename in MODULES:
        fig = module.build(df)
        out_path = args.outdir / filename
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"Wrote {out_path}")

    print(f"\nDone. {len(MODULES)} plot(s) written to '{args.outdir}'.")


if __name__ == "__main__":
    main()