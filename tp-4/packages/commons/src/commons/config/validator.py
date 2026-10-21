"""Helpers de validación + chequeos comunes.

Los rangos de cada valor individual ya los garantizan los ``type=`` del
parser (CLI y archivo). Acá van las validaciones entre campos y las de
existencia de archivos/columnas.
"""
import csv
import os
from typing import Any, Dict, Iterable


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_file(path: str | None, label: str) -> None:
    if not path:
        raise ValueError(f"{label} path is required.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} file not found at: {path}")


def require_dir(path: str | None, label: str) -> None:
    if not path:
        raise ValueError(f"{label} path is required.")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory not found at: {path}")


def csv_header(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return next(csv.reader(f), [])


def require_columns(header: Iterable[str], columns: Iterable[str] | None, option: str) -> None:
    header = set(header)
    missing = [c for c in (columns or []) if c not in header]
    if missing:
        raise ValueError(f"Columns {missing} (from {option}) not found in the dataset.")


def validate_common(cfg: Dict[str, Any]) -> None:
    """Chequeos de los argumentos de build_base_parser."""
    seed = cfg.get("seed")
    if seed is not None:
        require(seed >= 0, "Seed (--seed) must be >= 0 or 'none'.")

    out = str(cfg.get("output_dir") or "")
    require(out != "", "Output directory (--output-dir) cannot be empty.")
    require(not os.path.isfile(out), f"--output-dir '{out}' exists and is a file.")
