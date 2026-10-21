"""Parser base y tipos de argumento reutilizables.

Cada ejercicio llama a ``build_base_parser`` y le agrega sus propios
argument groups. Los tipos validan el rango de cada valor tanto si viene de
la CLI como del archivo de configuración (el loader también los aplica).
"""
import argparse
import time
from typing import Sequence


# --------------------------------------------------------------------- tipos
def _as_int(val) -> int:
    if isinstance(val, bool):
        raise argparse.ArgumentTypeError(f"Invalid int value: '{val}'")
    if isinstance(val, float):
        if not val.is_integer():
            raise argparse.ArgumentTypeError(f"Invalid int value: '{val}'")
        return int(val)
    try:
        return int(val)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"Invalid int value: '{val}'")


def _as_float(val) -> float:
    if isinstance(val, bool):
        raise argparse.ArgumentTypeError(f"Invalid float value: '{val}'")
    try:
        return float(val)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"Invalid float value: '{val}'")


def _is_none(val) -> bool:
    return isinstance(val, str) and val.strip().lower() in ("none", "null")


def nullable_int(val) -> int | None:
    return None if _is_none(val) else _as_int(val)


def nullable_float(val) -> float | None:
    return None if _is_none(val) else _as_float(val)


def positive_int(val) -> int:
    n = _as_int(val)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"Expected an integer > 0, got {n}")
    return n


def non_negative_int(val) -> int:
    n = _as_int(val)
    if n < 0:
        raise argparse.ArgumentTypeError(f"Expected an integer >= 0, got {n}")
    return n


def positive_float(val) -> float:
    x = _as_float(val)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"Expected a number > 0, got {x}")
    return x


def non_negative_float(val) -> float:
    x = _as_float(val)
    if x < 0:
        raise argparse.ArgumentTypeError(f"Expected a number >= 0, got {x}")
    return x


def unit_float(val) -> float:
    x = _as_float(val)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"Expected a number in [0, 1], got {x}")
    return x


def grid_shape(val) -> tuple[int, int]:
    """'10x10' / '10X5' / '10,5' / [10, 5]  ->  (rows, cols)."""
    if isinstance(val, Sequence) and not isinstance(val, str):
        parts = list(val)
    else:
        parts = str(val).lower().replace(",", "x").replace("×", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid grid shape '{val}', use ROWSxCOLS (e.g. 10x10)")
    return positive_int(parts[0]), positive_int(parts[1])


# ------------------------------------------------------------------- parser
def build_base_parser(description: str, prog: str | None = None) -> argparse.ArgumentParser:
    """Parser con los argumentos comunes a todos los ejercicios del TP4."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cfg = parser.add_argument_group("Configuration")
    cfg.add_argument("-c", "--config", type=str, default=None,
                     help="Path to a JSON/TOML configuration file. "
                          "Arguments passed explicitly on the CLI override the file.")

    gen = parser.add_argument_group("General")
    gen.add_argument("--seed", type=nullable_int, default=int(time.time()),
                     help="Random seed (use 'none' for a non-deterministic run).")

    out = parser.add_argument_group("Output & plots")
    out.add_argument("--output-dir", type=str, default="output",
                     help="Directory where figures and result tables are written.")

    return parser
