"""Helpers de salida comunes (HTML de plotly + CSV) para el Ejercicio 2 (Hopfield)."""

__all__ = [
    "DIVERGING",
    "SEQUENTIAL",
    "BINARY_SCALE",
    "out_dir",
    "save_fig",
    "save_csv",
    "format_ascii_grid",
]

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Escalas de color consistentes con el resto del TP4.
DIVERGING = ["#c0392b", "#f7f7f7", "#1e8449"]
SEQUENTIAL = "Viridis"
BINARY_SCALE = [[0.0, "#f5f6fa"], [1.0, "#2c3e50"]]


def out_dir(cfg: Dict[str, Any], sub: str = "") -> str:
    """Crea y devuelve el subdirectorio de salida dentro de cfg['output_dir']."""
    path = os.path.join(cfg["output_dir"], sub) if sub else cfg["output_dir"]
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig: go.Figure, directory: str, name: str) -> str:
    """Escribe el HTML interactivo; plotly.min.js se guarda una sola vez por carpeta."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.html")
    fig.update_layout(template="plotly_white")
    fig.write_html(path, include_plotlyjs="directory")
    return path


def save_csv(df: pd.DataFrame, directory: str, name: str, index: bool = False) -> str:
    """Escribe un DataFrame a CSV dentro de directory."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.csv")
    df.to_csv(path, index=index)
    return path


def format_ascii_grid(vec: np.ndarray, shape: tuple[int, int] = (5, 5)) -> str:
    """Representación en texto (* para +1, espacio para -1) como en el enunciado."""
    mat = vec.reshape(shape)
    lines = []
    for row in mat:
        lines.append(" ".join("*" if val > 0 else " " for val in row))
    return "\n".join(lines)
