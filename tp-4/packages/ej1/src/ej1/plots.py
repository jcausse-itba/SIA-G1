"""Helpers de salida comunes (HTML de plotly + CSV)."""
import os
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go

# Escala divergente usada en todo el TP (mismo criterio que los scripts originales).
DIVERGING = ["#c0392b", "#f7f7f7", "#1e8449"]
SEQUENTIAL = "Viridis"


def out_dir(cfg: Dict[str, Any], sub: str) -> str:
    path = os.path.join(cfg["output_dir"], sub)
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig: go.Figure, directory: str, name: str) -> str:
    """Escribe el HTML; plotly.min.js se guarda una sola vez por carpeta (funciona offline)."""
    path = os.path.join(directory, f"{name}.html")
    fig.update_layout(template="plotly_white")
    fig.write_html(path, include_plotlyjs="directory")
    return path


def save_csv(df: pd.DataFrame, directory: str, name: str, index: bool = False) -> str:
    path = os.path.join(directory, f"{name}.csv")
    df.to_csv(path, index=index)
    return path


def short(feature: str) -> str:
    """Nombre corto para ejes: saca sufijos repetitivos."""
    for suf in ("_per_1000", "_pct", "_seconds"):
        feature = feature.replace(suf, "")
    return feature
