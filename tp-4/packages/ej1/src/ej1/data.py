"""Carga, selección de features y preprocesamiento (estandarización sin log1p)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .cli import ID_COLUMNS


@dataclass
class Dataset:
    raw: pd.DataFrame            # CSV completo, sin tocar
    ids: pd.DataFrame            # creator_id, name
    features: List[str]          # features que entran a PCA / Kohonen / Oja
    transformed: pd.DataFrame    # features seleccionadas en escala original
    mean: np.ndarray             # media de `transformed`
    std: np.ndarray              # desvío estándar de `transformed`
    X: np.ndarray = field(repr=False)  # matriz estandarizada Z-score (n_creadores x n_features)

    @property
    def names(self) -> np.ndarray:
        return self.ids["name"].to_numpy()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Aplica la estandarización (media y std guardadas) a observaciones nuevas."""
        t = df[self.features].astype(float).copy()
        return (t.to_numpy() - self.mean) / self.std


def load_dataset(cfg: Dict[str, Any]) -> Dataset:
    raw = pd.read_csv(cfg["dataset_path"])
    excluded = set(cfg["exclude_features"] or [])
    features = [c for c in raw.columns
                if c not in ID_COLUMNS and c not in excluded and pd.api.types.is_numeric_dtype(raw[c])]

    transformed = raw[features].astype(float).copy()

    mean = transformed.mean().to_numpy()
    std = transformed.std(ddof=1).to_numpy()
    if np.any(std == 0):
        constant = [f for f, s in zip(features, std) if s == 0]
        raise ValueError(f"Constant features cannot be standardized: {constant}")
    X = (transformed.to_numpy() - mean) / std

    return Dataset(raw=raw, ids=raw[list(ID_COLUMNS)].copy(), features=features,
                   transformed=transformed, mean=mean, std=std, X=X)