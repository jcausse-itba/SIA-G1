"""Carga, selección de features y preprocesamiento (log1p + estandarización)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .cli import ID_COLUMNS


@dataclass
class Dataset:
    raw: pd.DataFrame            # CSV completo, sin tocar (se usa en el punto 4)
    ids: pd.DataFrame            # creator_id, name
    features: List[str]          # features que entran a PCA / Kohonen / Oja
    log_features: List[str]      # subconjunto de features con log1p
    transformed: pd.DataFrame    # features tras log1p (antes de estandarizar)
    mean: np.ndarray             # media de `transformed` (para proyectar creadores nuevos)
    std: np.ndarray              # desvío de `transformed`
    X: np.ndarray = field(repr=False)  # matriz estandarizada (n_creadores x n_features)

    @property
    def names(self) -> np.ndarray:
        return self.ids["name"].to_numpy()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Aplica el mismo preprocesamiento (con la media/std guardadas) a filas nuevas."""
        t = df[self.features].astype(float).copy()
        for c in self.log_features:
            t[c] = np.log1p(t[c])
        return (t.to_numpy() - self.mean) / self.std


def load_dataset(cfg: Dict[str, Any]) -> Dataset:
    raw = pd.read_csv(cfg["dataset_path"])
    excluded = set(cfg["exclude_features"] or [])
    features = [c for c in raw.columns
                if c not in ID_COLUMNS and c not in excluded and pd.api.types.is_numeric_dtype(raw[c])]
    log_features = [c for c in (cfg["log_features"] or []) if c in features]

    transformed = raw[features].astype(float).copy()
    for c in log_features:
        if (transformed[c] < 0).any():
            raise ValueError(f"log1p requires non-negative values; '{c}' has negatives.")
        transformed[c] = np.log1p(transformed[c])

    mean = transformed.mean().to_numpy()
    std = transformed.std(ddof=1).to_numpy()
    if np.any(std == 0):
        constant = [f for f, s in zip(features, std) if s == 0]
        raise ValueError(f"Constant features cannot be standardized: {constant}")
    X = (transformed.to_numpy() - mean) / std

    return Dataset(raw=raw, ids=raw[list(ID_COLUMNS)].copy(), features=features,
                   log_features=log_features, transformed=transformed, mean=mean, std=std, X=X)
