import numpy as np
from numpy.typing import NDArray

from .base import Noise


class Masking(Noise):
    """Zeroes out active pixels with probability `level` (Salt/Dropout noise)."""

    def __init__(self, level: float = 0.15) -> None:
        self.level = float(level)

    def corrupt(self, X: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        keep = rng.random(X.shape) >= self.level
        return X * keep
