import numpy as np
from numpy.typing import NDArray

from .base import Noise


class BitFlip(Noise):
    """Flips binary pixels (0 -> 1 or 1 -> 0) with probability `level`."""

    def __init__(self, level: float = 0.1) -> None:
        self.level = float(level)

    def corrupt(self, X: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        mask = rng.random(X.shape) < self.level
        return np.where(mask, 1.0 - X, X)
