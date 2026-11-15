import numpy as np
from numpy.typing import NDArray

from .base import Noise


class Gaussian(Noise):
    """Adds N(0, sigma) noise and clips back to [0, 1]."""

    def __init__(self, level: float = 0.2) -> None:
        self.level = float(level)

    def corrupt(self, X: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        return np.clip(X + rng.normal(0.0, self.level, X.shape), 0.0, 1.0)
