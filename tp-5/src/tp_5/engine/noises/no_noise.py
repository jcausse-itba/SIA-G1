import numpy as np
from numpy.typing import NDArray

from .base import Noise


class NoNoise(Noise):
    def corrupt(self, X: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        return X
