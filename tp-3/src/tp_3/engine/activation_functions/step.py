import numpy as np
from numpy.typing import NDArray

from .base_activation import ActivationFunction


class Step(ActivationFunction):
    def compute(self, value: NDArray) -> NDArray:
        return np.where(value >= 0, 1.0, -1.0).astype(value.dtype)

    def gradient(self, value: NDArray) -> NDArray:
        return np.zeros_like(value)
