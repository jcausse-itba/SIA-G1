import numpy as np
from numpy.typing import NDArray
from .base_activation import ActivationFunction

class Identity(ActivationFunction):

    def compute(self, value: NDArray) -> NDArray:
        return value

    def gradient(self, value: NDArray) -> NDArray:
        return np.ones_like(value)
