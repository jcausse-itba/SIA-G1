import numpy as np
from numpy.typing import NDArray
from activation_functions.base_activation import ActivationFunction

class ReLU(ActivationFunction):

    def compute(self, value: NDArray) -> NDArray:
        return np.maximum(0.0, value)

    def gradient(self, value: NDArray) -> NDArray:
        return (value > 0.0).astype(value.dtype)
