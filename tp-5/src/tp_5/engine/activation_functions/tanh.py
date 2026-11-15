import numpy as np
from numpy.typing import NDArray

from .base_activation import ActivationFunction

class Tanh(ActivationFunction):
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def compute(self, value: NDArray) -> NDArray:
        return np.tanh(self.beta * value)

    def gradient(self, value: NDArray) -> NDArray:
        output = self.compute(value)
        return self.beta * (1.0 - output**2)
