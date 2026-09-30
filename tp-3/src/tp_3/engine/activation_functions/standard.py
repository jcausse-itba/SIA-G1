import numpy as np
from numpy.typing import NDArray

from .base_activation import ActivationFunction


class Step(ActivationFunction):
    def compute(self, value: NDArray) -> NDArray:
        return np.where(value >= 0, 1.0, -1.0).astype(value.dtype)

    def gradient(self, value: NDArray) -> NDArray:
        return np.zeros_like(value)


class Sigmoid(ActivationFunction):
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def compute(self, value: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-self.beta * value))

    def gradient(self, value: NDArray) -> NDArray:
        output = self.compute(value)
        return self.beta * output * (1.0 - output)


class Tanh(ActivationFunction):
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def compute(self, value: NDArray) -> NDArray:
        return np.tanh(self.beta * value)

    def gradient(self, value: NDArray) -> NDArray:
        output = self.compute(value)
        return self.beta * (1.0 - output**2)