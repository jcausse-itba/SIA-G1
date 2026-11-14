import numpy as np
from numpy.typing import NDArray

from .base import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, weights: NDArray, gradients: NDArray) -> NDArray:
        return weights - self.lr * gradients