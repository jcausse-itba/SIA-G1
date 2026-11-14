from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from .base import Loss


class MeanSquaredError(Loss):
    def compute(self, y_pred: NDArray, y_true: NDArray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def gradient(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        return 2 * (y_pred - y_true) / y_pred.size
