import numpy as np
from numpy.typing import NDArray
from .base import Loss


class BinaryCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-15) -> None:
        self.eps = eps

    def compute(self, y_pred: NDArray, y_true: NDArray) -> float:
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        )
        return float(loss)

    def gradient(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        grad = (y_pred - y_true) / (y_pred * (1.0 - y_pred))
        return grad / y_pred.size
