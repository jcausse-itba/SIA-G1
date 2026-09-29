from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from optimizers.base import Optimizer

class Momentum(Optimizer):
    """
    Momentum Optimizer.
    
    Formula:
        Δw(t+1) = -η * (∂E / ∂w) + α * Δw(t)
        w(t+1) = w(t) + Δw(t+1)
    """

    def __init__(self, lr: np.float16 = np.float16(0.01), alpha: np.float16 = np.float16(0.9)):
        """
        :param lr: Learning rate (η).
        :param alpha: Momentum coefficient (α), usually between 0.8 and 0.9.
        """
        self.lr = np.float64(lr)
        self.alpha = np.float64(alpha)
        self.delta_w: NDArray[np.float64] | None = None

    def update(self, weights: NDArray[np.float16], gradients: NDArray[np.float16]) -> NDArray[np.float16]:
        if self.delta_w is None:
            self.delta_w = np.zeros_like(weights, dtype=np.float64)

        self.delta_w = -self.lr * gradients.astype(np.float64) + self.alpha * self.delta_w
        return (weights.astype(np.float64) + self.delta_w).astype(np.float16)
