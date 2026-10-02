from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from .base import Optimizer

class RMSProp(Optimizer):
    """
    RMSProp Optimizer.
    
    Formula:
        g_t = ∂E / ∂w
        S_t = γ * S_{t-1} + (1 - γ) * g_t^2
        Δw = - (η / √(S_t + ε)) * g_t
    """

    def __init__(
        self,
        lr: np.float16 = np.float16(0.001),
        gamma: np.float16 = np.float16(0.9),
        epsilon: np.float16 = np.float16(1e-8),
    ):
        """
        :param lr: Learning rate (η).
        :param gamma: Decay factor for the moving average of squared gradients (γ).
        :param epsilon: Small constant to prevent division by zero (ε).
        """
        self.lr = np.float64(lr)
        self.gamma = np.float64(gamma)
        self.epsilon = np.float64(epsilon)
        self.S: NDArray[np.float64] | None = None

    def update(self, weights: NDArray[np.float16], gradients: NDArray[np.float16]) -> NDArray[np.float16]:
        if self.S is None:
            self.S = np.zeros_like(weights, dtype=np.float64)

        g_t = gradients.astype(np.float64)
        self.S = self.gamma * self.S + (np.float64(1.0) - self.gamma) * (g_t ** 2)

        delta_w = -(self.lr / np.sqrt(self.S + self.epsilon)) * g_t
        return (weights.astype(np.float64) + delta_w).astype(np.float16)
