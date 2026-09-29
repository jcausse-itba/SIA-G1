from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from optimizers.base import Optimizer

class Adam(Optimizer):
    """
    Adam Optimizer (Kingma & Ba, 2015).
    
    Formula:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - α * m̂_t / (√(v̂_t) + ε)
    """

    def __init__(
        self,
        lr: np.float16 = np.float16(0.001),
        beta1: np.float16 = np.float16(0.9),
        beta2: np.float16 = np.float16(0.999),
        epsilon: np.float16 = np.float16(1e-8),
    ):
        """
        :param lr: Learning rate (α).
        :param beta1: Exponential decay rate for first moment vector (β1).
        :param beta2: Exponential decay rate for second moment vector (β2).
        :param epsilon: Small constant to prevent division by zero (ε).
        """
        self.lr = np.float64(lr)
        self.beta1 = np.float64(beta1)
        self.beta2 = np.float64(beta2)
        self.epsilon = np.float64(epsilon)

        self.m: NDArray[np.float64] | None = None
        self.v: NDArray[np.float64] | None = None
        self.t: int = 0

    def update(self, weights: NDArray[np.float16], gradients: NDArray[np.float16]) -> NDArray[np.float16]:
        if self.m is None or self.v is None:
            self.m = np.zeros_like(weights, dtype=np.float64)
            self.v = np.zeros_like(weights, dtype=np.float64)

        self.t += 1
        g_t = gradients.astype(np.float64)

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (np.float64(1.0) - self.beta1) * g_t
        self.v = self.beta2 * self.v + (np.float64(1.0) - self.beta2) * (g_t ** 2)

        # Compute bias-corrected estimates
        m_hat = self.m / (np.float64(1.0) - self.beta1 ** self.t)
        v_hat = self.v / (np.float64(1.0) - self.beta2 ** self.t)

        # Update parameter vector
        return (weights.astype(np.float64) - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)).astype(np.float16)
