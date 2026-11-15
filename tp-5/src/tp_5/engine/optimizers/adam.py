import numpy as np
from numpy.typing import NDArray
from .base import Optimizer

class Adam(Optimizer):
    """
    Adam Optimizer (Kingma & Ba, 2015).
    
    Formula:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - α * m̂_t / (√(v̂_t) + ε)
        
    Proof of Scalar Optimization Identity:
            θ_t = θ_{t-1} - α * [m_t / (1 - β1^t)] / [√(v_t / (1 - β2^t)) + ε]
            θ_t = θ_{t-1} - [α * √(1 - β2^t) / (1 - β1^t)] * m_t / [√(v_t) + ε * √(1 - β2^t)]
            α_t = α * √(1 - β2^t) / (1 - β1^t)
            ε_t = ε * √(1 - β2^t)
            
            θ_t = θ_{t-1} - α_t * m_t / (√(v_t) + ε_t)
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

        self.m: NDArray[np.float64] | None = None
        self.v: NDArray[np.float64] | None = None
        self.t: int = 0

    def update(self, weights: NDArray, gradients: NDArray) -> NDArray:
        if self.m is None or self.v is None:
            self.m = np.zeros_like(weights, dtype=np.float64)
            self.v = np.zeros_like(weights, dtype=np.float64)

        self.t += 1
        g_t = gradients.astype(np.float64, copy=False)

        # Update biased moment estimates
        self.m *= self.beta1
        self.m += (1.0 - self.beta1) * g_t
        
        self.v *= self.beta2
        self.v += (1.0 - self.beta2) * np.square(g_t)

        beta1_t = 1.0 - self.beta1 ** self.t
        beta2_t = 1.0 - self.beta2 ** self.t
        lr_t = self.lr * (np.sqrt(beta2_t) / beta1_t)
        eps_t = self.epsilon * np.sqrt(beta2_t)

        # Parameter update
        step = np.sqrt(self.v)
        step += eps_t
        np.divide(self.m, step, out=step)
        step *= lr_t

        updated_weights = weights - step
        return updated_weights.astype(weights.dtype, copy=False)
