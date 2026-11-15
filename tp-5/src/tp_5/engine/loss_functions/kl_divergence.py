import numpy as np
from numpy.typing import NDArray

from .base import Loss


class KLDivergence(Loss):
    """
    Kullback-Leibler Divergence loss against a standard normal prior N(0, I):
        D_KL = -0.5 * mean(1 + 2 * log(std) - mu^2 - std^2)
    """

    def compute(self, mu: NDArray[np.float64], std: NDArray[np.float64]) -> float:
        kl = -0.5 * np.mean(1.0 + 2.0 * np.log(std) - np.square(mu) - np.square(std))
        return float(kl)

    def gradient(
        self, mu: NDArray[np.float64], std: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        scale = 1.0 / mu.size
        grad_mu = scale * mu
        grad_std = scale * (std - 1.0 / std)
        return grad_mu, grad_std
