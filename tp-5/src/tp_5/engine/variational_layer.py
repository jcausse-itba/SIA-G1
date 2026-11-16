from typing import Any

import numpy as np
from numpy.typing import NDArray

from .activation_functions.base_activation import ActivationFunction
from .activation_functions.identity import Identity
from .layer import Layer
from .loss_functions.kl_divergence import KLDivergence


class VariationalLayer(Layer):
    """
    Reparameterization bottleneck layer:
        [mu, raw_std] = x @ W + b
        z = mu + std * eps,  eps ~ N(0, I)
    """

    def __init__(
        self,
        in_dimm: int,
        out_dimm: int,
        kl_loss: KLDivergence,
        kl_weight: float = 0.001,
        activation_function: ActivationFunction | None = None,
        min_std: float = 1e-2,
        max_std: float = 10.0,
        seed: int = 42,
        initialization: Any = None,
    ) -> None:
        super().__init__(
            in_dimm=in_dimm,
            out_dimm=out_dimm * 2,
            activation_function=activation_function or Identity(),
            initialization=initialization,
        )
        self.latent_dim = out_dimm
        self.kl_loss = kl_loss
        self.kl_weight = float(kl_weight)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.rng = np.random.default_rng(seed)
        self.training: bool = False

        self.mu: NDArray[np.float64] = np.array([])
        self.raw_std: NDArray[np.float64] = np.array([])
        self.std: NDArray[np.float64] = np.array([])
        self.eps: NDArray[np.float64] = np.array([])

    def feed_forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raw = super().feed_forward(x)
        self.mu = raw[:, : self.latent_dim]
        self.raw_std = raw[:, self.latent_dim :]
        self.std = np.clip(self.raw_std, self.min_std, self.max_std)

        if self.training:
            self.eps = self.rng.standard_normal(self.mu.shape)
            return self.mu + self.std * self.eps

        self.eps = np.zeros_like(self.mu)
        return self.mu

    def backward(
        self, grad_output: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        kl_grad_mu, kl_grad_std = self.kl_loss.gradient(self.mu, self.std)

        grad_mu = grad_output + self.kl_weight * kl_grad_mu
        grad_std = grad_output * self.eps + self.kl_weight * kl_grad_std
        grad_std = np.where(
            ((self.raw_std > self.min_std) | (grad_std < 0.0))
            & ((self.raw_std < self.max_std) | (grad_std > 0.0)),
             grad_std,
             0.0,
         )

        grad_raw = np.hstack([grad_mu, grad_std])
        return super().backward(grad_raw)
