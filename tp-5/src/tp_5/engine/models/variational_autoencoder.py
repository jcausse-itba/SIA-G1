from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..activation_functions.base_activation import ActivationFunction
from ..loss_functions.base import Loss
from ..loss_functions.elbo import ELBOLoss
from ..loss_functions.kl_divergence import KLDivergence
from ..optimizers.base import Optimizer
from ..variational_layer import VariationalLayer
from .autoencoder import Autoencoder
from .mlp import MLP


class VariationalAutoencoder(Autoencoder):
    """
    Variational Autoencoder built by injecting a VariationalLayer at the latent
    bottleneck and composing the reconstruction loss with KLDivergence via ELBOLoss.
    """

    def __init__(
        self,
        encoder_layers: list[int],
        hidden_activation: ActivationFunction,
        latent_activation: ActivationFunction,
        output_activation: ActivationFunction,
        loss_function: Loss,
        optimizer: Callable[..., Optimizer],
        kl_loss: KLDivergence,
        decoder_layers: list[int] | None = None,
        kl_weight: float = 0.001,
        seed: int = 42,
    ) -> None:
        super().__init__(
            encoder_layers=encoder_layers,
            hidden_activation=hidden_activation,
            latent_activation=latent_activation,
            output_activation=output_activation,
            loss_function=loss_function,
            optimizer=optimizer,
            decoder_layers=decoder_layers,
        )

        latent_in_dim = self.encoder_sizes[-2]
        latent_out_dim = self.encoder_sizes[-1]

        self.variational_layer = VariationalLayer(
            in_dimm=latent_in_dim,
            out_dimm=latent_out_dim,
            kl_loss=kl_loss,
            kl_weight=kl_weight,
            activation_function=latent_activation,
            seed=seed,
        )
        self.layers[self.latent_layer_idx] = self.variational_layer
        self.loss_fn = ELBOLoss(
            reconstruction_loss=loss_function,
            kl_loss=kl_loss,
            variational_layer=self.variational_layer,
            kl_weight=kl_weight,
        )

    def encode_distribution(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        prev_state = self.variational_layer.training
        self.variational_layer.training = False
        try:
            _ = self.encode(x)
            return self.variational_layer.mu, self.variational_layer.std
        finally:
            self.variational_layer.training = prev_state

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        batch_size: int | None = None,
        epochs: int = 1000,
        lr: float = 0.001,
        print_every: int = 100,
        epoch_callback: Callable[[int, MLP], None] | None = None,
        callback_epochs: set[int] | None = None,
    ) -> list[float]:
        target = X if y is None else y
        self.variational_layer.training = True
        try:
            return super().fit(
                X=X,
                y=target,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                print_every=print_every,
                epoch_callback=epoch_callback,
                callback_epochs=callback_epochs,
            )
        finally:
            self.variational_layer.training = False
