from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from ..activation_functions.base_activation import ActivationFunction
from ..activation_functions.identity import Identity
from ..layer import Layer
from ..loss_functions.base import Loss
from ..optimizers.base import Optimizer
from .base import BaseModel
from .mlp import MLP


class VariationalAutoencoder(BaseModel):
    """
    Variational Autoencoder (VAE) using the reparameterization trick:
        z = mu + std * eps,  eps ~ N(0, I)
    """

    def __init__(
        self,
        encoder_layers: list[int],
        hidden_activation: ActivationFunction,
        output_activation: ActivationFunction,
        loss_function: Loss,
        optimizer: Callable[..., Optimizer],
        decoder_layers: list[int] | None = None,
        kl_weight: float = 0.001,
        seed: int = 42,
    ) -> None:
        if len(encoder_layers) < 2:
            raise ValueError("encoder_layers must have at least [input_dim, ..., latent_dim].")

        self.latent_dim = encoder_layers[-1]
        enc_backbone_sizes = encoder_layers[:-1]
        dec_sizes = [self.latent_dim] + list(decoder_layers) if decoder_layers else [self.latent_dim] + list(reversed(enc_backbone_sizes))

        self.encoder_backbone = MLP(
            layer_sizes=enc_backbone_sizes,
            input_activation=hidden_activation,
            output_activation=hidden_activation,
            loss_function=loss_function,
            optimizer=optimizer,
        )
        last_enc_dim = enc_backbone_sizes[-1]
        self.mu_layer = Layer(last_enc_dim, self.latent_dim, Identity())
        self.std_layer = Layer(last_enc_dim, self.latent_dim, Identity())

        self.decoder_net = MLP(
            layer_sizes=dec_sizes,
            input_activation=hidden_activation,
            output_activation=output_activation,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.kl_weight = float(kl_weight)
        self.rng = np.random.default_rng(seed)

        self._mu: NDArray[np.float64] = np.array([])
        self._std: NDArray[np.float64] = np.array([])
        self._eps: NDArray[np.float64] = np.array([])

    def encode_distribution(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        h = self.encoder_backbone.forward(x)
        mu = self.mu_layer.feed_forward(h)
        # Std must be > 0. Clip at 1e-8 to avoid log(0) and division by zero in gradients
        std = np.clip(self.std_layer.feed_forward(h), 1e-8, 10.0)
        return mu, std

    def encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        mu, _ = self.encode_distribution(x)
        return mu

    def decode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.decoder_net.forward(z)

    def forward(self, x: NDArray[np.float64], training: bool = False) -> NDArray[np.float64]:
        self._mu, self._std = self.encode_distribution(x)
        if training:
            self._eps = self.rng.standard_normal(self._mu.shape)
            z = self._mu + self._std * self._eps
        else:
            z = self._mu
        return self.decode(z)

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        batch_size: int | None = None,
        epochs: int = 1000,
        lr: float = 0.001,
        print_every: int = 100,
        epoch_callback: Callable[[int, Any], None] | None = None,
        callback_epochs: set[int] | None = None,
    ) -> list[float]:
        target = X if y is None else y
        all_layers = [
            *self.encoder_backbone.layers,
            self.mu_layer,
            self.std_layer,
            *self.decoder_net.layers,
        ]
        opts_W = [self.optimizer(lr=lr) for _ in all_layers]
        opts_b = [self.optimizer(lr=lr) for _ in all_layers]

        history: list[float] = []
        current_batch_size = batch_size or len(X)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for start in range(0, len(X), current_batch_size):
                batch_X = X[start : start + current_batch_size]
                batch_y = target[start : start + current_batch_size]
                n_samples = len(batch_X)

                y_pred = self.forward(batch_X, training=True)
                recon_loss = float(self.loss_fn.compute(y_pred, batch_y))
                
                # KL Divergence formula calculated directly from standard deviation: 
                # -0.5 * (1 + 2*log(std) - mu^2 - std^2)
                kl_loss = float(
                    -0.5 * np.mean(1.0 + 2.0 * np.log(self._std) - np.square(self._mu) - np.square(self._std))
                )
                total_loss = recon_loss + self.kl_weight * kl_loss
                epoch_loss += total_loss * n_samples

                # Backward pass through decoder
                grad_recon = self.loss_fn.gradient(y_pred, batch_y)
                grad_z = grad_recon
                dec_grads = []
                for layer in reversed(self.decoder_net.layers):
                    grad_z, gW, gb = layer.backward(grad_z)
                    dec_grads.append((gW, gb))
                dec_grads.reverse()

                # Reparameterization + KL gradients directly w.r.t standard deviation
                kl_scale = self.kl_weight / self._mu.size
                grad_mu = grad_z + kl_scale * self._mu
                # Derivative of -0.5*(1 + 2*log(std) - mu^2 - std^2) w.r.t std is (std - 1/std)
                grad_std = grad_z * self._eps + kl_scale * (self._std - 1.0 / self._std)

                grad_h_mu, gW_mu, gb_mu = self.mu_layer.backward(grad_mu)
                grad_h_std, gW_std, gb_std = self.std_layer.backward(grad_std)
                enc_grads = self.encoder_backbone.backward(grad_h_mu + grad_h_std)

                all_grads = [*enc_grads, (gW_mu, gb_mu), (gW_std, gb_std), *dec_grads]
                for idx, layer in enumerate(all_layers):
                    gW, gb = all_grads[idx]
                    layer.W = opts_W[idx].update(layer.W, gW)
                    layer.b = opts_b[idx].update(layer.b, gb)

            epoch_loss /= len(X)
            history.append(epoch_loss)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{epochs} | VAE Total Loss: {epoch_loss:.6f}")

            if epoch_callback is not None and (callback_epochs is None or epoch in callback_epochs):
                if epoch_callback(epoch, self):
                    break

        return history

    def evaluate_reconstruction(
        self,
        X_input: NDArray[np.float64],
        X_target: NDArray[np.float64] | None = None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        target = X_input if X_target is None else X_target
        probs = self.forward(X_input, training=False)
        reconstructed = (probs >= threshold).astype(np.float64)
        target_bin = (target >= threshold).astype(np.float64)
        pixel_errors = np.sum(np.abs(target_bin - reconstructed), axis=1).astype(int)

        return {
            "loss": float(self.loss_fn.compute(probs, target)),
            "max_pixel_error": int(np.max(pixel_errors)),
            "mean_pixel_error": float(np.mean(pixel_errors)),
            "total_wrong_pixels": int(np.sum(pixel_errors)),
            "patterns_le_1_error": int(np.sum(pixel_errors <= 1)),
            "perfect_patterns": int(np.sum(pixel_errors == 0)),
            "num_patterns": int(len(target)),
            "pixel_errors": pixel_errors,
            "reconstructed_probs": probs,
            "reconstructed_bin": reconstructed,
            "latent_coords": self.encode(X_input),
        }
