from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from ..activation_functions.base_activation import ActivationFunction
from ..loss_functions.base import Loss
from ..optimizers.base import Optimizer
from .mlp import MLP


class Autoencoder(MLP):
    """
    Standard Autoencoder inheriting training/saving mechanics from MLP while
    exposing `encode(X)` -> 2D latent space and `decode(Z)` -> reconstruction.
    """

    def __init__(
        self,
        encoder_layers: list[int],
        hidden_activation: ActivationFunction,
        latent_activation: ActivationFunction,
        output_activation: ActivationFunction,
        loss_function: Loss,
        optimizer: Callable[..., Optimizer],
        decoder_layers: list[int] | None = None,
    ) -> None:
        if len(encoder_layers) < 2:
            raise ValueError("encoder_layers must have at least [input_dim, latent_dim].")

        dec_layers = decoder_layers if decoder_layers is not None else list(reversed(encoder_layers[:-1]))

        super().__init__(
            layer_sizes=list(encoder_layers) + list(dec_layers),
            input_activation=hidden_activation,
            output_activation=output_activation,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        self.encoder_sizes = list(encoder_layers)
        self.decoder_sizes = list(dec_layers)
        self.latent_layer_idx = len(encoder_layers) - 2
        self.layers[self.latent_layer_idx].activation_function = latent_activation

    def encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Projects input patterns into the latent space (e.g. R^2)."""
        out = x
        for layer in self.layers[: self.latent_layer_idx + 1]:
            out = layer.feed_forward(out)
        return out

    def decode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generates/reconstructs patterns from latent space coordinates."""
        out = z
        for layer in self.layers[self.latent_layer_idx + 1 :]:
            out = layer.feed_forward(out)
        return out

    def evaluate_reconstruction(
        self,
        X_input: NDArray[np.float64],
        X_target: NDArray[np.float64] | None = None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Evaluates pixel-level reconstruction accuracy and latent encodings."""
        target = X_input if X_target is None else X_target
        probs = self.forward(X_input)
        reconstructed = (probs >= threshold).astype(np.float64)
        target_bin = (target >= threshold).astype(np.float64)

        pixel_errors = np.sum(np.abs(target_bin - reconstructed), axis=1).astype(int)
        loss = float(self.loss_fn.compute(probs, target))

        return {
            "loss": loss,
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
