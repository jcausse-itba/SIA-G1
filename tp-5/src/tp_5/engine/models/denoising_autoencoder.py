from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..activation_functions.base_activation import ActivationFunction
from ..loss_functions.base import Loss
from ..noises.base import Noise
from ..schedulers.base import Scheduler
from ..optimizers.base import Optimizer
from .autoencoder import Autoencoder
from .mlp import MLP


class DenoisingAutoencoder(Autoencoder):
    """Corrupts input patterns dynamically during training while targeting clean inputs."""

    def __init__(
        self,
        encoder_layers: list[int],
        hidden_activation: ActivationFunction,
        latent_activation: ActivationFunction,
        output_activation: ActivationFunction,
        loss_function: Loss,
        optimizer: Callable[..., Optimizer],
        noise_strategy: Noise,
        lr_scheduler: Scheduler | None = None,
        quantization_scale: float = 1.0,
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
            lr_scheduler=lr_scheduler,
            quantization_scale=quantization_scale,
        )
        self.noise_strategy = noise_strategy
        self.rng = np.random.default_rng(seed)

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        batch_size: int | None = None,
        epochs: int = 1000,
        lr: float = 0.001,
        print_every: int = 100,
        epoch_callback: Callable[[int, "MLP"], None] | None = None,
        callback_epochs: set[int] | None = None,
    ) -> list[float]:
        clean_target = X if y is None else y
        optimizers_W = [self.optimizer(lr=lr) for _ in self.layers]
        optimizers_b = [self.optimizer(lr=lr) for _ in self.layers]
        history: list[float] = []
        current_batch_size = batch_size or len(X)

        for epoch in range(1, epochs + 1):
            current_lr = self.lr_scheduler.get_lr(lr, epoch, epochs)
            for opt_w, opt_b in zip(optimizers_W, optimizers_b):
                opt_w.lr = current_lr #TODO
                opt_b.lr = current_lr
            epoch_loss = 0.0
            noisy_X = self.noise_strategy.corrupt(X, self.rng)

            for start in range(0, len(X), current_batch_size):
                batch_X = noisy_X[start : start + current_batch_size]
                batch_y = clean_target[start : start + current_batch_size]

                y_pred = self.forward(batch_X)
                epoch_loss += float(self.loss_fn.compute(y_pred, batch_y)) * len(batch_X)

                loss_grad = self.loss_fn.gradient(y_pred, batch_y)
                grads = self.backward(loss_grad)

                for i, layer in enumerate(self.layers):
                    grad_W, grad_b = grads[i]
                    layer.W = optimizers_W[i].update(layer.W, grad_W)
                    layer.b = optimizers_b[i].update(layer.b, grad_b)

            epoch_loss /= len(X)
            history.append(epoch_loss)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{epochs} | Denoising Loss: {epoch_loss:.6f}")

            if epoch_callback is not None and (callback_epochs is None or epoch in callback_epochs):
                if epoch_callback(epoch, self):
                    break

        return history
