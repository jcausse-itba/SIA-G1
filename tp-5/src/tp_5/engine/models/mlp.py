from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..activation_functions.base_activation import ActivationFunction
from ..layer import Layer
from ..loss_functions.base import Loss
from ..optimizers.base import Optimizer
from ..schedulers.base import Scheduler
from ..schedulers.constant import Constant
from .base import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        layer_sizes: list[int],
        input_activation: ActivationFunction,
        output_activation: ActivationFunction,
        loss_function: Loss,
        optimizer: Callable[..., Optimizer],
        lr_scheduler: Scheduler | None = None,
    ):
        """
        Initializes an MLP given layer dimensions and injected collaborators.
        """
        self.layers: list[Layer] = []
        self.loss_fn: Loss = loss_function
        self.optimizer: Callable[..., Optimizer] = optimizer
        self.lr_scheduler: Scheduler = lr_scheduler or Constant()

        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            is_last = i == len(layer_sizes) - 2

            if is_last and output_activation is not None:
                act = output_activation
            else:
                act = input_activation

            self.layers.append(Layer(in_dimm=in_dim, out_dimm=out_dim, activation_function=act))

    def forward(self, x: NDArray) -> NDArray:
        """Passes input through all layers sequentially."""
        out = x
        for layer in self.layers:
            out = layer.feed_forward(out)
        return out

    def backward(self, loss_gradient: NDArray) -> list[tuple[NDArray, NDArray]]:
        """Passes gradient backward through all layers in reverse order."""
        grads = []
        grad_input = loss_gradient

        for layer in reversed(self.layers):
            grad_input, grad_W, grad_b = layer.backward(grad_input)
            grads.append((grad_W, grad_b))

        grads.reverse()
        return grads

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        batch_size: int | None = None,
        epochs: int = 1000,
        lr: float = 0.001,
        print_every: int = 100,
        epoch_callback: Callable[[int, "MLP"], None] | None = None,
        callback_epochs: set[int] | None = None,
    ) -> list[float]:
        if len(X) == 0:
            raise ValueError("Training data must contain at least one sample.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        optimizers_W = [self.optimizer(lr=lr) for _ in self.layers]
        optimizers_b = [self.optimizer(lr=lr) for _ in self.layers]
        history = []
        current_batch_size = batch_size or len(X)
        is_full_batch = current_batch_size == len(X)
        rng = np.random.default_rng(42)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            current_lr = self.lr_scheduler.get_lr(lr, epoch, epochs)
            for opt_w, opt_b in zip(optimizers_W, optimizers_b):
                opt_w.lr = current_lr
                opt_b.lr = current_lr

            perm = None if is_full_batch else rng.permutation(len(X))

            for start in range(0, len(X), current_batch_size):
                if is_full_batch:
                    batch_X, batch_y = X, y
                else:
                    idx = perm[start : start + current_batch_size]
                    batch_X = X[idx] #TODO
                    batch_y = y[idx]

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
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {epoch_loss:.6f}")

            if epoch_callback is not None and (callback_epochs is None or epoch in callback_epochs):
                if epoch_callback(epoch, self):
                    break

        return history
