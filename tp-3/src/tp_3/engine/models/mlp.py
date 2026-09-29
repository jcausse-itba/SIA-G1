import numpy as np
from numpy.typing import NDArray
from layer import Layer
from activation_functions.base_activation import ActivationFunction
from optimizers.adam import Adam

class MLP:
    def __init__(self, layer_sizes: list[int], input_activation: ActivationFunction, output_activation: ActivationFunction, loss_function):
        """
        Initializes an MLP given layer dimensions.
        """
        self.layers: list[Layer] = []
        self.loss_fn = loss_function
        
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            is_last = (i == len(layer_sizes) - 2)

            # Use output_activation for final layer if specified, else ReLU
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
            # Insert at beginning to preserve layer order [layer_0, layer_1, ...]
            grads.insert(0, (grad_W, grad_b))

        return grads

    def fit(
        self, 
        X: NDArray, 
        y: NDArray, 
        epochs: int = 1000, 
        lr: float = 0.001, 
        print_every: int = 100
    ):

        # Each weight matrix and bias vector MUST have its own Adam optimizer instance 
        # to correctly preserve moment states (m and v).
        optimizers_W = [Adam(lr=lr) for _ in self.layers]
        optimizers_b = [Adam(lr=lr) for _ in self.layers]

        for epoch in range(1, epochs + 1):
            # 1. Forward Pass
            y_pred = self.forward(X)

            # 2. Calculate Loss & Initial Gradient
            loss = self.loss_fn.compute(y_pred, y)
            loss_grad =  self.loss_fn.gradient(y_pred, y)

            # 3. Backpropagation
            grads = self.backward(loss_grad)

            # 4. Update Parameters with Adam
            for i, layer in enumerate(self.layers):
                grad_W, grad_b = grads[i]
                layer.W = optimizers_W[i].update(layer.W, grad_W)
                layer.b = optimizers_b[i].update(layer.b, grad_b)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f}")
