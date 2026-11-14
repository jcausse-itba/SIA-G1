from typing import Any
import numpy as np
from numpy.typing import NDArray
import rust_matmul

from .activation_functions.base_activation import ActivationFunction

class Layer:

    def __init__(
        self,
        in_dimm: int,
        out_dimm: int,
        activation_function: ActivationFunction,
        initialization: Any = None,
    ) -> None:
        self.W: NDArray[np.float64]
        self.b: NDArray[np.float64]
        self.input: NDArray[np.float64]
        self.z: NDArray[np.float64]

        if initialization is None:
            std = np.sqrt(2.0 / in_dimm)
            self.W = np.random.randn(in_dimm, out_dimm) * std
            self.b = np.zeros(out_dimm, dtype=np.float64)
        else:
            raise NotImplementedError("Only default He initialization implemented")

        self.activation_function: ActivationFunction = activation_function

    def feed_forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.input = x
        self.z = rust_matmul.matmul(x, self.W) + self.b
        return self.activation_function.compute(self.z)

    def backward(
        self, grad_output: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        grad_z = grad_output * self.activation_function.gradient(self.z)
        grad_W = rust_matmul.matmul(self.input.T, grad_z)
        grad_b = np.sum(grad_z, axis=0)

        grad_input = rust_matmul.matmul(grad_z, self.W.T)
        return grad_input, grad_W, grad_b