import numpy as np
from numpy.typing import NDArray
from activation_functions.base_activation import ActivationFunction

class ReLU(ActivationFunction):

    def compute(self, value: NDArray[np.float16]) -> NDArray[np.float16]:
        return np.maximum(np.float16(0.0), value)

    def gradient(self, value: NDArray[np.float16]) -> NDArray[np.float16]:
        return (value > np.float16(0.0)).astype(np.float16)
