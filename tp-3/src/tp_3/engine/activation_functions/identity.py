import numpy as np
from numpy.typing import NDArray
from activation_functions.base_activation import ActivationFunction

class Identity(ActivationFunction):

    def compute(self, value: NDArray[np.float16]) -> NDArray[np.float16]:
        return value.astype(np.float16, copy=False)

    def gradient(self, value: NDArray[np.float16]) -> NDArray[np.float16]:
        return np.ones_like(value, dtype=np.float16)
