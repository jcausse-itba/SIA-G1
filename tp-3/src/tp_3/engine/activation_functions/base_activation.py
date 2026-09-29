from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class ActivationFunction(ABC):

    @abstractmethod
    def compute(self, value: NDArray[np.float16]) -> NDArray[np.float16]:
        """Forward: calculates f(x) for an input array."""
        pass

    @abstractmethod
    def gradient(self, value: NDArray[np.float16]) -> NDArray[np.float16]:
        """Backward: calculates f'(x) for an input array."""
        pass
