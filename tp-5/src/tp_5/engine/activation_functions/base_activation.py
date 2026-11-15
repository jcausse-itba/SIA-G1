from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class ActivationFunction(ABC):

    @abstractmethod
    def compute(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward: calculates f(x) for an input array."""
        pass

    @abstractmethod
    def gradient(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        """Backward: calculates f'(x) for an input array."""
        pass