from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Optimizer(ABC):

    @abstractmethod
    def update(self, weights: NDArray[np.float16], gradients: NDArray[np.float16]) -> NDArray[np.float16]:
        """
        Compute and return the updated parameter array using the given gradients.

        :param weights: Current parameter values.
        :param gradients: Gradients of the loss with respect to these parameters.
        :return: Updated parameter values with the same shape as `weights`.
        """
        pass
