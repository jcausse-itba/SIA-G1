from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class Loss(ABC):
    """Abstract Base Class for Loss Functions."""

    @abstractmethod
    def compute(self, y_pred: NDArray, y_true: NDArray) -> float:
        """
        Compute the scalar loss value.
        
        :param y_pred: Predictions output by the model.
        :param y_true: Ground truth target values.
        :return: Scalar loss value.
        """
        pass

    @abstractmethod
    def gradient(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        """
        Compute the derivative of the loss with respect to y_pred (dL/dy_pred).
        
        :param y_pred: Predictions output by the model.
        :param y_true: Ground truth target values.
        :return: Gradient array with the same shape as y_pred.
        """
        pass
