from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Noise(ABC):
    @abstractmethod
    def corrupt(self, X: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        pass
