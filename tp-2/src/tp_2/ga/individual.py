import numpy as np
from typing import Optional

class Individual:
    """
    Representa a un individuo de la población, vectorizado vía NumPy.
    El genoma es una matriz (N, 10) donde N = número de triángulos.
    Columnas: x1, y1, x2, y2, x3, y3, h, c, l, alpha
    """
    def __init__(self, genome: np.ndarray, fitness: Optional[float] = None):
        self.genome = genome
        self.fitness = fitness

    @classmethod
    def random_init(cls, num_triangles: int) -> 'Individual':
        """Crea un individuo generando N triángulos al azar en un array NumPy contiguo."""
        genome = np.empty((num_triangles, 10), dtype=np.float32)
        genome[:, 0:6] = np.random.rand(num_triangles, 6)           # x1, y1, x2, y2, x3, y3
        genome[:, 6] = np.random.uniform(0.0, 360.0, num_triangles) # Hue
        genome[:, 7] = np.random.uniform(0.0, 130.0, num_triangles) # Chroma
        genome[:, 8] = np.random.uniform(0.0, 100.0, num_triangles) # Lightness
        genome[:, 9] = np.random.uniform(0.05, 1.0, num_triangles)  # Alpha
        return cls(genome)

    def __deepcopy__(self, memo=None) -> 'Individual':
        """
        Bypasses Python's slow deepcopy introspection logic.
        Clones the contiguous memory block at native C-level speed.
        """
        return Individual(self.genome.copy(), self.fitness)
