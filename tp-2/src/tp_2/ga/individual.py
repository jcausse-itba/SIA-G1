import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Triangle:
    """
    Representa un gen compuesto: 3 vértices normalizados y un color HCL.
    """
    vertices: List[Tuple[float, float]]
    color: Tuple[float, float, float, float]

    @classmethod
    def random_init(cls) -> 'Triangle':
        """Inicializa un triángulo completamente al azar."""
        vertices = [(random.random(), random.random()) for _ in range(3)]
        color = (
            random.uniform(0.0, 360.0), # Hue
            random.uniform(0.0, 130.0), # Chroma
            random.uniform(0.0, 100.0), # Lightness
            random.uniform(0.0, 1.0)    # Alpha
        )
        return cls(vertices, color)


@dataclass
class Individual:
    """
    Representa a un individuo de la población (la lista de N triángulos).
    """
    triangles: List[Triangle]
    fitness: Optional[float] = None  # Permite asignar None cuando el valor no fue evaluado

    @classmethod
    def random_init(cls, num_triangles: int) -> 'Individual':
        """Crea un individuo generando N triángulos al azar."""
        return cls([Triangle.random_init() for _ in range(num_triangles)])