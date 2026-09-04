import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Triangle:
    """
    Representa un gen compuesto: 3 vértices normalizados y un color HSLuv.
    """
    # Vértices: [(x1,y1), (x2,y2), (x3,y3)] normalizados en [0.0, 1.0]
    vertices: List[Tuple[float, float]]
    
    # Color HSLuv+Alpha: (H, S, L, A) 
    # Rangos: H[0,360], S[0,100], L[0,100], A[0.0, 1.0]
    color: Tuple[float, float, float, float]

    @classmethod
    def random_init(cls) -> 'Triangle':
        """Inicializa un triángulo completamente al azar."""
        vertices = [(random.random(), random.random()) for _ in range(3)]
        color = (
            random.uniform(0.0, 360.0), # Hue
            random.uniform(0.0, 100.0), # Saturation
            random.uniform(0.0, 100.0), # Lightness
            random.uniform(0.0, 1.0)    # Alpha (Translucidez)
        )
        return cls(vertices, color)


@dataclass
class Individual:
    """
    Representa a un individuo de la población (la lista de N triángulos).
    """
    triangles: List[Triangle]
    fitness: float = None # Se calcula y asigna en la fase de evaluación

    @classmethod
    def random_init(cls, num_triangles: int) -> 'Individual':
        """Crea un individuo generando N triángulos al azar."""
        return cls([Triangle.random_init() for _ in range(num_triangles)])