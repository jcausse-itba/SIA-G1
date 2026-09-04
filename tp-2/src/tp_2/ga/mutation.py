import random
import copy
from typing import List, Tuple
from tp_2.ga.individual import Individual, Triangle

class Mutation:
    """Métodos de mutación para individuos y triángulos."""

    @staticmethod
    def _mutate_component(val: float, min_val: float, max_val: float, delta_ratio: float = 0.1) -> float:
        """Aplica un ruido gaussiano o uniforme a un componente escalar acotado."""
        scale = (max_val - min_val) * delta_ratio
        new_val = val + random.gauss(0, scale)
        return max(min_val, min(max_val, new_val))

    @classmethod
    def mutate_triangle(cls, tri: Triangle, delta_ratio: float = 0.1) -> Triangle:
        """Muta vértices y color de un triángulo."""
        new_vertices = []
        for x, y in tri.vertices:
            nx = cls._mutate_component(x, 0.0, 1.0, delta_ratio)
            ny = cls._mutate_component(y, 0.0, 1.0, delta_ratio)
            new_vertices.append((nx, ny))

        h, s, l, a = tri.color
        nh = cls._mutate_component(h, 0.0, 360.0, delta_ratio)
        ns = cls._mutate_component(s, 0.0, 100.0, delta_ratio)
        nl = cls._mutate_component(l, 0.0, 100.0, delta_ratio)
        na = cls._mutate_component(a, 0.0, 1.0, delta_ratio)

        return Triangle(new_vertices, (nh, ns, nl, na))

    @classmethod
    def apply(cls, ind: Individual, p_mut: float, method: str, gen: int = 1, max_gen: int = 100) -> Individual:
        """
        Aplica mutación al individuo según el método especificado.
        Non-uniform mutation: delta_ratio disminuye a medida que avanzan las generaciones.
        """
        new_ind = copy.deepcopy(ind)
        # Factor no uniforme de ajuste fino (disminuye con las generaciones)
        delta_ratio = max(0.01, 0.2 * (1.0 - (gen / max_gen)))

        if random.random() > p_mut:
            return new_ind

        n = len(new_ind.triangles)

        if method == "single_gene":
            idx = random.randint(0, n - 1)
            new_ind.triangles[idx] = cls.mutate_triangle(new_ind.triangles[idx], delta_ratio)

        elif method in ("multigene_limited", "multigene_uniform"):
            for i in range(n):
                if random.random() < 0.2:  # 20% de probabilidad por triángulo
                    new_ind.triangles[i] = cls.mutate_triangle(new_ind.triangles[i], delta_ratio)

        elif method == "complete":
            for i in range(n):
                new_ind.triangles[i] = cls.mutate_triangle(new_ind.triangles[i], delta_ratio)

        return new_ind