import copy
import random
from typing import List, Tuple

from tp_2.ga.individual import Individual, Triangle

_RANGES = [
    (0.0, 1.0),    # x1
    (0.0, 1.0),    # y1
    (0.0, 1.0),    # x2
    (0.0, 1.0),    # y2
    (0.0, 1.0),    # x3
    (0.0, 1.0),    # y3
    (0.0, 360.0),  # H
    (0.0, 100.0),  # S
    (0.0, 100.0),  # L
    (0.05, 1.0),   # A
]
_N_COMPONENTS = len(_RANGES)


def _get_components(tri: Triangle) -> List[float]:
    x1, y1 = tri.vertices[0]
    x2, y2 = tri.vertices[1]
    x3, y3 = tri.vertices[2]
    h, s, l, a = tri.color
    return [x1, y1, x2, y2, x3, y3, h, s, l, a]


def _set_components(components: List[float]) -> Triangle:
    x1, y1, x2, y2, x3, y3, h, s, l, a = components
    # Clamp al rango válido de cada componente
    clamped = [
        max(lo, min(hi, v)) for v, (lo, hi) in zip(components, _RANGES)
    ]
    x1, y1, x2, y2, x3, y3, h, s, l, a = clamped
    return Triangle(
        vertices=[(x1, y1), (x2, y2), (x3, y3)],
        color=(h % 360.0, s, l, a),
    )


def _perturb_gaussian(val: float, lo: float, hi: float, scale: float) -> float:
    return max(lo, min(hi, val + random.gauss(0, (hi - lo) * scale)))


def _perturb_uniform(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


class Mutation:

    @staticmethod
    def single_gene(
        ind: Individual,
        p_ind: float,
        scale: float = 0.1,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        new_ind = copy.deepcopy(ind)
        idx = random.randrange(len(new_ind.triangles))
        comp = _get_components(new_ind.triangles[idx])
        c = random.randrange(_N_COMPONENTS)
        lo, hi = _RANGES[c]
        comp[c] = _perturb_gaussian(comp[c], lo, hi, scale)
        new_ind.triangles[idx] = _set_components(comp)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def multigene(
        ind: Individual,
        p_ind: float,
        k: int = 0,
        p_comp: float = 0.2,
        scale: float = 0.1,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        new_ind = copy.deepcopy(ind)
        n = len(new_ind.triangles)
        k_actual = k if k > 0 else max(1, -(-n // 4))   # ceil(N/4)
        k_actual = min(k_actual, n)

        indices = random.sample(range(n), k_actual)
        for i in indices:
            comp = _get_components(new_ind.triangles[i])
            mutated = False
            for c in range(_N_COMPONENTS):
                if random.random() < p_comp:
                    lo, hi = _RANGES[c]
                    comp[c] = _perturb_gaussian(comp[c], lo, hi, scale)
                    mutated = True
            if mutated:
                new_ind.triangles[i] = _set_components(comp)
        new_ind.fitness = None
        return new_ind
    @staticmethod
    def uniform(
        ind: Individual,
        p_ind: float,
        p_tri: float = 0.2,
        p_comp: float = 0.15,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        new_ind = copy.deepcopy(ind)
        for i, tri in enumerate(new_ind.triangles):
            if random.random() > p_tri:
                continue
            comp = _get_components(tri)
            mutated = False
            for c in range(_N_COMPONENTS):
                if random.random() < p_comp:
                    lo, hi = _RANGES[c]
                    comp[c] = _perturb_uniform(lo, hi)
                    mutated = True
            if mutated:
                new_ind.triangles[i] = _set_components(comp)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def non_uniform(
        ind: Individual,
        p_ind: float,
        p_tri: float = 0.3,
        p_comp: float = 0.2,
        generation: int = 1,
        max_generations: int = 1000,
        min_scale: float = 0.005,
        max_scale: float = 0.15,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        t = min(generation / max(max_generations, 1), 1.0)
        scale = max_scale * (1.0 - t) + min_scale * t

        new_ind = copy.deepcopy(ind)
        for i, tri in enumerate(new_ind.triangles):
            if random.random() > p_tri:
                continue
            comp = _get_components(tri)
            mutated = False
            for c in range(_N_COMPONENTS):
                if random.random() < p_comp:
                    lo, hi = _RANGES[c]
                    comp[c] = _perturb_gaussian(comp[c], lo, hi, scale)
                    mutated = True
            if mutated:
                new_ind.triangles[i] = _set_components(comp)
        new_ind.fitness = None
        return new_ind
    @staticmethod
    def apply(
        ind: Individual,
        p_ind: float,
        method: str,
        generation: int = 1,
        max_generations: int = 1000,
        p_tri: float = 0.3,
        p_comp: float = 0.2,
    ) -> Individual:
        if method == "single_gene":
            return Mutation.single_gene(ind, p_ind)
        elif method == "multigene":
            # k=0 → ceil(N/4) automático
            return Mutation.multigene(ind, p_ind, p_comp=p_comp)
        elif method == "uniform":
            return Mutation.uniform(ind, p_ind, p_tri=p_tri, p_comp=p_comp)
        elif method == "non_uniform":
            return Mutation.non_uniform(
                ind, p_ind, p_tri=p_tri, p_comp=p_comp,
                generation=generation, max_generations=max_generations,
            )
        else:
            return Mutation.multigene(ind, p_ind)