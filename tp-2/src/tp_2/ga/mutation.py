import copy
import math
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
    (0.0, 130.0),  # C
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
    h = h % 360.0
    
    clamped = [
        max(lo, min(hi, v)) for v, (lo, hi) in zip(components[7:], _RANGES[7:])
    ]
    s, l, a = clamped
    
    v_clamped = [max(0.0, min(1.0, v)) for v in [x1, y1, x2, y2, x3, y3]]
    
    return Triangle(
        vertices=[(v_clamped[0], v_clamped[1]), (v_clamped[2], v_clamped[3]), (v_clamped[4], v_clamped[5])],
        color=(h, s, l, a),
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
    def llm_de_non_linear_differential(
        ind: Individual,
        p_ind: float,
        p_tri: float = 0.3,
        p_comp: float = 0.2,
        generation: int = 1,
        max_generations: int = 1000,
        f_base: float = 0.5,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        t = min(generation / max(max_generations, 1), 1.0)
        # Factor de escala no lineal F con modulación sinusoidal y decaimiento cuadrático
        f_scale = f_base * (1.0 - t**2) * (0.8 + 0.2 * math.cos(math.pi * t))

        new_ind = copy.deepcopy(ind)
        n_triangles = len(new_ind.triangles)

        for i, tri in enumerate(new_ind.triangles):
            if random.random() > p_tri:
                continue
            comp = _get_components(tri)
            mutated = False

            # Triángulos de referencia para vector diferencial dentro del individuo
            r1_idx, r2_idx = (
                random.sample(range(n_triangles), 2) if n_triangles >= 2 else (i, i)
            )
            comp_r1 = _get_components(ind.triangles[r1_idx])
            comp_r2 = _get_components(ind.triangles[r2_idx])

            for c in range(_N_COMPONENTS):
                if random.random() < p_comp:
                    lo, hi = _RANGES[c]
                    diff = comp_r1[c] - comp_r2[c]
                    delta = f_scale * diff + random.gauss(0, (hi - lo) * 0.02 * (1.0 - t))
                    comp[c] = max(lo, min(hi, comp[c] + delta))
                    mutated = True
            if mutated:
                new_ind.triangles[i] = _set_components(comp)

        new_ind.fitness = None
        return new_ind

    @staticmethod
    def reevo_adaptive_neighborhood(
        ind: Individual,
        p_ind: float,
        p_tri: float = 0.3,
        p_comp: float = 0.2,
        generation: int = 1,
        max_generations: int = 1000,
        initial_radius: float = 0.25,
        alpha: float = 2.0,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        t = min(generation / max(max_generations, 1), 1.0)
        # Adaptación no lineal del radio del vecindario
        radius = initial_radius * ((1.0 - t) ** alpha) + 0.005

        new_ind = copy.deepcopy(ind)
        for i, tri in enumerate(new_ind.triangles):
            if random.random() > p_tri:
                continue
            comp = _get_components(tri)
            mutated = False
            for c in range(_N_COMPONENTS):
                if random.random() < p_comp:
                    lo, hi = _RANGES[c]
                    scale = radius * (hi - lo)
                    u = random.random() - 0.5
                    step = math.tan(math.pi * u * 0.45) * scale
                    comp[c] = max(lo, min(hi, comp[c] + step))
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
        elif method in ("llm_de", "llm_de_non_linear_differential"):
            return Mutation.llm_de_non_linear_differential(
                ind, p_ind, p_tri=p_tri, p_comp=p_comp,
                generation=generation, max_generations=max_generations,
            )
        elif method in ("reevo", "reevo_adaptive", "reevo_adaptive_neighborhood"):
            return Mutation.reevo_adaptive_neighborhood(
                ind, p_ind, p_tri=p_tri, p_comp=p_comp,
                generation=generation, max_generations=max_generations,
            )
        else:
            return Mutation.multigene(ind, p_ind)
