import copy
import math
import random
import numpy as np
from typing import List, Tuple

from tp_2.ga.individual import Individual

# Diferencias máximas de rango por cada componente (x1..y3, H, C, L, A)
_RANGES_DIFF = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 360.0, 130.0, 100.0, 0.95], dtype=np.float32)

def _clip_genome(genome: np.ndarray):
    """Restringe los valores in-place de la matriz NumPy completa según los límites de cada componente."""
    genome[:, 0:6] = np.clip(genome[:, 0:6], 0.0, 1.0)
    genome[:, 6] = genome[:, 6] % 360.0
    genome[:, 7] = np.clip(genome[:, 7], 0.0, 130.0)
    genome[:, 8] = np.clip(genome[:, 8], 0.0, 100.0)
    genome[:, 9] = np.clip(genome[:, 9], 0.05, 1.0)

class Mutation:

    @staticmethod
    def single_gene(
        ind: Individual, p_ind: float, scale: float = 0.1,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        new_ind = copy.deepcopy(ind)
        n = new_ind.genome.shape[0]
        idx = random.randrange(n)
        c = random.randrange(10)
        
        new_ind.genome[idx, c] += random.gauss(0, _RANGES_DIFF[c] * scale)
        _clip_genome(new_ind.genome)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def multigene(
        ind: Individual, p_ind: float, k: int = 0, p_comp: float = 0.2, scale: float = 0.1,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        new_ind = copy.deepcopy(ind)
        n = new_ind.genome.shape[0]
        k_actual = min(k if k > 0 else max(1, -(-n // 4)), n)

        indices = random.sample(range(n), k_actual)
        tri_mask = np.zeros(n, dtype=bool)
        tri_mask[indices] = True
        
        comp_mask = np.random.rand(n, 10) < p_comp
        noise = np.random.normal(0, scale * _RANGES_DIFF, size=(n, 10))
        
        final_mask = tri_mask[:, None] & comp_mask
        new_ind.genome = np.where(final_mask, new_ind.genome + noise, new_ind.genome)
        
        _clip_genome(new_ind.genome)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def uniform(
        ind: Individual, p_ind: float, p_tri: float = 0.2, p_comp: float = 0.15,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        new_ind = copy.deepcopy(ind)
        n = new_ind.genome.shape[0]
        
        tri_mask = np.random.rand(n) < p_tri
        comp_mask = np.random.rand(n, 10) < p_comp
        final_mask = tri_mask[:, None] & comp_mask
        
        random_genome = np.empty((n, 10), dtype=np.float32)
        random_genome[:, 0:6] = np.random.rand(n, 6)
        random_genome[:, 6] = np.random.uniform(0.0, 360.0, n)
        random_genome[:, 7] = np.random.uniform(0.0, 130.0, n)
        random_genome[:, 8] = np.random.uniform(0.0, 100.0, n)
        random_genome[:, 9] = np.random.uniform(0.05, 1.0, n)
        
        new_ind.genome = np.where(final_mask, random_genome, new_ind.genome)
        _clip_genome(new_ind.genome)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def non_uniform(
        ind: Individual, p_ind: float, p_tri: float = 0.3, p_comp: float = 0.2,
        generation: int = 1, max_generations: int = 1000,
        min_scale: float = 0.005, max_scale: float = 0.15,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        t = min(generation / max(max_generations, 1), 1.0)
        scale = max_scale * (1.0 - t) + min_scale * t

        new_ind = copy.deepcopy(ind)
        n = new_ind.genome.shape[0]
        
        tri_mask = np.random.rand(n) < p_tri
        comp_mask = np.random.rand(n, 10) < p_comp
        noise = np.random.normal(0, scale * _RANGES_DIFF, size=(n, 10))
        
        final_mask = tri_mask[:, None] & comp_mask
        new_ind.genome = np.where(final_mask, new_ind.genome + noise, new_ind.genome)
        
        _clip_genome(new_ind.genome)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def llm_de_non_linear_differential(
        ind: Individual, p_ind: float, p_tri: float = 0.3, p_comp: float = 0.2,
        generation: int = 1, max_generations: int = 1000, f_base: float = 0.5,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        t = min(generation / max(max_generations, 1), 1.0)
        # Factor de escala no lineal F con modulación sinusoidal y decaimiento cuadrático
        f_scale = f_base * (1.0 - t**2) * (0.8 + 0.2 * math.cos(math.pi * t))

        new_ind = copy.deepcopy(ind)
        n = new_ind.genome.shape[0]
        
        tri_mask = np.random.rand(n) < p_tri
        comp_mask = np.random.rand(n, 10) < p_comp

        r1_idx = np.random.randint(0, n, size=n)
        r2_idx = np.random.randint(0, n, size=n)
        diff = ind.genome[r1_idx] - ind.genome[r2_idx]
        
        noise = np.random.normal(0, 0.02 * (1.0 - t) * _RANGES_DIFF, size=(n, 10))
        delta = f_scale * diff + noise
        
        final_mask = tri_mask[:, None] & comp_mask
        new_ind.genome = np.where(final_mask, new_ind.genome + delta, new_ind.genome)

        _clip_genome(new_ind.genome)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def reevo_adaptive_neighborhood(
        ind: Individual, p_ind: float, p_tri: float = 0.3, p_comp: float = 0.2,
        generation: int = 1, max_generations: int = 1000,
        initial_radius: float = 0.25, alpha: float = 2.0,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        t = min(generation / max(max_generations, 1), 1.0)
        # Adaptación no lineal del radio del vecindario
        radius = initial_radius * ((1.0 - t) ** alpha) + 0.005

        new_ind = copy.deepcopy(ind)
        n = new_ind.genome.shape[0]
        
        tri_mask = np.random.rand(n) < p_tri
        comp_mask = np.random.rand(n, 10) < p_comp
        
        scale = radius * _RANGES_DIFF
        u = np.random.rand(n, 10) - 0.5
        step = np.tan(np.pi * u * 0.45) * scale
        
        final_mask = tri_mask[:, None] & comp_mask
        new_ind.genome = np.where(final_mask, new_ind.genome + step, new_ind.genome)

        _clip_genome(new_ind.genome)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def scale_adaptive_gaussian(
        ind: Individual, p_ind: float, p_tri: float = 0.3, p_comp: float = 0.2,
        generation: int = 1, max_generations: int = 1000,
        min_scale: float = 0.001, max_scale: float = 0.2,
    ) -> Individual:
        if random.random() > p_ind:
            return ind
        t = min(generation / max(max_generations, 1), 1.0)
        scale = max_scale * (1.0 - t) + min_scale * t

        new_ind = copy.deepcopy(ind)
        n = new_ind.genome.shape[0]

        tri_mask = np.random.rand(n) < p_tri
        comp_mask = np.random.rand(n, 10) < p_comp
        noise = np.random.normal(0, scale * _RANGES_DIFF, size=(n, 10))

        final_mask = tri_mask[:, None] & comp_mask
        new_ind.genome = np.where(final_mask, new_ind.genome + noise, new_ind.genome)

        _clip_genome(new_ind.genome)
        new_ind.fitness = None
        return new_ind

    @staticmethod
    def apply(
        ind: Individual, p_ind: float, method: str,
        generation: int = 1, max_generations: int = 1000,
        p_tri: float = 0.3, p_comp: float = 0.2,
    ) -> Individual:
        if method == "single_gene":
            return Mutation.single_gene(ind, p_ind)
        elif method == "multigene":
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
        elif method in ("scale_adaptive_gaussian", "gaussian_adaptive", "gaussian"):
            return Mutation.scale_adaptive_gaussian(
                ind, p_ind, p_tri=p_tri, p_comp=p_comp,
                generation=generation, max_generations=max_generations,
            )
        else:
            return Mutation.multigene(ind, p_ind)
