import random
import numpy as np
from typing import Tuple
from tp_2.ga.individual import Individual

class Crossover:
    """Métodos de cruza para el motor genético (vectorizado vía NumPy)."""

    @staticmethod
    def one_point(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        n = p1.genome.shape[0]
        if n < 2:
            return Individual(p1.genome.copy()), Individual(p2.genome.copy())
        point = random.randint(1, n - 1)
        
        c1_genome = np.concatenate((p1.genome[:point], p2.genome[point:]), axis=0)
        c2_genome = np.concatenate((p2.genome[:point], p1.genome[point:]), axis=0)
        
        return Individual(c1_genome), Individual(c2_genome)

    @staticmethod
    def two_point(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        n = p1.genome.shape[0]
        if n < 3:
            return Crossover.one_point(p1, p2)
        pt1, pt2 = sorted(random.sample(range(1, n), 2))
        
        c1 = np.concatenate((p1.genome[:pt1], p2.genome[pt1:pt2], p1.genome[pt2:]), axis=0)
        c2 = np.concatenate((p2.genome[:pt1], p1.genome[pt1:pt2], p2.genome[pt2:]), axis=0)
        
        return Individual(c1), Individual(c2)

    @staticmethod
    def uniform(p1: Individual, p2: Individual, p: float = 0.5) -> Tuple[Individual, Individual]:
        n = p1.genome.shape[0]
        mask = (np.random.rand(n) < p).reshape((-1,) + (1,) * (p1.genome.ndim - 1))
        
        c1 = np.where(mask, p1.genome, p2.genome)
        c2 = np.where(mask, p2.genome, p1.genome)
        
        return Individual(c1), Individual(c2)

    @staticmethod
    def annular(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        n = p1.genome.shape[0]
        if n < 1:
            return Individual(p1.genome.copy()), Individual(p2.genome.copy())
        start = random.randint(0, n - 1)
        length = random.randint(1, max(1, n // 2))
        
        mask = np.zeros((n,) + (1,) * (p1.genome.ndim - 1), dtype=bool)
        for i in range(length):
            mask[(start + i) % n] = True
            
        c1 = np.where(mask, p2.genome, p1.genome)
        c2 = np.where(mask, p1.genome, p2.genome)
        
        return Individual(c1), Individual(c2)
    
    @staticmethod
    def adaptive_layer_spatial(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """
        SOTA Adaptive Layer-Preserving Crossover (ALPC).
        Preserves structural z-index depth ordering while dynamically recombining
        foreground details based on a non-linear layer-decay function.
        """
        n = p1.genome.shape[0]
        focal_point = random.random()
        bandwidth = 0.2 + 0.3 * random.random()

        layer_ratios = np.arange(n) / max(1, n - 1)
        dists = (layer_ratios - focal_point) / bandwidth
        swap_probs = np.exp(-0.5 * (dists ** 2)).reshape((-1,) + (1,) * (p1.genome.ndim - 1))
        
        mask = np.random.rand(*swap_probs.shape) < swap_probs
        c1 = np.where(mask, p2.genome, p1.genome)
        c2 = np.where(mask, p1.genome, p2.genome)
                
        return Individual(c1), Individual(c2)
