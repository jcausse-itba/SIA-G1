from typing import List
import numpy as np
import cv2
from tp_2.ga.individual import Individual
from tp_2.image.render import render_individuals
from tp_2.ga.color import rgb_to_lab_vectorized

try:
    from tp_2._tp_2_rust import evaluate_fitness_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class FitnessEvaluator:
    def __init__(self, target_rgb_img: np.ndarray, eval_scale: float = 0.5):
        """
        eval_scale: Factor de reescalado (0.0 < eval_scale <= 1.0).
        Reducir la escala acelera dramáticamente el cálculo de fitness.
        """
        self.eval_scale = eval_scale
        
        if eval_scale != 1.0:
            h, w = target_rgb_img.shape[:2]
            new_w = max(1, int(w * eval_scale))
            new_h = max(1, int(h * eval_scale))
            target_rgb_img = cv2.resize(target_rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.height, self.width = target_rgb_img.shape[:2]
        rgb_only = target_rgb_img[:, :, :3]

        self.target_lab = np.ascontiguousarray(rgb_to_lab_vectorized(rgb_only), dtype=np.float32)
    
    def evaluate(self, population: List[Individual]) -> List[float]:
        if RUST_AVAILABLE:
            genomes = np.ascontiguousarray([ind.genome for ind in population], dtype=np.float32)
            fitnesses = evaluate_fitness_rust(genomes, self.target_lab, self.width, self.height)
            for ind, fit in zip(population, fitnesses):
                ind.fitness = fit
            return fitnesses
        return []
