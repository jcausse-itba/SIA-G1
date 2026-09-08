from typing import List
import numpy as np
import cv2
from tp_2.ga.individual import Individual
from tp_2.image.render import render_individuals
from tp_2.ga.color import rgb_to_lab_vectorized


class FitnessEvaluator:
    def __init__(self, target_rgb_img: np.ndarray, eval_scale: float = 1.0):
        """
        eval_scale: Factor de reescalado para acelerar la evaluación.
        """
        self.eval_scale = eval_scale
        
        if eval_scale != 1.0:
            h, w = target_rgb_img.shape[:2]
            new_w = max(1, int(w * eval_scale))
            new_h = max(1, int(h * eval_scale))
            target_rgb_img = cv2.resize(target_rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.height, self.width = target_rgb_img.shape[:2]
        rgb_only = target_rgb_img[:, :, :3]

        self.target_lab = rgb_to_lab_vectorized(rgb_only)
    
    def evaluate(self, population: List[Individual]) -> List[float]:
        rendered_labs = render_individuals(population, self.width, self.height)
        fitnesses = []

        for individual, rendered_lab in zip(population, rendered_labs):
            diff = self.target_lab - rendered_lab
            delta_e = np.sqrt(np.einsum('...i,...i->...', diff, diff))
            
            # Normalización a rango [0.0, 1.0]
            fitness = max(0.0, 1.0 - (np.mean(delta_e) / 100.0))

            individual.fitness = fitness
            fitnesses.append(fitness)
            
        return fitnesses
