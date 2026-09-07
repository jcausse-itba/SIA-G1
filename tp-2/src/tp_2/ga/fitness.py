import numpy as np
import cv2
from tp_2.ga.individual import Individual
from tp_2.image.render import render_individual
from tp_2.ga.color import rgb_to_lab_vectorized

class FitnessEvaluator:
    def __init__(self, target_rgb_img: np.ndarray, eval_scale: float = 0.5):
        """
        eval_scale: Factor de reescalado (0.0 < eval_scale <= 1.0).
        Reducir la escala acelera dramáticamente el cálculo de fitness.
        """
        self.eval_scale = eval_scale
        
        # Redimensionar la imagen target si eval_scale < 1.0
        if eval_scale != 1.0:
            h, w = target_rgb_img.shape[:2]
            new_w = max(1, int(w * eval_scale))
            new_h = max(1, int(h * eval_scale))
            target_rgb_img = cv2.resize(target_rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.height, self.width = target_rgb_img.shape[:2]
        rgb_only = target_rgb_img[:, :, :3]

        # Target precalculado en CIELAB a la resolución ajustada
        self.target_lab = rgb_to_lab_vectorized(rgb_only)
    
    def evaluate(self, individual: Individual) -> float:
        # Renderizado ultrarrápido con Numba + Blending RGB exacto
        rendered_lab = render_individual(individual, self.width, self.height)

        # Distancia Euclídea promedio por píxel en CIELAB
        delta_e = np.linalg.norm(self.target_lab - rendered_lab, axis=-1)
        mean_delta_e = np.mean(delta_e)
        fitness = 10000.0 / (1.0 + mean_delta_e)

        individual.fitness = fitness
        return fitness
