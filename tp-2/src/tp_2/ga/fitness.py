import numpy as np
from tp_2.ga.individual import Individual
from tp_2.image.render import render_individual_lab_fast
from tp_2.ga.color import rgb_to_lab_vectorized

class FitnessEvaluator:
    def __init__(self, target_rgb_img: np.ndarray, eval_scale: float = 1.0):
        """
        eval_scale: Mantenido en 1.0 para mantener la resolución y máxima calidad.
        """
        self.height, self.width = target_rgb_img.shape[:2]
        rgb_only = target_rgb_img[:, :, :3]

        # Target precalculado en CIELAB
        self.target_lab = rgb_to_lab_vectorized(rgb_only)

    def evaluate(self, individual: Individual) -> float:
        # Renderizado ultrarrápido con Numba + Blending RGB exacto
        rendered_lab = render_individual_lab_fast(individual, self.width, self.height)

        # Cálculo de MSE perceptual en CIELAB
        mse = np.mean((self.target_lab - rendered_lab) ** 2)
        fitness = 10000.0 / (1.0 + mse)

        individual.fitness = fitness
        return fitness