import numpy as np
from tp_2.ga.individual import Individual
from tp_2.image.render import render_individual
from tp_2.ga.color import rgb_to_lab_vectorized

class FitnessEvaluator:
    def __init__(self, target_rgb_img: np.ndarray):
        self.height, self.width = target_rgb_img.shape[:2]
        # Si la imagen viene en RGBA (4 canales), recortamos a RGB (3 canales)
        rgb_only = target_rgb_img[:, :, :3]
        self.target_lab = rgb_to_lab_vectorized(rgb_only)

    def evaluate(self, individual: Individual) -> float:
        rendered_rgb = render_individual(individual, self.width, self.height)
        rendered_lab = rgb_to_lab_vectorized(rendered_rgb)
        
        mse = np.mean((self.target_lab - rendered_lab) ** 2)
        fitness = 10000.0 / (1.0 + mse)
        
        individual.fitness = fitness
        return fitness