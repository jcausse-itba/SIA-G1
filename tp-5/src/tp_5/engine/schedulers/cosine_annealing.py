import numpy as np

from .base import Scheduler


class CosineAnnealing(Scheduler):
    """
    Cosine Annealing learning rate schedule:
        lr_t = initial_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(pi * epoch / max_epochs)))
    """

    def __init__(self, min_lr_ratio: float = 0.02) -> None:
        self.min_lr_ratio = float(min_lr_ratio)

    def get_lr(self, initial_lr: float, epoch: int, max_epochs: int) -> float:
        progress = min(1.0, max(0.0, (epoch - 1) / max(1, max_epochs - 1)))
        cosine_decay = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        return float(initial_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay))
