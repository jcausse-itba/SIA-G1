from .base import Scheduler


class LinearDecay(Scheduler):
    """
    Linear learning rate decay from initial_lr down to initial_lr * min_lr_ratio.
    """

    def __init__(self, min_lr_ratio: float = 0.02) -> None:
        self.min_lr_ratio = float(min_lr_ratio)

    def get_lr(self, initial_lr: float, epoch: int, max_epochs: int) -> float:
        progress = min(1.0, max(0.0, (epoch - 1) / max(1, max_epochs - 1)))
        return float(initial_lr * (1.0 - (1.0 - self.min_lr_ratio) * progress))
