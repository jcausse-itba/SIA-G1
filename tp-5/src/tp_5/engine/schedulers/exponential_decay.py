from .base import Scheduler


class ExponentialDecay(Scheduler):
    """
    Exponential learning rate decay:
        lr_t = initial_lr * (decay_rate ** ((epoch - 1) / max(1, max_epochs - 1)))
    """

    def __init__(self, decay_rate: float = 0.05) -> None:
        self.decay_rate = float(decay_rate)

    def get_lr(self, initial_lr: float, epoch: int, max_epochs: int) -> float:
        progress = min(1.0, max(0.0, (epoch - 1) / max(1, max_epochs - 1)))
        return float(initial_lr * (self.decay_rate ** progress))
