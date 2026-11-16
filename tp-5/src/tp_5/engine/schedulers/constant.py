from .base import Scheduler


class Constant(Scheduler):
    """Keeps the learning rate constant across all epochs."""

    def get_lr(self, initial_lr: float, epoch: int, max_epochs: int) -> float:
        return float(initial_lr)
