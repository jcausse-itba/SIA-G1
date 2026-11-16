from abc import ABC, abstractmethod

class Scheduler(ABC):
    """Abstract Base Class for Learning Rate Schedulers."""
    @abstractmethod
    def get_lr(self, initial_lr: float, epoch: int, max_epochs: int) -> float:
        """Computes the learning rate for the given epoch (1-indexed)."""
        pass
