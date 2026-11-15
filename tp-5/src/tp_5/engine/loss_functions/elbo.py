from numpy.typing import NDArray

from ..variational_layer import VariationalLayer
from .base import Loss
from .kl_divergence import KLDivergence


class ELBOLoss(Loss):
    """
    Evidence Lower Bound (ELBO) composite loss receiving its reconstruction loss,
    KL divergence loss, and variational latent layer via Dependency Injection.
    """

    def __init__(
        self,
        reconstruction_loss: Loss,
        kl_loss: KLDivergence,
        variational_layer: VariationalLayer,
        kl_weight: float = 0.001,
    ) -> None:
        self.reconstruction_loss = reconstruction_loss
        self.kl_loss = kl_loss
        self.variational_layer = variational_layer
        self.kl_weight = float(kl_weight)

    def compute(self, y_pred: NDArray, y_true: NDArray) -> float:
        recon = self.reconstruction_loss.compute(y_pred, y_true)
        kl = self.kl_loss.compute(self.variational_layer.mu, self.variational_layer.std)
        return float(recon + self.kl_weight * kl)

    def gradient(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        return self.reconstruction_loss.gradient(y_pred, y_true)
