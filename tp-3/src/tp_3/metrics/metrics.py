"""Binary-classification metrics for the fraud exercise (per-epoch and per-threshold)."""
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_POINTS = 100
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "tpr", "fpr")


def select_epochs(total_epochs: int, n_points: int = DEFAULT_POINTS) -> NDArray[np.int64]:
    return np.arange(1, min(total_epochs, n_points) + 1)


def binarize(targets: NDArray, cutoff: float) -> NDArray[np.bool_]:
    """Turns the BigModel probability into a hard fraud label (>= cutoff is fraud)."""
    return np.asarray(targets).ravel() >= cutoff


def binary_metrics(actual: NDArray[np.bool_], scores: NDArray, threshold: float) -> dict[str, float]:
    """accuracy, precision, recall, f1, tpr, fpr at one threshold (0 when a ratio is undefined)."""
    pred = np.asarray(scores).ravel() >= threshold
    tp = float(np.sum(pred & actual))
    fp = float(np.sum(pred & ~actual))
    fn = float(np.sum(~pred & actual))
    tn = float(np.sum(~pred & ~actual))
    div = lambda a, b: a / b if b > 0 else 0.0
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "tpr": recall,
        "fpr": div(fp, fp + tn),
    }


def threshold_sweep(
    actual: NDArray[np.bool_], scores: NDArray, n_points: int = DEFAULT_POINTS
) -> dict[str, NDArray]:
    """Metrics for `n_points` thresholds evenly spaced in [0, 1]."""
    if n_points < 1:
        raise ValueError(f"metrics_points must be at least 1, got {n_points}.")
    thresholds = np.linspace(0.0, 1.0, n_points)
    rows = [binary_metrics(actual, scores, t) for t in thresholds]
    out: dict[str, NDArray] = {"threshold": thresholds}
    for name in METRIC_NAMES:
        out[name] = np.array([r[name] for r in rows])
    return out


def area_under(x: NDArray, y: NDArray) -> float:
    """Trapezoidal area, sorting by x first."""
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


class EpochMetricsTracker:
    """Callable for MLP.fit(epoch_callback=...): records loss + metrics on train and validation."""

    def __init__(
        self,
        train_inputs: NDArray,
        train_outputs: NDArray,
        val_inputs: NDArray,
        val_outputs: NDArray,
        threshold: float,
        fraud_cutoff: float,
        total_epochs: int,
        n_points: int = DEFAULT_POINTS,
    ) -> None:
        self.data = {
            "train": (train_inputs, train_outputs),
            "val": (val_inputs, val_outputs),
        }
        self.threshold = threshold
        self.fraud_cutoff = fraud_cutoff
        self.epochs = select_epochs(total_epochs, n_points)
        self.epoch_set = {int(e) for e in self.epochs}
        self.records: dict[str, Any] = {"epoch": []}
        for split in self.data:
            self.records[f"{split}_loss"] = []
            for name in METRIC_NAMES:
                self.records[f"{split}_{name}"] = []

    def __call__(self, epoch: int, model: Any) -> None:
        self.records["epoch"].append(epoch)
        for split, (x, y) in self.data.items():
            raw = model.forward(x)
            self.records[f"{split}_loss"].append(float(model.loss_fn.compute(raw, y)))
            scores = np.clip(raw.ravel(), 0.0, 1.0)
            metrics = binary_metrics(binarize(y, self.fraud_cutoff), scores, self.threshold)
            for name in METRIC_NAMES:
                self.records[f"{split}_{name}"].append(metrics[name])