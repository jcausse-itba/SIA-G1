import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseModel(ABC):
    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray, **kwargs) -> list[float]:
        pass

    def evaluate(
        self,
        X: NDArray,
        y: NDArray,
        is_classification: bool,
        threshold: float = 0.5,
        target_column: str | None = None
    ) -> tuple[dict[str, float], NDArray, NDArray]:
        """Evaluates the model on the given dataset and returns metrics, actuals, and predictions."""
        predictions = self.forward(X)
        if not is_classification:
            actual = y.ravel()
            predicted = predictions.ravel()
            if target_column == "big_model_fraud_probability":
                predicted = np.clip(predicted, 0.0, 1.0)
            residuals = predicted - actual
            metrics = {
                "MAE": float(np.mean(np.abs(residuals))),
                "RMSE": float(np.sqrt(np.mean(residuals**2))),
            }
            total_variation = np.sum((actual - actual.mean()) ** 2)
            if total_variation > 0:
                metrics["R2"] = float(1 - np.sum(residuals**2) / total_variation)
            if np.unique(actual).size == 2:
                class_predictions = (predicted >= threshold).astype(int)
                metrics["Accuracy"] = float(np.mean(class_predictions == actual))
            return metrics, actual, predicted
        else:
            actual = np.argmax(y, axis=1)
            predicted = np.argmax(predictions, axis=1)
            metrics = {"Accuracy": float(np.mean(predicted == actual))}
            return metrics, actual, predicted

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Saves the model and optional metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"model": self}
        if metadata:
            data.update(metadata)
        with path.open("wb") as model_file:
            pickle.dump(data, model_file)
