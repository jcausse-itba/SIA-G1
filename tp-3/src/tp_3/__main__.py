import importlib
import inspect
import json
import pickle
import pkgutil
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tp_3.config.loader import load_and_merge_config
from tp_3.config.parser import build_parser
from tp_3.config.validator import validate_config
from tp_3.metrics.metrics import EpochMetricsTracker
from tp_3.metrics.report import write_fraud_metrics_report
from tp_3.training_report import write_training_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "mlp_training_results.html"
FRAUD_TARGET = "big_model_fraud_probability"


def resolve_class(package_path: str, name: str) -> Any:
    """Dynamically locates and loads a class from a package by name/alias."""
    pkg = importlib.import_module(package_path)
    target = name.lower().replace("_", "")
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        mod = importlib.import_module(modname)
        for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls_name.lower() == target or (target == "linear" and cls_name == "Identity"):
                return cls
    raise ValueError(f'Could not resolve class "{name}" in "{package_path}"')


def instantiate_with_reflection(cls: Any, config: dict) -> Any:
    """Instantiates a class by dynamically matching constructor parameters to config keys."""
    sig = inspect.signature(cls.__init__)
    param_map = {"alpha": "momentum_beta"}
    kwargs = {p: config[param_map.get(p, p)] for p in sig.parameters if param_map.get(p, p) in config}
    return cls(**kwargs)


def features_to_inputs(features: pd.DataFrame) -> np.ndarray:
    """Converts a features frame to a 2D float array (parsing the 'image' column if present)."""
    if "image" in features.columns:
        image_values = [
            v if isinstance(v, list) else json.loads(str(v))
            for v in features["image"]
        ]
        return np.array(image_values, dtype=np.float64)
    return features.to_numpy(dtype=np.float64)


class ClassificationHistoryTracker:
    """Registra loss y accuracy de train/validación por época para problemas de clasificación.

    Compatible con el hook `epoch_callback(epoch, model)` de MLP.fit().
    """

    def __init__(
        self,
        train_inputs: np.ndarray,
        train_outputs: np.ndarray,
        val_inputs: np.ndarray,
        val_outputs: np.ndarray,
        loss_fn: Any,
        total_epochs: int,
        n_points: int = 100,
    ) -> None:
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.val_inputs = val_inputs
        self.val_outputs = val_outputs
        self.loss_fn = loss_fn
        stride = max(1, total_epochs // max(1, n_points))
        self.epoch_set = set(range(stride, total_epochs + 1, stride)) | {1, total_epochs}
        self.history: list[dict] = []

    @staticmethod
    def _accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)))

    def __call__(self, epoch: int, model: Any) -> None:
        train_pred = model.forward(self.train_inputs)
        val_pred = model.forward(self.val_inputs)
        self.history.append(
            {
                "epoch": epoch,
                "train_loss": float(self.loss_fn.compute(train_pred, self.train_outputs)),
                "train_acc": self._accuracy(train_pred, self.train_outputs),
                "val_loss": float(self.loss_fn.compute(val_pred, self.val_outputs)),
                "val_acc": self._accuracy(val_pred, self.val_outputs),
            }
        )


def run_pipeline(config: dict) -> dict:
    """Ejecuta el pipeline de entrenamiento a partir de un diccionario de configuración.

    Devuelve un registro con los resultados de la corrida (historial por época,
    métricas finales, matriz de confusión, etc.) apto para serializar a JSON.
    """
    for key in ("dataset_path", "test_dataset_path", "save_model_path", "load_model_path"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            config[key] = str((PROJECT_ROOT / value).resolve())
    validate_config(config)

    frame = pd.read_csv(
        config["dataset_path"],
        usecols=lambda column: column != "flagged_fraud",
    )
    target_column = next(
        (column for column in (FRAUD_TARGET, "label") if column in frame.columns),
        frame.columns[-1],
    )
    targets = frame[target_column].to_numpy()
    features = frame.drop(columns=[target_column])
    inputs = features_to_inputs(features)

    if config["model_type"] == "multilayer":
        layer_sizes = config["architecture"]
        if inputs.shape[1] != layer_sizes[0]:
            raise ValueError(
                f"Architecture expects {layer_sizes[0]} inputs, but the dataset has {inputs.shape[1]} features."
            )
        output_size = layer_sizes[-1]
    else:
        layer_sizes = [inputs.shape[1], 1]
        output_size = 1

    if output_size == 1:
        outputs = targets.astype(np.float64).reshape(-1, 1)
        labels = None
    else:
        labels, class_ids = np.unique(targets, return_inverse=True)
        if len(labels) > output_size:
            raise ValueError("Architecture output size is smaller than the number of dataset classes.")
        outputs = np.eye(output_size, dtype=np.float64)[class_ids]

    if "timestamp" in frame.columns:
        order = np.argsort(frame["timestamp"].to_numpy(), kind="stable")
    else:
        order = np.random.default_rng(42).permutation(len(frame))

    validation_method = config.get("validation_method", "split")

    folds: list[tuple[Any, Any]]
    if validation_method == "split":
        split_index = int(len(order) * config["split_ratio"])
        folds = [(order[:split_index], order[split_index:])]
    elif validation_method == "k_fold":
        fold_chunks = np.array_split(order, config["k_folds"])
        folds = [
            (np.concatenate([fold_chunks[j] for j in range(config["k_folds"]) if j != i]), fold_chunks[i])
            for i in range(config["k_folds"])
        ]
    else:
        folds = [(None, None)]  # Placeholder for explicit loading

    scaling_method = config.get("scaling", "standardization")

    # Initialised up-front so they are always bound after the fold loop.
    model: Any = None
    feature_mean = np.zeros(inputs.shape[1], dtype=np.float64)
    feature_scale = np.ones(inputs.shape[1], dtype=np.float64)
    loss_history: list[float] = []
    actual = np.array([])
    predicted = np.array([])

    all_metrics = []
    tracker: EpochMetricsTracker | None = None
    history_tracker: ClassificationHistoryTracker | None = None
    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        if validation_method == "explicit":
            train_inputs, train_outputs = inputs, outputs

            test_frame = pd.read_csv(
                config["test_dataset_path"],
                usecols=lambda column: column != "flagged_fraud",
            )
            test_targets = test_frame[target_column].to_numpy()
            test_inputs = features_to_inputs(test_frame.drop(columns=[target_column]))

            if output_size == 1:
                test_outputs = test_targets.astype(np.float64).reshape(-1, 1)
            else:
                try:
                    test_class_ids = np.array([np.where(labels == val)[0][0] for val in test_targets])
                except IndexError:
                    raise ValueError("Test dataset contains classes not present in the training dataset.")
                test_outputs = np.eye(output_size, dtype=np.float64)[test_class_ids]
        else:
            train_inputs, test_inputs = inputs[train_indices], inputs[test_indices]
            train_outputs, test_outputs = outputs[train_indices], outputs[test_indices]

        if scaling_method == "minmax":
            feature_mean = train_inputs.min(axis=0)
            feature_scale = train_inputs.max(axis=0) - feature_mean
            feature_scale[feature_scale == 0] = 1.0
        elif scaling_method in ("standardization", "standard"):
            feature_mean = train_inputs.mean(axis=0)
            feature_scale = train_inputs.std(axis=0)
            feature_scale[feature_scale == 0] = 1.0
        else:
            feature_mean = np.zeros(train_inputs.shape[1], dtype=np.float64)
            feature_scale = np.ones(train_inputs.shape[1], dtype=np.float64)

        train_inputs = (train_inputs - feature_mean) / feature_scale
        test_inputs = (test_inputs - feature_mean) / feature_scale

        make_act = lambda name: instantiate_with_reflection(
            resolve_class("tp_3.engine.activation_functions", name), config
        )
        configured_activation = make_act(config["activation"])
        if type(configured_activation).__name__ == "Step":
            raise ValueError("Step activation has zero gradient and cannot be trained with MLP.fit().")

        is_simple = config["model_type"].startswith("simple")
        hidden_activation = make_act("linear" if is_simple else config["activation"])
        output_activation = make_act("linear" if config["model_type"] == "simple_linear" else config["activation"])

        opt_cls = resolve_class("tp_3.engine.optimizers", config["optimizer"])
        opt_sig = inspect.signature(opt_cls.__init__)
        opt_kwargs = {
            p: config[k]
            for p, k in [("alpha", "momentum_beta")]
            if p in opt_sig.parameters and k in config
        }
        optimizer_factory = partial(opt_cls, **opt_kwargs) if opt_kwargs else opt_cls

        mlp_cls = resolve_class("tp_3.engine.models", "mlp")
        loss_cls = resolve_class("tp_3.engine.loss_functions", config["loss"])
        loss_fn = instantiate_with_reflection(loss_cls, config)

        np.random.seed(42)
        if config.get("load_model_path"):
            with open(config["load_model_path"], "rb") as model_file:
                checkpoint = pickle.load(model_file)
            model = checkpoint.get("model", checkpoint)
        else:
            model = mlp_cls(
                layer_sizes,
                hidden_activation,
                output_activation,
                loss_fn,
                optimizer_factory,
            )

        tracker = None
        history_tracker = None
        if output_size == 1 and target_column == FRAUD_TARGET:
            tracker = EpochMetricsTracker(
                train_inputs, train_outputs, test_inputs, test_outputs,
                threshold=config["threshold"],
                fraud_cutoff=config.get("fraud_cutoff", 0.5),
                total_epochs=config["max_epochs"],
                n_points=config.get("metrics_points", 100),
            )
        elif output_size > 1:
            history_tracker = ClassificationHistoryTracker(
                train_inputs, train_outputs, test_inputs, test_outputs,
                loss_fn,
                total_epochs=config["max_epochs"],
                n_points=config.get("metrics_points", 100),
            )
        epoch_callback = tracker if tracker is not None else history_tracker

        loss_history = model.fit(
            train_inputs,
            train_outputs,
            epochs=config["max_epochs"],
            lr=config["learning_rate"],
            print_every=max(1, config["max_epochs"] // 10),
            batch_size=config["batch_size"],
            epoch_callback=epoch_callback,
            callback_epochs=epoch_callback.epoch_set if epoch_callback else None,
        )

        predictions = model.forward(test_inputs)
        if output_size == 1:
            actual = test_outputs.ravel()
            predicted = predictions.ravel()
            if target_column == FRAUD_TARGET:
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
                class_predictions = (predicted >= config["threshold"]).astype(int)
                metrics["Accuracy"] = float(np.mean(class_predictions == actual))

        else:
            actual = np.argmax(test_outputs, axis=1)
            predicted = np.argmax(predictions, axis=1)
            metrics = {"Accuracy": float(np.mean(predicted == actual))}

        all_metrics.append(metrics)
        if validation_method == "k_fold":
            print(f"Fold {fold_idx + 1} metrics: ", ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))

    # Average out the collected metrics
    avg_metrics = {}
    for k in all_metrics[0].keys():
        avg_metrics[k] = float(np.mean([m[k] for m in all_metrics if k in m]))

    model_path = Path(config["save_model_path"]) if config.get("save_model_path") else None
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as model_file:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": features.columns.tolist(),
                    "feature_mean": feature_mean,
                    "feature_scale": feature_scale,
                    "scaling_method": scaling_method,
                    "target_column": target_column,
                },
                model_file,
            )

    write_training_report(
        loss_history,
        actual,
        predicted,
        RESULTS_PATH,
        is_classification=output_size > 1,
    )
    if tracker is not None:
        metrics_dir = Path(config.get("metrics_dir") or "metrics")
        if not metrics_dir.is_absolute():
            metrics_dir = PROJECT_ROOT / metrics_dir
        summary = write_fraud_metrics_report(
            tracker, actual, predicted, metrics_dir, n_points=config.get("metrics_points", 100)
        )
        print(f"Metric charts written to: {metrics_dir}")
        print(f"Best-F1 threshold: {summary['best_threshold']:.2f} (F1={summary['best_f1']:.4f}), ROC AUC≈{summary['roc_auc']:.4f}")
    print(f"Training complete: {config['model_type']} ({config['activation']})")
    print("Average CV metrics:" if validation_method == "k_fold" else "Held-out metrics:",
          ", ".join(f"{name}={value:.4f}" for name, value in avg_metrics.items())
    )
    print(f"Final training MSE: {loss_history[-1]:.5f}")
    print(f"Training graphs: {RESULTS_PATH}")
    if model_path:
        print(f"Model saved to: {model_path}")

    # ---- Registro de resultados de la corrida (serializable a JSON) ----
    history = history_tracker.history if history_tracker else []
    last_epoch = history[-1] if history else {}

    confusion = None
    if output_size > 1 and actual.size:
        confusion = np.zeros((output_size, output_size), dtype=int)
        for a, p in zip(actual, predicted):
            confusion[int(a), int(p)] += 1
        confusion = confusion.tolist()

    n_params = (
        sum(int(layer.W.size + layer.b.size) for layer in model.layers)
        if model is not None and hasattr(model, "layers")
        else None
    )

    return {
        "config": {
            "model_type": config["model_type"],
            "optimizer": config["optimizer"],
            "activation": config["activation"],
            "learning_rate": config["learning_rate"],
            "architecture": list(layer_sizes),
            "loss": config["loss"],
            "batch_size": config["batch_size"],
            "max_epochs": config["max_epochs"],
            "validation_method": validation_method,
        },
        "optimizer": config["optimizer"],
        "lr": config["learning_rate"],
        "architecture": "x".join(map(str, layer_sizes)),
        "n_params": n_params,
        "history": history,
        "final_train_loss": last_epoch.get("train_loss", float(loss_history[-1])),
        "final_train_acc": last_epoch.get("train_acc"),
        "final_val_loss": last_epoch.get("val_loss"),
        "final_val_acc": last_epoch.get("val_acc", avg_metrics.get("Accuracy")),
        "confusion": confusion,
        "labels": [str(label) for label in labels] if labels is not None else None,
        "avg_metrics": avg_metrics,
    }


def main() -> None:
    parser = build_parser()
    try:
        config = load_and_merge_config(parser)
        run_pipeline(config)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
