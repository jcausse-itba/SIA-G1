import ast
import contextlib
from functools import partial
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tp_3.config.loader import load_and_merge_config
from tp_3.config.parser import build_parser
from tp_3.config.validator import validate_config
import importlib
import inspect
import pkgutil


def resolve_class(package_path: str, name: str):
    """Dynamically locates and loads a class from a package by name/alias."""
    pkg = importlib.import_module(package_path)
    target = name.lower().replace("_", "")
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        mod = importlib.import_module(modname)
        for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls_name.lower() == target or (target == "linear" and cls_name == "Identity"):
                return cls
    raise ValueError(f'Could not resolve class "{name}" in "{package_path}"')


def instantiate_with_reflection(cls, config: dict):
    """Instantiates a class by dynamically matching constructor parameters to config keys."""
    sig = inspect.signature(cls.__init__)
    param_map = {"alpha": "momentum_beta"}
    kwargs = {p: config[param_map.get(p, p)] for p in sig.parameters if param_map.get(p, p) in config}
    return cls(**kwargs)
from tp_3.training_report import write_training_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "mlp_training_results.html"
FRAUD_TARGET = "big_model_fraud_probability"


def main() -> None:
    parser = build_parser()
    try:
        config = load_and_merge_config(parser)
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
        if "image" in features.columns:
            image_values = features["image"].map(
                lambda value: np.asarray(
                    value if isinstance(value, list) else ast.literal_eval(str(value)),
                    dtype=np.float64,
                )
            )
            inputs = np.stack(image_values.to_numpy())
        else:
            inputs = features.to_numpy(dtype=np.float64)

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

        all_metrics = []
        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            if validation_method == "explicit":
                train_inputs, train_outputs = inputs, outputs
                
                test_frame = pd.read_csv(
                    config["test_dataset_path"],
                    usecols=lambda column: column != "flagged_fraud",
                )
                test_targets = test_frame[target_column].to_numpy()
                test_features = test_frame.drop(columns=[target_column])
                if "image" in test_features.columns:
                    test_image_values = test_features["image"].map(
                        lambda value: np.asarray(
                            value if isinstance(value, list) else ast.literal_eval(str(value)),
                            dtype=np.float64,
                        )
                    )
                    test_inputs = np.stack(test_image_values.to_numpy())
                else:
                    test_inputs = test_features.to_numpy(dtype=np.float64)

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

            scaling_method = config.get("scaling", "standardization")
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

            is_simple = config["model_type"].startswith("simple") #TODO es redundante complejiza mucho logica, es mejor q si queres simple armes un mlp con arquitectura para q sea simple TODO: eliminar arg
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

            mlp_cls = resolve_class("tp_3.engine.models", "mlp") #TODO parametrizar
            loss_cls = resolve_class("tp_3.engine.loss_functions", "meansquarederror") #TODO parametrizar

            np.random.seed(42)
            model = mlp_cls(
                layer_sizes,
                hidden_activation,
                output_activation,
                loss_cls(),
                optimizer_factory,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                loss_history = model.fit(
                    train_inputs,
                    train_outputs,
                    epochs=config["max_epochs"],
                    lr=config["learning_rate"],
                    print_every=max(1, config["max_epochs"] // 10),
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
        print(f"Training complete: {config['model_type']} ({config['activation']})")
        print("Average CV metrics:" if validation_method == "k_fold" else "Held-out metrics:",
              ", ".join(f"{name}={value:.4f}" for name, value in avg_metrics.items())
        )
        print(f"Final training MSE: {loss_history[-1]:.5f}")
        print(f"Training graphs: {RESULTS_PATH}")
        if model_path:
            print(f"Model saved to: {model_path}")
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
