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
from tp_3.engine.activation_functions.identity import Identity
from tp_3.engine.activation_functions.relu import ReLU
from tp_3.engine.activation_functions.standard import Sigmoid, Step, Tanh
from tp_3.engine.loss_functions.mse import MeanSquaredError
from tp_3.engine.models.mlp import MLP
from tp_3.engine.optimizers.adam import Adam
from tp_3.engine.optimizers.momentum import Momentum
from tp_3.engine.optimizers.sgd import SGD
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
		split_index = int(len(order) * config["split_ratio"])
		train_indices, test_indices = order[:split_index], order[split_index:]
		train_inputs, test_inputs = inputs[train_indices], inputs[test_indices]
		train_outputs, test_outputs = outputs[train_indices], outputs[test_indices]

		feature_mean = train_inputs.mean(axis=0)
		feature_scale = train_inputs.std(axis=0)
		feature_scale[feature_scale == 0] = 1.0
		train_inputs = (train_inputs - feature_mean) / feature_scale
		test_inputs = (test_inputs - feature_mean) / feature_scale

		activation_types = {
			"step": Step,
			"linear": Identity,
			"sigmoid": partial(Sigmoid, beta=config["beta"]),
			"tanh": partial(Tanh, beta=config["beta"]),
			"relu": ReLU,
		}
		optimizer_types = {
			"sgd": SGD,
			"momentum": partial(Momentum, alpha=config["momentum_beta"]),
			"adam": Adam,
		}
		configured_activation = activation_types[config["activation"]]()
		if isinstance(configured_activation, Step):
			raise ValueError("Step activation has zero gradient and cannot be trained with MLP.fit().")

		if config["model_type"] == "simple_linear":
			hidden_activation = Identity()
			output_activation = Identity()
		elif config["model_type"] == "simple_non_linear":
			hidden_activation = Identity()
			output_activation = configured_activation
		else:
			hidden_activation = configured_activation
			output_activation = configured_activation

		np.random.seed(42)
		model = MLP(
			layer_sizes,
			hidden_activation,
			output_activation,
			MeanSquaredError(),
			optimizer_types[config["optimizer"]],
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
		print("Held-out metrics:", ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
		print(f"Final training MSE: {loss_history[-1]:.5f}")
		print(f"Training graphs: {RESULTS_PATH}")
		if model_path:
			print(f"Model saved to: {model_path}")
	except (FileNotFoundError, ValueError, KeyError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()



