import ast
from functools import partial

import numpy as np
import pandas as pd

from tp_3.config.loader import load_and_merge_config
from tp_3.config.parser import build_parser
from tp_3.config.validator import validate_config


def main() -> None:
	parser = build_parser()
	try:
		config = load_and_merge_config(parser)
		validate_config(config)

		if config["model_type"] != "multilayer":
			raise ValueError(f"Model type '{config['model_type']}' is not implemented; use 'multilayer'.")

		from tp_3.engine.activation_functions.identity import Identity
		from tp_3.engine.activation_functions.relu import ReLU
		from tp_3.engine.activation_functions.standard import Sigmoid, Step, Tanh
		from tp_3.engine.loss_functions.mse import MeanSquaredError
		from tp_3.engine.models.mlp import MLP
		from tp_3.engine.optimizers.adam import Adam
		from tp_3.engine.optimizers.momentum import Momentum
		from tp_3.engine.optimizers.sgd import SGD

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
		frame = pd.read_csv(config["dataset_path"])
		target_column = "label" if "label" in frame.columns else frame.columns[-1]
		targets = frame[target_column].to_numpy()
		features = frame.drop(columns=[target_column])
		if "image" in features.columns:
			features = features["image"].map(
				lambda value: np.asarray(
					value if isinstance(value, list) else ast.literal_eval(str(value)),
					dtype=np.float64,
				)
			)
			inputs = np.stack(features.to_numpy())
		else:
			inputs = features.to_numpy(dtype=np.float64)

		output_size = config["architecture"][-1]
		if output_size == 1:
			outputs = targets.astype(np.float64).reshape(-1, 1)
		else:
			labels, class_ids = np.unique(targets, return_inverse=True)
			if len(labels) > output_size:
				raise ValueError("Architecture output size is smaller than the number of dataset classes.")
			outputs = np.eye(output_size, dtype=np.float64)[class_ids]

		if inputs.shape[1] != config["architecture"][0]:
			raise ValueError(
				f"Architecture expects {config['architecture'][0]} inputs, but the dataset has {inputs.shape[1]} features."
			)

		activation = activation_types[config["activation"]]()
		model = MLP(
			config["architecture"],
			activation,
			activation,
			MeanSquaredError(),
			optimizer_types[config["optimizer"]],
		)
		model.fit(
			inputs,
			outputs,
			epochs=config["max_epochs"],
			lr=config["learning_rate"],
		)
	except (FileNotFoundError, ValueError, KeyError) as error:
		parser.error(str(error))

if __name__ == "__main__":
	main()



