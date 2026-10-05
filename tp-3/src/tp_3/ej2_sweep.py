import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from tp_3.metrics.ej2_metrics import write_ej2_plots
from tp_3.engine.models.mlp import MLP
from tp_3.engine.activation_functions.sigmoid import Sigmoid
from tp_3.engine.activation_functions.relu import ReLU
from tp_3.engine.activation_functions.tanh import Tanh
from tp_3.engine.activation_functions.identity import Identity
from tp_3.engine.loss_functions.mse import MeanSquaredError
from tp_3.engine.optimizers.sgd import SGD
from tp_3.engine.optimizers.momentum import Momentum
from tp_3.engine.optimizers.adam import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = str(PROJECT_ROOT / "../data and documentation/fraud_dataset.csv")
EPOCHS = 10
LR = 0.01
BATCH_SIZE = 32
METRICS_DIR = "metrics/ej2"


def get_ej2_experiments(input_dim: int) -> dict:
    """Returns a dictionary of experiments using explicit class references."""
    experiments = {}

    # Optimizers
    for opt in [SGD, Momentum, Adam]:
        experiments[f"opt_{opt.__name__.lower()}"] = {
            "experiment": "optimizer", "x": opt.__name__.lower(),
            "opt_cls": opt, "act_cls": Sigmoid, "lr": LR, "arch": [input_dim, 16, 1]
        }

    # Activations
    for act in [Sigmoid, ReLU, Tanh]:
        experiments[f"act_{act.__name__.lower()}"] = {
            "experiment": "activation", "x": act.__name__.lower(),
            "opt_cls": SGD, "act_cls": act, "lr": LR, "arch": [input_dim, 16, 1]
        }

    # Learning Rates
    for lr in [0.001, 0.01, 0.05, 0.1]:
        experiments[f"lr_{lr}"] = {
            "experiment": "lr", "x": lr,
            "opt_cls": SGD, "act_cls": Sigmoid, "lr": lr, "arch": [input_dim, 16, 1]
        }

    # Architectures
    architectures = [
        [input_dim, 32, 1],
        [input_dim, 64, 32, 1],
        [input_dim, 128, 64, 1],
    ]
    for arch in architectures:
        arch_str = "x".join(map(str, arch))
        experiments[f"arch_{arch_str}"] = {
            "experiment": "depth", "x": len(arch) - 2,
            "opt_cls": SGD, "act_cls": Sigmoid, "lr": LR, "arch": arch
        }

    return experiments


def run_experiment(name: str, cfg: dict, X: np.ndarray, y: np.ndarray, target_column: str) -> dict:
    model = MLP(
        layer_sizes=cfg["arch"],
        input_activation=cfg["act_cls"](),
        output_activation=Identity(),
        loss_function=MeanSquaredError(),
        optimizer=cfg["opt_cls"], 
    )

    history = model.fit(X, y, epochs=EPOCHS, lr=cfg["lr"], batch_size=BATCH_SIZE)
    
    # Delegated metric evaluation 
    metrics, actual, predicted = model.evaluate(X, y, is_classification=False, target_column=target_column)
    
    return {
        "label": name,
        "experiment": cfg["experiment"],
        "x": cfg["x"],
        "history": history,
        "metrics": metrics,
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
    }


def _metrics_dir() -> Path:
    metrics_dir = Path(METRICS_DIR)
    if not metrics_dir.is_absolute():
        metrics_dir = PROJECT_ROOT / metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def main() -> None:
    exp_parser = argparse.ArgumentParser(
        description="Experiment sweep runner for Exercise 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_parser.add_argument(
        "--exp", type=str, default="all",
        help="Name of the experiment to run or 'all' to run the full suite."
    )
    exp_parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (only write results.json)."
    )
    args, _ = exp_parser.parse_known_args()

    # --- 1. Dataset is read and pre-processed once at the start ---
    print(f"Loading dataset from: {DATASET_PATH}")
    frame = pd.read_csv(DATASET_PATH)
    if "flagged_fraud" in frame.columns:
        frame = frame.drop(columns=["flagged_fraud"])
        
    target_column = "big_model_fraud_probability" if "big_model_fraud_probability" in frame.columns else frame.columns[-1]
    y = frame[target_column].to_numpy().reshape(-1, 1)
    X = frame.drop(columns=[target_column]).to_numpy(dtype=np.float64)
    
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    X = (X - mean) / scale

    # --- 2. Resolve parameters & experiments ---
    experiments = get_ej2_experiments(X.shape[1])

    if args.exp != "all" and args.exp not in experiments:
        print(f"The experiment '{args.exp}' does not exist. Available experiments:")
        for name in experiments.keys():
            print(f" - {name}")
        return

    selected = experiments.items() if args.exp == "all" else [(args.exp, experiments[args.exp])]
    if args.exp == "all":
        print(f"=== Starting Exercise 2 Sweep ({len(experiments)} experiments) ===")
    else:
        print(f"=== Running experiment: {args.exp} ===")

    # --- 3. Run selected experiments ---
    runs: list[dict] = []
    for name, cfg in selected:
        print(f"\n" + "=" * 60)
        print(f" Running experiment: {name}")
        print("=" * 60)
        try:
            runs.append(run_experiment(name, cfg, X, y, target_column))
        except Exception as e:
            print(f"[!] Error running {name}: {e}")

    if not runs:
        print("No runs completed; nothing to report.")
        return

    meta = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seeds": 1,
        "base_optimizer": SGD.__name__,
        "base_lr": LR,
        "width_strategy": "explicit",
        "labels": next((r["labels"] for r in runs if r.get("labels")), []),
    }
    results = {"runs": runs, "meta": meta}

    metrics_dir = _metrics_dir()
    (metrics_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults written to: {metrics_dir / 'results.json'}")

    if not args.no_plots:
        write_ej2_plots(metrics_dir, results)
        print(f"Plots written to: {metrics_dir}")


if __name__ == "__main__":
    main()
