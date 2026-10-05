import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

from tp_3.__main__ import PROJECT_ROOT, run_pipeline
from tp_3.config.loader import load_and_merge_config
from tp_3.config.parser import build_parser
from tp_3.metrics.ej2_metrics import write_ej2_plots


def get_ej2_experiments(base_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    experiments = {}

    for opt in ["sgd", "momentum", "adam"]:
        cfg = copy.deepcopy(base_config)
        cfg["optimizer"] = opt
        experiments[f"opt_{opt}"] = cfg

    for act in ["sigmoid", "relu", "tanh"]:
        cfg = copy.deepcopy(base_config)
        cfg["activation"] = act
        experiments[f"act_{act}"] = cfg

    for lr in [0.001, 0.01, 0.05, 0.1]:
        cfg = copy.deepcopy(base_config)
        cfg["learning_rate"] = lr
        experiments[f"lr_{lr}"] = cfg

    architectures = [
        [784, 32, 10],
        [784, 64, 32, 10],
        [784, 128, 64, 10],
    ]
    for arch in architectures:
        arch_str = "x".join(map(str, arch))
        cfg = copy.deepcopy(base_config)
        cfg["architecture"] = arch
        experiments[f"arch_{arch_str}"] = cfg

    cfg_split = copy.deepcopy(base_config)
    cfg_split["validation_method"] = "split"
    cfg_split["split_ratio"] = 0.8
    experiments["val_split"] = cfg_split

    cfg_kfold = copy.deepcopy(base_config)
    cfg_kfold["validation_method"] = "k_fold"
    cfg_kfold["k_folds"] = 5
    experiments["val_kfold"] = cfg_kfold

    if base_config.get("test_dataset_path"):
        cfg_explicit = copy.deepcopy(base_config)
        cfg_explicit["validation_method"] = "explicit"
        experiments["val_explicit"] = cfg_explicit

    return experiments


def _classify(name: str, cfg: Dict[str, Any]) -> tuple[str, Any]:
    """Mapea el nombre del experimento al esquema (experiment, x) que espera ej2_metrics."""
    if name.startswith("opt_"):
        return "optimizer", cfg["optimizer"]
    if name.startswith("act_"):
        return "activation", cfg["activation"]
    if name.startswith("lr_"):
        return "lr", cfg["learning_rate"]
    if name.startswith("arch_"):
        return "depth", len(cfg["architecture"]) - 2  # cantidad de capas ocultas
    if name.startswith("val_"):
        return "validation", cfg["validation_method"]
    return "other", name


def run_experiment(name: str, cfg: Dict[str, Any]) -> dict:
    """Corre un experimento y devuelve el registro etiquetado para ej2_metrics."""
    record = run_pipeline(cfg)
    experiment, x = _classify(name, cfg)
    record["label"] = name
    record["experiment"] = experiment
    record["x"] = x
    return record


def _metrics_dir(base_config: Dict[str, Any]) -> Path:
    metrics_dir = Path(base_config.get("metrics_dir") or "metrics/ej2")
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
        "-c", "--config", type=str, default="configs/ej2.toml",
        help="Path to the base configuration TOML file."
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

    base_parser = build_parser()
    base_config = load_and_merge_config(base_parser)

    experiments = get_ej2_experiments(base_config)

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

    runs: list[dict] = []
    for name, cfg in selected:
        print(f"\n" + "=" * 60)
        print(f" Running experiment: {name}")
        print("=" * 60)
        try:
            runs.append(run_experiment(name, cfg))
        except Exception as e:
            print(f"[!] Error running {name}: {e}")

    if not runs:
        print("No runs completed; nothing to report.")
        return

    meta = {
        "epochs": base_config["max_epochs"],
        "batch_size": base_config["batch_size"],
        "seeds": 1,
        "base_optimizer": base_config["optimizer"],
        "base_lr": base_config["learning_rate"],
        "width_strategy": "explicit",
        "labels": next((r["labels"] for r in runs if r.get("labels")), []),
    }
    results = {"runs": runs, "meta": meta}

    metrics_dir = _metrics_dir(base_config)
    (metrics_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults written to: {metrics_dir / 'results.json'}")

    if not args.no_plots:
        write_ej2_plots(metrics_dir, results)
        print(f"Plots written to: {metrics_dir}")


if __name__ == "__main__":
    main()
