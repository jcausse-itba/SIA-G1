"""Exercise 1 study (fraud / knowledge distillation).

Stages (each one stores its raw results in --out-dir; charts are built by tp_3/metrics/ej1_metrics.py):
  design   dataset / split description         -> design.json
  compare  linear vs non-linear, ALL samples   -> compare.json     (needs --compare-config)
  study    k-fold study of the chosen model    -> study.json + study.npz
  all      the three above, then the charts

Examples (project root):
  uv run python -m tp_3.ej1_study -c configs/ej1-non_linear.toml --compare-config configs/ej1.toml
  uv run python -m tp_3.ej1_study -c configs/ej1-non_linear.toml --stage study --cv forward --drop-features timestamp
  uv run python -m tp_3.ej1_study -c configs/ej1-non_linear.toml --set max_epochs=300 --set k_folds=5
"""
import argparse
import inspect
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tp_3.__main__ import FRAUD_TARGET, PROJECT_ROOT, features_to_inputs, instantiate_with_reflection, resolve_class
from tp_3.config.loader import _read_config_file
from tp_3.config.parser import build_parser
from tp_3.metrics.metrics import EpochMetricsTracker


# ------------------------------------------------------------------------------ config
def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else PROJECT_ROOT / p


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(path: str, overrides: list[str]) -> dict[str, Any]:
    """Parser defaults < config file < --set KEY=VALUE (same precedence idea as the main CLI)."""
    config = vars(build_parser().parse_args([]))
    config.update(_read_config_file(str(_resolve(path))))
    for item in overrides:
        key, _, value = item.partition("=")
        config[key.strip()] = _parse_value(value)
    if config.get("dataset_path") and not Path(config["dataset_path"]).is_absolute():
        config["dataset_path"] = str((PROJECT_ROOT / config["dataset_path"]).resolve())
    return config


# -------------------------------------------------------------------------------- data
@dataclass
class FraudData:
    inputs: np.ndarray
    targets: np.ndarray            # BigModel probability, shape (N,)
    order: np.ndarray              # row indices in chronological order
    feature_names: list[str]
    has_timestamp: bool


def load_fraud_data(config: dict[str, Any], drop_features: list[str]) -> FraudData:
    frame = pd.read_csv(config["dataset_path"], usecols=lambda c: c != "flagged_fraud")
    if FRAUD_TARGET not in frame.columns:
        raise ValueError(f"Column '{FRAUD_TARGET}' not found in {config['dataset_path']}.")
    missing = [c for c in drop_features if c not in frame.columns]
    if missing:
        raise ValueError(f"--drop-features: columns not found: {missing}")
    features = frame.drop(columns=[FRAUD_TARGET, *drop_features])
    has_ts = "timestamp" in frame.columns
    order = (np.argsort(frame["timestamp"].to_numpy(), kind="stable") if has_ts
             else np.random.default_rng(42).permutation(len(frame)))
    return FraudData(features_to_inputs(features), frame[FRAUD_TARGET].to_numpy(dtype=np.float64),
                     order, features.columns.tolist(), has_ts)


def make_folds(order: np.ndarray, k: int, cv: str):
    """kfold: like the main script (train on every other chunk, past and future).
    forward: expanding window, train only on chunks before the validation chunk."""
    if cv == "kfold":
        chunks = np.array_split(order, k)
        folds = [(np.concatenate([chunks[j] for j in range(k) if j != i]), chunks[i]) for i in range(k)]
    elif cv == "forward":
        chunks = np.array_split(order, k + 1)
        folds = [(np.concatenate(chunks[:i]), chunks[i]) for i in range(1, k + 1)]
    else:
        raise ValueError(f"Unknown --cv '{cv}'")
    return chunks, folds


def standardize(train: np.ndarray, others: list[np.ndarray], method: str):
    if method == "minmax":
        shift = train.min(axis=0)
        scale = train.max(axis=0) - shift
    elif method in ("standardization", "standard"):
        shift, scale = train.mean(axis=0), train.std(axis=0)
    else:
        shift, scale = np.zeros(train.shape[1]), np.ones(train.shape[1])
    scale = np.where(scale == 0, 1.0, scale)
    return [(a - shift) / scale for a in (train, *others)]


# ------------------------------------------------------------------------------- model
def build_model(config: dict[str, Any], n_features: int) -> Any:
    """Same construction rules as tp_3/__main__.py for simple_linear / simple_non_linear."""
    make_act = lambda name: instantiate_with_reflection(
        resolve_class("tp_3.engine.activation_functions", name), config)
    simple = config["model_type"].startswith("simple")
    hidden = make_act("linear" if simple else config["activation"])
    output = make_act("linear" if config["model_type"] == "simple_linear" else config["activation"])
    opt_cls = resolve_class("tp_3.engine.optimizers", config["optimizer"])
    opt_kwargs = {p: config[k] for p, k in [("alpha", "momentum_beta")]
                  if p in inspect.signature(opt_cls.__init__).parameters and k in config}
    factory = partial(opt_cls, **opt_kwargs) if opt_kwargs else opt_cls
    loss_fn = instantiate_with_reflection(resolve_class("tp_3.engine.loss_functions", config["loss"]), config)
    sizes = [n_features, 1] if simple else config["architecture"]
    np.random.seed(42)
    return resolve_class("tp_3.engine.models", "mlp")(sizes, hidden, output, loss_fn, factory)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    res = predicted - actual
    var = float(np.sum((actual - actual.mean()) ** 2))
    return {"MAE": float(np.mean(np.abs(res))), "RMSE": float(np.sqrt(np.mean(res ** 2))),
            "R2": float(1 - np.sum(res ** 2) / var) if var > 0 else float("nan")}


# ------------------------------------------------------------------------------ stages
def stage_design(config, data: FraudData, args, out: Path) -> None:
    cutoff = config.get("fraud_cutoff", 0.5)
    k = config["k_folds"]
    chunks, _ = make_folds(data.order, k, args.cv)
    n = len(data.order)
    sorted_targets = data.targets[data.order]
    counts, edges = np.histogram(data.targets, bins=50, range=(0.0, 1.0))
    rows, start = [], 0
    for j, chunk in enumerate(chunks):
        t = data.targets[chunk]
        rows.append({"chunk": j, "start_pct": 100 * start / n, "width_pct": 100 * len(chunk) / n, "n": len(chunk),
                     "positives": int(np.sum(t >= cutoff)), "rate": float(np.mean(t >= cutoff))})
        start += len(chunk)
    split = int(n * config["split_ratio"])
    roles = {}
    for i in range(len(chunks) if args.cv == "kfold" else k):
        vi = i if args.cv == "kfold" else i + 1
        roles[f"Fold {i + 1}"] = ["val" if j == vi else ("train" if (args.cv == "kfold" or j < vi) else "unused")
                                  for j in range(len(chunks))]
    out.mkdir(parents=True, exist_ok=True)
    (out / "design.json").write_text(json.dumps({
        "cutoff": cutoff, "cv": args.cv, "k": k, "n_rows": n, "n_features": len(data.feature_names),
        "feature_names": data.feature_names, "has_timestamp": data.has_timestamp,
        "positive_rate": float(np.mean(data.targets >= cutoff)), "n_outside_unit": int(np.sum((data.targets < 0) | (data.targets > 1))),
        "hist": {"counts": counts.tolist(), "edges": edges.tolist()},
        "chunks": rows, "roles": roles,
        "holdout": {"split_ratio": config["split_ratio"], "n_train": split, "n_val": n - split,
                    "rate_train": float(np.mean(sorted_targets[:split] >= cutoff)),
                    "rate_val": float(np.mean(sorted_targets[split:] >= cutoff))},
    }))
    print(f"design -> {out / 'design.json'}")


def stage_compare(config, other_config, data: FraudData, drop, out: Path) -> None:
    """Both perceptrons trained AND evaluated on every sample (no split), as the exercise asks."""
    sample = np.sort(np.random.default_rng(0).choice(len(data.targets), min(3000, len(data.targets)), replace=False))
    models = []
    for role, cfg in (("chosen", config), ("other", other_config)):
        x, = standardize(data.inputs, [], cfg.get("scaling", "standardization"))
        y = data.targets.reshape(-1, 1)
        model = build_model(cfg, x.shape[1])
        print(f"compare: training {cfg['model_type']} ({cfg['activation']}) on all {len(x)} samples")
        history = model.fit(x, y, epochs=cfg["max_epochs"], lr=cfg["learning_rate"],
                            print_every=max(1, cfg["max_epochs"] // 10), batch_size=cfg["batch_size"])
        pred = model.forward(x).ravel()
        models.append({
            "role": role, "model_type": cfg["model_type"], "activation": cfg["activation"],
            "learning_rate": cfg["learning_rate"], "optimizer": cfg["optimizer"], "epochs": cfg["max_epochs"],
            "loss_history": [float(v) for v in history],
            "metrics": regression_metrics(data.targets, pred),
            "metrics_clipped": regression_metrics(data.targets, np.clip(pred, 0, 1)),
            "share_outside_unit": float(np.mean((pred < 0) | (pred > 1))),
            "sample_actual": data.targets[sample].tolist(), "sample_pred": pred[sample].tolist(),
        })
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps({"models": models, "n_samples": len(data.targets)}))
    print(f"compare -> {out / 'compare.json'}")


def stage_study(config, data: FraudData, args, out: Path) -> None:
    k, cutoff = config["k_folds"], config.get("fraud_cutoff", 0.5)
    chunks, folds = make_folds(data.order, k, args.cv)
    arrays: dict[str, np.ndarray] = {}
    fold_meta = []
    for i, (tr, va) in enumerate(folds):
        train_x, val_x = standardize(data.inputs[tr], [data.inputs[va]], config.get("scaling", "standardization"))
        train_y, val_y = data.targets[tr].reshape(-1, 1), data.targets[va].reshape(-1, 1)
        tracker = EpochMetricsTracker(train_x, train_y, val_x, val_y, threshold=config["threshold"],
                                      fraud_cutoff=cutoff, total_epochs=config["max_epochs"],
                                      n_points=config.get("metrics_points", 100))
        model = build_model(config, train_x.shape[1])
        print(f"study: fold {i + 1}/{len(folds)} (train={len(tr)}, val={len(va)})")
        history = model.fit(train_x, train_y, epochs=config["max_epochs"], lr=config["learning_rate"],
                            print_every=max(1, config["max_epochs"] // 5), batch_size=config["batch_size"],
                            epoch_callback=tracker, callback_epochs=tracker.epoch_set)
        arrays[f"scores_{i}"] = model.forward(val_x).ravel()
        arrays[f"targets_{i}"] = data.targets[va]
        arrays[f"rows_{i}"] = va
        fold_meta.append({"fold": i, "n_train": len(tr), "n_val": len(va), "records": tracker.records,
                          "loss_history": [float(v) for v in history]})
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "study.npz", **arrays)
    (out / "study.json").write_text(json.dumps({
        "model_type": config["model_type"], "activation": config["activation"], "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"], "batch_size": config["batch_size"], "epochs": config["max_epochs"],
        "threshold": config["threshold"], "fraud_cutoff": cutoff, "cv": args.cv, "k": k,
        "metrics_points": config.get("metrics_points", 100), "features": data.feature_names, "folds": fold_meta}))
    print(f"study -> {out / 'study.json'}, study.npz")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TP3 ej1: fraud study for the chosen perceptron")
    p.add_argument("-c", "--config", required=True, help="Config of the CHOSEN model (e.g. configs/ej1-non_linear.toml)")
    p.add_argument("--compare-config", default=None, help="Config of the other perceptron (linear vs non-linear stage)")
    p.add_argument("--stage", choices=["design", "compare", "study", "all"], default="all")
    p.add_argument("--out-dir", default="metrics/ej1")
    p.add_argument("--cv", choices=["kfold", "forward"], default="kfold",
                   help="kfold: as main (trains also on future chunks). forward: expanding window (past only)")
    p.add_argument("--drop-features", nargs="*", default=[], help="Columns not used as inputs (e.g. timestamp)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override config values")
    p.add_argument("--recommend", default="f1", help="Threshold rule: 'f1', 'cost:<FN/FP ratio>' or a number")
    p.add_argument("--cost-ratios", type=float, nargs="+", default=[1, 5, 10, 50])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_cli().parse_args()
    config = load_config(args.config, args.set)
    out = Path(args.out_dir)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    data = load_fraud_data(config, args.drop_features)
    print(f"{len(data.targets)} rows, {len(data.feature_names)} features: {data.feature_names}")

    if args.stage in ("design", "all"):
        stage_design(config, data, args, out)
    if args.stage in ("compare", "all"):
        if args.compare_config:
            stage_compare(config, load_config(args.compare_config, args.set), data, args.drop_features, out)
        elif args.stage == "compare":
            raise SystemExit("--compare-config is required for the compare stage")
        else:
            print("compare: skipped (no --compare-config)")
    if args.stage in ("study", "all"):
        stage_study(config, data, args, out)
    if not args.no_plots:
        from tp_3.metrics.ej1_metrics import write_ej1_plots
        write_ej1_plots(out, recommend=args.recommend, cost_ratios=args.cost_ratios, bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()