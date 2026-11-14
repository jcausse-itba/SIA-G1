"""Exercise 2 (digits) sweep: learning rate, optimizer, activation, depth and learning rate x optimizer.

* digits.csv is split into train / validation (hyper-parameter selection); digits_test.csv is only used, once,
  to evaluate the BEST model (best final validation accuracy over every run), like "production".
* Stores results.json (format read by tp_3/metrics/ej2_metrics.py), protocol.json, best_model.npz/.json and
  best.json (read by tp_3/metrics/ej2_extra_metrics.py) in --out-dir.

  uv run python -m tp_3.study.ej2_sweep --exp lr --set max_epochs=100
  uv run python -m tp_3.study.ej2_sweep --exp all --seeds 3 --set max_epochs=100
  uv run python -m tp_3.study.ej2_sweep --exp best          # only re-evaluate the best model + charts
"""
import argparse
import contextlib
import inspect
import io
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tp_3.__main__ import PROJECT_ROOT, features_to_inputs
from tp_3.study.ej1_study import load_config, standardize
from tp_3.engine.activation_functions.identity import Identity
from tp_3.engine.activation_functions.relu import ReLU
from tp_3.engine.activation_functions.sigmoid import Sigmoid
from tp_3.engine.activation_functions.tanh import Tanh
from tp_3.engine.loss_functions.bce import BinaryCrossEntropy
from tp_3.engine.loss_functions.mse import MeanSquaredError
from tp_3.engine.models.mlp import MLP
from tp_3.engine.optimizers.adam import Adam
from tp_3.engine.optimizers.momentum import Momentum
from tp_3.engine.optimizers.sgd import SGD
from tp_3.metrics.ej2_metrics import write_ej2_plots

ACTIVATIONS = {"sigmoid": Sigmoid, "relu": ReLU, "tanh": Tanh, "linear": Identity}
OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}
LOSSES = {"mean_squared_error": MeanSquaredError, "binary_cross_entropy": BinaryCrossEntropy}
GROUPS = ("lr", "optimizer", "activation", "depth", "grid")
WIDTH_STRATEGIES = ("constant", "halving", "budget")


# ------------------------------------------------------------------------------ data
@dataclass
class DigitsData:
    train_x: np.ndarray
    train_y: np.ndarray
    train_ids: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_ids: np.ndarray
    labels: list[str]
    raw_train: np.ndarray            # un-scaled pixels (for the example mosaics)
    test_x: np.ndarray | None = None
    test_y: np.ndarray | None = None
    test_ids: np.ndarray | None = None
    raw_test: np.ndarray | None = None


def load_digits(config: dict[str, Any]) -> DigitsData:
    frame = pd.read_csv(config["dataset_path"])
    target = "label" if "label" in frame.columns else frame.columns[0]
    inputs = features_to_inputs(frame.drop(columns=[target]))
    train_targets = frame[target].to_numpy()

    test_inputs = test_targets = None
    test_path = config.get("test_dataset_path")
    if test_path and Path(test_path).exists():
        test_frame = pd.read_csv(test_path)
        test_inputs = features_to_inputs(test_frame.drop(columns=[target]))
        test_targets = test_frame[target].to_numpy()
    elif test_path:
        print(f"[!] test dataset not found at {test_path}: the best model will be evaluated on validation instead")

    # The class space is fixed (digits 0-9): labels come from the UNION of train and test.
    # A class missing from train (e.g. the 8) keeps its output neuron but is never learned;
    # its failures in "production" (test) are part of the analysis.
    all_targets = train_targets if test_targets is None else np.concatenate([train_targets, test_targets])
    labels = np.unique(all_targets)
    only_in_test = sorted(set(labels.tolist()) - set(train_targets.tolist()))
    if only_in_test:
        print(f"[!] classes present in test but NOT in train: {only_in_test} "
              f"(the model cannot learn them; expected recall = 0 in production)")
    lookup = {v: k for k, v in enumerate(labels.tolist())}
    ids = np.array([lookup[v] for v in train_targets])
    onehot = np.eye(len(labels))[ids]

    order = np.random.default_rng(42).permutation(len(frame))      # same split rule as the main script
    cut = int(len(order) * config["split_ratio"])
    tr, va = order[:cut], order[cut:]

    test_ids = None
    if test_targets is not None:
        test_ids = np.array([lookup[v] for v in test_targets])

    others = [inputs[va]] + ([test_inputs] if test_inputs is not None else [])
    scaled = standardize(inputs[tr], others, config.get("scaling", "standardization"))
    data = DigitsData(scaled[0], onehot[tr], ids[tr], scaled[1], onehot[va], ids[va], [str(v) for v in labels],
                      inputs[tr])
    if test_inputs is not None:
        data.test_x, data.test_ids, data.raw_test = scaled[2], test_ids, test_inputs
        data.test_y = np.eye(len(labels))[test_ids]
    return data


# ------------------------------------------------------------------------ architecture
def n_params(sizes: list[int]) -> int:
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def hidden_widths(strategy: str, depth: int, width: int, n_in: int, n_out: int, budget: int) -> list[int]:
    """constant: same width everywhere | halving: width, width/2, ... | budget: constant width that keeps the
    total parameter count closest to `budget` (depths compared at equal capacity)."""
    if depth <= 0:
        return []
    if strategy == "constant":
        return [width] * depth
    if strategy == "halving":
        return [max(n_out, width // 2 ** i) for i in range(depth)]
    best = min(range(1, 8193), key=lambda w: abs(n_params([n_in] + [w] * depth + [n_out]) - budget))
    return [best] * depth


# ----------------------------------------------------------------------------- model
def _make(cls: Any, config: dict[str, Any]) -> Any:
    params = inspect.signature(cls.__init__).parameters
    return cls(**{k: config[k] for k in params if k in config})


def build_mlp(sizes: list[int], hidden_act: str, output_act: str, optimizer: str, loss: str,
              config: dict[str, Any], seed: int) -> MLP:
    opt_cls = OPTIMIZERS[optimizer]
    factory = (partial(opt_cls, alpha=config["momentum_beta"])
               if "alpha" in inspect.signature(opt_cls.__init__).parameters and "momentum_beta" in config else opt_cls)
    np.random.seed(seed)
    return MLP(sizes, _make(ACTIVATIONS[hidden_act], config), _make(ACTIVATIONS[output_act], config),
               _make(LOSSES[loss], config), factory)


def predict(model: MLP, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    return np.vstack([model.forward(x[i:i + batch]) for i in range(0, len(x), batch)])


def _finite(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


class CurveTracker:
    """epoch_callback for MLP.fit: loss and accuracy on train and validation."""

    def __init__(self, data: DigitsData) -> None:
        self.data, self.rows = data, []

    def __call__(self, epoch: int, model: MLP) -> None:
        row: dict[str, Any] = {"epoch": epoch}
        for split, x, y, ids in (("train", self.data.train_x, self.data.train_y, self.data.train_ids),
                                 ("val", self.data.val_x, self.data.val_y, self.data.val_ids)):
            out = predict(model, x)
            row[f"{split}_loss"] = _finite(model.loss_fn.compute(out, y))
            row[f"{split}_acc"] = float(np.mean(np.argmax(out, axis=1) == ids))
        self.rows.append(row)


def run_once(data: DigitsData, spec: dict[str, Any], config: dict[str, Any], seed: int, epochs: int,
             curve_points: int) -> tuple[dict[str, Any], MLP]:
    sizes = [data.train_x.shape[1], *spec["hidden"], len(data.labels)]
    model = build_mlp(sizes, spec["hidden_act"], spec["output_act"], spec["optimizer"], config["loss"], config, seed)
    tracker = CurveTracker(data)
    points = {int(e) for e in np.round(np.linspace(1, epochs, max(1, min(curve_points, epochs))))}
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):   # fit() always prints epoch 1; progress is reported per run
        model.fit(data.train_x, data.train_y, batch_size=config["batch_size"], epochs=epochs, lr=spec["lr"],
                  print_every=epochs + 1, epoch_callback=tracker, callback_epochs=points)
    seconds = time.perf_counter() - start
    predicted = np.argmax(predict(model, data.val_x), axis=1)
    confusion = np.zeros((len(data.labels),) * 2, dtype=int)
    np.add.at(confusion, (data.val_ids, predicted), 1)
    last = tracker.rows[-1]
    record = {"label": spec["label"], "experiment": spec["experiment"], "x": spec["x"], "seed": seed,
              "lr": spec["lr"], "optimizer": spec["optimizer"], "hidden_activation": spec["hidden_act"],
              "architecture": sizes, "depth": len(spec["hidden"]), "n_params": n_params(sizes), "seconds": seconds,
              "final_train_loss": last["train_loss"], "final_val_loss": last["val_loss"],
              "final_train_acc": last["train_acc"], "final_val_acc": last["val_acc"],
              "history": tracker.rows, "confusion": confusion.tolist()}
    return record, model


# ------------------------------------------------------------------------- experiments
def plan(args: argparse.Namespace, config: dict[str, Any], n_in: int, n_out: int, groups) -> list[dict[str, Any]]:
    base_hidden = list(config["architecture"][1:-1])
    base_act, base_lr, base_opt = config["activation"], config["learning_rate"], config["optimizer"]
    budget = args.param_budget or n_params([n_in, *base_hidden, n_out])
    width = args.width or base_hidden[0]
    widths = lambda d: hidden_widths(args.width_strategy, d, width, n_in, n_out, budget)
    lrs = sorted(set(args.lrs) | {base_lr})
    common = dict(lr=base_lr, optimizer=base_opt, hidden=base_hidden, hidden_act=base_act, output_act=base_act)
    specs: list[dict[str, Any]] = []
    if "lr" in groups:
        specs += [{**common, "experiment": "lr", "x": lr, "label": f"lr={lr:g}", "lr": lr} for lr in lrs]
    if "optimizer" in groups:
        specs += [{**common, "experiment": "optimizer", "x": o, "label": o, "optimizer": o} for o in args.optimizers]
    if "activation" in groups:
        specs += [{**common, "experiment": "activation", "x": a, "label": a, "hidden_act": a} for a in args.activations]
    if "depth" in groups:
        specs += [{**common, "experiment": "depth", "x": d, "label": f"{d} capas ocultas", "hidden": widths(d)}
                  for d in args.depths]
    if "grid" in groups:
        specs += [{**common, "experiment": "grid", "x": None, "label": f"{o} / lr={lr:g}", "optimizer": o, "lr": lr}
                  for o in args.optimizers for lr in lrs]
    return specs


# ------------------------------------------------------------------------ best model
def save_best(out: Path, model: MLP, record: dict[str, Any], spec: dict[str, Any], config: dict[str, Any],
              epochs: int) -> None:
    np.savez_compressed(out / "best_model.npz", **{f"{k}{i}": getattr(layer, k)
                                                   for i, layer in enumerate(model.layers) for k in ("W", "b")})
    (out / "best_model.json").write_text(json.dumps({
        "label": record["label"], "experiment": record["experiment"], "seed": record["seed"],
        "val_acc": record["final_val_acc"], "architecture": record["architecture"], "n_params": record["n_params"],
        "hidden_act": spec["hidden_act"], "output_act": spec["output_act"], "optimizer": spec["optimizer"],
        "lr": spec["lr"], "loss": config["loss"], "batch_size": config["batch_size"], "epochs": epochs,
        "scaling": config.get("scaling", "standardization"), "momentum_beta": config.get("momentum_beta"),
        "beta": config.get("beta")}))


def evaluate_best(out: Path, data: DigitsData, config: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuilds the stored best model and measures it on train / validation / test (test = 'production')."""
    if not (out / "best_model.json").exists():
        return None
    desc = json.loads((out / "best_model.json").read_text())
    weights = np.load(out / "best_model.npz")
    cfg = {**config, "momentum_beta": desc["momentum_beta"] or 0.9, "beta": desc["beta"] or 1.0}
    model = build_mlp(desc["architecture"], desc["hidden_act"], desc["output_act"], desc["optimizer"], desc["loss"],
                      cfg, 0)
    for i, layer in enumerate(model.layers):
        layer.W, layer.b = weights[f"W{i}"], weights[f"b{i}"]

    n = len(data.labels)
    sets = {"train": (data.train_x, data.train_y, data.train_ids), "val": (data.val_x, data.val_y, data.val_ids)}
    if data.test_x is not None:
        sets["test"] = (data.test_x, data.test_y, data.test_ids)
    acc, loss, outputs = {}, {}, {}
    for name, (x, y, ids) in sets.items():
        outputs[name] = predict(model, x)
        acc[name] = float(np.mean(np.argmax(outputs[name], axis=1) == ids))
        loss[name] = _finite(model.loss_fn.compute(outputs[name], y))
    eval_set = "test" if "test" in sets else "val"
    ids, out_eval = sets[eval_set][2], outputs[eval_set]
    pred = np.argmax(out_eval, axis=1)
    confusion = np.zeros((n, n), dtype=int)
    np.add.at(confusion, (ids, pred), 1)
    tp = np.diag(confusion).astype(float)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    precision, recall = div(tp, confusion.sum(axis=0).astype(float)), div(tp, confusion.sum(axis=1).astype(float))
    f1 = div(2 * precision * recall, precision + recall)

    raw = data.raw_test if eval_set == "test" else None
    wrong = np.where(pred != ids)[0]
    wrong = wrong[np.argsort(-out_eval[wrong, pred[wrong]])][:16]      # most confidently wrong first
    mis = ([{"true": data.labels[ids[k]], "pred": data.labels[pred[k]], "score": float(out_eval[k, pred[k]]),
             "pixels": raw[k].tolist()} for k in wrong] if raw is not None else [])
    result = {"run": desc, "eval_set": eval_set, "accuracy": acc, "loss": loss, "labels": data.labels,
              "confusion": confusion.tolist(), "precision": precision.tolist(), "recall": recall.tolist(),
              "f1": f1.tolist(), "support": confusion.sum(axis=1).tolist(), "misclassified": mis}
    (out / "best.json").write_text(json.dumps(result))
    print(f"best model [{desc['label']} / {desc['experiment']}]: " +
          ", ".join(f"{k}={v:.4f}" for k, v in acc.items()))
    return result


def write_protocol(out: Path, data: DigitsData, config: dict[str, Any]) -> None:
    n = len(data.labels)
    count = lambda ids: np.bincount(ids, minlength=n).tolist()
    side = int(round(np.sqrt(data.raw_train.shape[1])))
    examples = []
    for c in range(n):
        idx = np.where(data.train_ids == c)[0][:3]
        examples.append([data.raw_train[i].tolist() for i in idx])
    (out / "protocol.json").write_text(json.dumps({
        "labels": data.labels, "split_ratio": config["split_ratio"], "side": side,
        "sizes": {"train": len(data.train_x), "val": len(data.val_x),
                  "test": None if data.test_x is None else len(data.test_x)},
        "counts": {"train": count(data.train_ids), "val": count(data.val_ids),
                   "test": None if data.test_ids is None else count(data.test_ids)},
        "dataset": Path(config["dataset_path"]).name, "test_dataset": Path(config.get("test_dataset_path") or "").name,
        "examples": examples}))


# -------------------------------------------------------------------------------- CLI
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment sweep runner for Exercise 2 (digits)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-c", "--config", default="configs/ej2.toml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override config values")
    p.add_argument("--exp", default="all", choices=[*GROUPS, "all", "protocol", "best"],
                   help="Experiment group; 'protocol'/'best' only rebuild those files and charts")
    p.add_argument("--out-dir", default="metrics/ej2")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    p.add_argument("--optimizers", nargs="+", default=["sgd", "momentum", "adam"], choices=list(OPTIMIZERS))
    p.add_argument("--activations", nargs="+", default=["sigmoid", "relu", "tanh"], choices=list(ACTIVATIONS))
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--width-strategy", choices=WIDTH_STRATEGIES, default="halving")
    p.add_argument("--width", type=int, default=None, help="First hidden width (default: from architecture)")
    p.add_argument("--param-budget", type=int, default=None)
    p.add_argument("--seeds", type=int, default=1, help="Repetitions with different weight init")
    p.add_argument("--curve-points", type=int, default=100, help="Epochs sampled per learning curve")
    return p


def main() -> None:
    args = build_cli().parse_args()
    config = load_config(args.config, args.set)
    for key in ("test_dataset_path",):
        if config.get(key) and not Path(config[key]).is_absolute():
            config[key] = str((PROJECT_ROOT / config[key]).resolve())
    out = Path(args.out_dir)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    data = load_digits(config)
    n_in, n_out = data.train_x.shape[1], len(data.labels)
    arch = config["architecture"]
    if arch[0] != n_in or arch[-1] != n_out:
        raise SystemExit(f"architecture {arch} does not match the data ({n_in} inputs, {n_out} classes)")
    epochs = config["max_epochs"]
    print(f"digits: train={len(data.train_x)}, val={len(data.val_x)}, "
          f"test={'-' if data.test_x is None else len(data.test_x)}; {n_in} inputs, {n_out} classes")
    write_protocol(out, data, config)

    groups = list(GROUPS) if args.exp == "all" else [args.exp] if args.exp in GROUPS else []
    results_path = out / "results.json"
    stored = json.loads(results_path.read_text()) if results_path.exists() else {"meta": {}, "runs": []}
    if groups:
        specs = plan(args, config, n_in, n_out, groups)
        print(f"=== {len(specs)} configs x {args.seeds} seed(s), {epochs} epochs, batch {config['batch_size']} ===")
        runs, cache = [], {}
        best_acc, best = -1.0, None
        for spec in specs:
            for seed in range(args.seeds):
                key = (spec["lr"], spec["optimizer"], tuple(spec["hidden"]), spec["hidden_act"], spec["output_act"], seed)
                if key not in cache:
                    cache[key] = run_once(data, spec, config, 42 + seed, epochs, args.curve_points)
                record, model = cache[key]
                runs.append({**record, "label": spec["label"], "experiment": spec["experiment"], "x": spec["x"]})
                print(f"  {spec['experiment']:<10} {spec['label']:<24} seed={seed} arch={record['architecture']} "
                      f"val_acc={record['final_val_acc']:.4f} ({record['seconds']:.1f}s)")
                if record["final_val_acc"] > best_acc:
                    best_acc, best = record["final_val_acc"], (record, spec, model)
        stored["runs"] = [r for r in stored["runs"] if r["experiment"] not in groups] + runs
        stored["meta"].update(epochs=epochs, batch_size=config["batch_size"], seeds=args.seeds, labels=data.labels,
                              base_optimizer=config["optimizer"], base_lr=config["learning_rate"],
                              width_strategy=args.width_strategy, hidden_activation=config["activation"],
                              loss=config["loss"], split_ratio=config["split_ratio"])
        results_path.write_text(json.dumps(stored))
        print(f"Results written to: {results_path}")
        previous = (json.loads((out / "best_model.json").read_text())["val_acc"]
                    if (out / "best_model.json").exists() else -1.0)
        if best is not None and best_acc > previous:
            save_best(out, best[2], best[0], best[1], config, epochs)
            print(f"New best model: {best[1]['experiment']} / {best[1]['label']} (val_acc={best_acc:.4f})")

    evaluate_best(out, data, config)
    if not args.no_plots:
        if stored["runs"]:
            write_ej2_plots(out, stored)
        from tp_3.metrics.ej2_metrics_extra import write_ej2_extra_plots
        write_ej2_extra_plots(out)


if __name__ == "__main__":
    main()