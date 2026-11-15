"""Loss landscape of a trained MLP (Li et al., 2018, filter-normalised random directions).

  theta(alpha, beta) = theta* + alpha * d1 + beta * d2,   theta* in R^N,  N = #weights + #biases

* d1, d2 are random Gaussian directions rescaled FILTER BY FILTER (a filter = the incoming weights of one
  neuron) so each filter has the same norm as the matching filter of theta*. d2 is first made orthogonal to d1
  filter by filter, which keeps d1 . d2 = 0 exactly after the rescaling.
* Biases: direction 0 by default (as in the paper); --bias-direction normalize treats each layer's bias vector
  as one more filter.

Trains the model from the config (same construction rules as the main script), evaluates the grid and stores
landscape.npz / landscape.json in --out-dir. Charts: tp_3/metrics/landscape_metrics.py.

  uv run python -m tp_3.loss_landscape -c configs/ej2.toml --range 1.0 --grid 25 --max-samples 5000
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tp_3.__main__ import FRAUD_TARGET, PROJECT_ROOT, features_to_inputs
from .ej1_study import build_model, load_config, standardize


# --------------------------------------------------------------------------------- data
def load_task(config: dict[str, Any]):
    """Train/validation split of the configured dataset (fraud regression or digits classification)."""
    frame = pd.read_csv(config["dataset_path"], usecols=lambda c: c != "flagged_fraud")
    target_column = next((c for c in (FRAUD_TARGET, "label") if c in frame.columns), frame.columns[-1])
    targets = frame[target_column].to_numpy()
    inputs = features_to_inputs(frame.drop(columns=[target_column]))
    if config["model_type"] == "multilayer":
        sizes = config["architecture"]
        if inputs.shape[1] != sizes[0]:
            raise ValueError(f"Architecture expects {sizes[0]} inputs, dataset has {inputs.shape[1]} features.")
        out_size = sizes[-1]
    else:
        out_size = 1
    if out_size == 1:
        outputs = targets.astype(np.float64).reshape(-1, 1)
    else:
        labels, ids = np.unique(targets, return_inverse=True)
        if len(labels) > out_size:
            raise ValueError("Architecture output size is smaller than the number of classes.")
        outputs = np.eye(out_size)[ids]
    order = (np.argsort(frame["timestamp"].to_numpy(), kind="stable") if "timestamp" in frame.columns
             else np.random.default_rng(42).permutation(len(frame)))
    cut = int(len(order) * config["split_ratio"])
    tr, va = order[:cut], order[cut:]
    x_tr, x_va = standardize(inputs[tr], [inputs[va]], config.get("scaling", "standardization"))
    return x_tr, outputs[tr], x_va, outputs[va]


# ------------------------------------------------------------------------- directions
def _ortho_pair(ref: np.ndarray, rng: np.random.Generator):
    """Two random directions with the same filter structure as `ref` (rows = filters), filter-normalised,
    orthogonal to each other filter by filter."""
    g1, g2 = rng.standard_normal(ref.shape), rng.standard_normal(ref.shape)
    if ref.shape[1] == 1:                       # a 1-D filter cannot hold two orthogonal directions
        g2 = np.zeros_like(g2)
    else:
        g2 = g2 - (np.sum(g1 * g2, axis=1, keepdims=True) / np.sum(g1 * g1, axis=1, keepdims=True)) * g1
    ref_norm = np.linalg.norm(ref, axis=1, keepdims=True)
    out = []
    for g in (g1, g2):
        norm = np.linalg.norm(g, axis=1, keepdims=True)
        out.append(np.where(norm > 0, g / np.where(norm > 0, norm, 1.0), 0.0) * ref_norm)
    return out[0], out[1]


def make_directions(params, rng, bias_direction: str):
    d1, d2 = [], []
    for W, b in params:                          # W is (in, out): filters are its columns
        w1, w2 = _ortho_pair(W.T, rng)
        if bias_direction == "normalize":
            b1, b2 = _ortho_pair(b[None, :], rng)
            b1, b2 = b1[0], b2[0]
        else:
            b1, b2 = np.zeros_like(b), np.zeros_like(b)
        d1.append((w1.T, b1))
        d2.append((w2.T, b2))
    return d1, d2


def _dot(a, b) -> float:
    return float(sum(np.sum(wa * wb) + np.sum(ba * bb) for (wa, ba), (wb, bb) in zip(a, b)))


# ------------------------------------------------------------------------ evaluation
def loss_on(model, x: np.ndarray, y: np.ndarray, batch: int) -> float:
    preds = np.vstack([model.forward(x[i:i + batch]) for i in range(0, len(x), batch)])
    return float(model.loss_fn.compute(preds, y))


def set_point(model, params, d1, d2, alpha: float, beta: float) -> None:
    for layer, (W, b), (w1, b1), (w2, b2) in zip(model.layers, params, d1, d2):
        layer.W = W + alpha * w1 + beta * w2
        layer.b = b + alpha * b1 + beta * b2


def main() -> None:
    p = argparse.ArgumentParser(description="Loss landscape of a trained MLP")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out-dir", default="metrics/landscape")
    p.add_argument("--range", type=float, default=1.0, dest="span", help="alpha, beta in [-range, range]")
    p.add_argument("--grid", type=int, default=25, help="points per axis (made odd so (0, 0) is on the grid)")
    p.add_argument("--data", choices=["train", "val"], default="train", help="Set the loss is evaluated on")
    p.add_argument("--max-samples", type=int, default=None, help="Fixed random subset of that set (speed)")
    p.add_argument("--seed", type=int, default=0, help="Seed of the random directions")
    p.add_argument("--bias-direction", choices=["zero", "normalize"], default="zero")
    p.add_argument("--eval-batch", type=int, default=8192)
    p.add_argument("--zscale", choices=["linear", "log"], default="linear")
    p.add_argument("--tol", type=float, default=0.1, help="Flatness tolerance (fraction of the grid loss range)")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    config = load_config(args.config, args.set)
    x_tr, y_tr, x_va, y_va = load_task(config)
    model = build_model(config, x_tr.shape[1])
    print(f"Training {config['model_type']} ({config['activation']}) for {config['max_epochs']} epochs...")
    model.fit(x_tr, y_tr, epochs=config["max_epochs"], lr=config["learning_rate"],
              print_every=max(1, config["max_epochs"] // 10), batch_size=config["batch_size"])

    x, y = (x_tr, y_tr) if args.data == "train" else (x_va, y_va)
    if args.max_samples and len(x) > args.max_samples:
        keep = np.sort(np.random.default_rng(0).choice(len(x), args.max_samples, replace=False))
        x, y = x[keep], y[keep]

    params = [(l.W.copy(), l.b.copy()) for l in model.layers]
    n_w, n_b = sum(W.size for W, _ in params), sum(b.size for _, b in params)
    d1, d2 = make_directions(params, np.random.default_rng(args.seed), args.bias_direction)
    n1, n2 = np.sqrt(_dot(d1, d1)), np.sqrt(_dot(d2, d2))
    cosine = _dot(d1, d2) / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
    print(f"N = {n_w + n_b:,} parameters ({n_w:,} weights + {n_b:,} biases); cos(d1, d2) = {cosine:.2e}")

    n = args.grid + (1 - args.grid % 2)
    alphas = np.linspace(-args.span, args.span, n)
    betas = alphas.copy()
    h = alphas[1] - alphas[0]
    ev = lambda a, b: (set_point(model, params, d1, d2, a, b), loss_on(model, x, y, args.eval_batch))[1]

    center = ev(0.0, 0.0)
    axis = {"alpha_plus": ev(h, 0.0), "alpha_minus": ev(-h, 0.0), "beta_plus": ev(0.0, h), "beta_minus": ev(0.0, -h)}
    z = np.empty((n, n))
    for j, beta in enumerate(betas):
        for i, alpha in enumerate(alphas):
            z[j, i] = ev(alpha, beta)
        print(f"  row {j + 1}/{n}", end="\r")
    set_point(model, params, d1, d2, 0.0, 0.0)  # restore theta*
    print(f"\ncenter loss = {center:.6g}; grid min = {np.nanmin(z):.6g}; grid max = {np.nanmax(z):.6g}")

    out = Path(args.out_dir)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "landscape.npz", alphas=alphas, betas=betas, z=z)
    (out / "landscape.json").write_text(json.dumps({
        "center_loss": center, "axis": axis, "h": float(h), "range": args.span, "grid": n, "data": args.data,
        "n_samples": int(len(x)), "n_params": int(n_w + n_b), "n_weights": int(n_w), "n_biases": int(n_b),
        "cosine_d1_d2": cosine, "bias_direction": args.bias_direction, "seed": args.seed, "zscale": args.zscale,
        "tol": args.tol, "model_type": config["model_type"], "activation": config["activation"],
        "optimizer": config["optimizer"], "learning_rate": config["learning_rate"], "epochs": config["max_epochs"],
        "architecture": [params[0][0].shape[0]] + [W.shape[1] for W, _ in params]}, indent=2))
    print(f"landscape -> {out / 'landscape.npz'}")
    if not args.no_plots:
        from ..metrics.landscape_metrics import write_landscape_plots
        write_landscape_plots(out)


if __name__ == "__main__":
    main()