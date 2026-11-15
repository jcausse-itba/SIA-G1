"""Exercise 3 sweep & analysis runner (Independent execution).

Features:
1. Retrains the best fixed architecture with `digits.csv` and `more_data_digits.csv`.
2. Evaluates performance improvement (Ej2 vs Ej3) autonomously without depending on Ej2 runs.
3. Evaluates robustness to Gaussian noise on test samples.
4. Generates interpretability visualizations (first-layer weight heatmaps & input saliency maps).
5. Exports structured JSON metrics for plotting via ej3_metrics.py.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tp_3.__main__ import PROJECT_ROOT
from tp_3.engine.models.mlp import MLP
from tp_3.study.ej1_study import load_config
from tp_3.study.ej2_sweep import DigitsData, build_mlp, load_digits, n_params, predict

# -----------------------------------------------------------------------------
# CONSTANTES DE CONFIGURACIÓN GANADORAS (Sin dependencia del Ejercicio 2)
# -----------------------------------------------------------------------------
DEFAULT_EJ2_DATASET_PATH = PROJECT_ROOT / "data" / "digits.csv"
DEFAULT_EJ3_DATASET_PATH = PROJECT_ROOT / "data" / "more_digits.csv"
DEFAULT_TEST_DATASET_PATH = PROJECT_ROOT / "data" / "digits_test.csv"

# Mejor configuración seleccionada del Ejercicio 2
BEST_ARCH = [784, 64, 32, 10]        
BEST_HIDDEN_ACT = "relu"
BEST_OUTPUT_ACT = "sigmoid"
BEST_OPTIMIZER = "momentum"     # Configuración ganadora: Momentum
BEST_LR = 0.1                   # Tasa de aprendizaje óptima: 0.1


def load_dataset_by_path(config: dict[str, Any], dataset_path: Path) -> DigitsData:
    """Utility to load a specific dataset while keeping test_dataset_path consistent."""
    cfg = dict(config)
    cfg["dataset_path"] = str(dataset_path.resolve())
    
    test_path = cfg.get("test_dataset_path", DEFAULT_TEST_DATASET_PATH)
    if test_path:
        cfg["test_dataset_path"] = str((PROJECT_ROOT / Path(test_path)).resolve())
        
    return load_digits(cfg)


def evaluate_noise_robustness(
    model: MLP,
    x_test: np.ndarray,
    y_test_ids: np.ndarray,
    noise_levels: list[float],
) -> list[dict[str, Any]]:
    """Evaluates test accuracy and loss under varying levels of Gaussian noise."""
    results = []
    rng = np.random.default_rng(42)
    for sigma in noise_levels:
        noise = rng.normal(0, sigma, x_test.shape)
        noisy_x = x_test + noise
        probs = predict(model, noisy_x)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_test_ids))
        loss = float(model.loss_fn.compute(probs, np.eye(probs.shape[1])[y_test_ids]))
        results.append({"sigma": sigma, "accuracy": acc, "loss": loss})
    return results


def compute_saliency_maps(
    model: MLP,
    x_samples: np.ndarray,
    y_true_ids: np.ndarray,
) -> list[np.ndarray]:
    """Computes gradient saliency maps (|dOutput/dX|) for sample inputs."""
    saliencies = []
    for x, target in zip(x_samples, y_true_ids):
        x_in = x.copy().reshape(1, -1)

        # 1. Forward pass para calcular y guardar z e inputs internos
        out = x_in
        for layer in model.layers:
            out = layer.feed_forward(out)

        # 2. Gradiente unitario en la neurona objetivo (1.0 en la clase real, 0.0 en las demás)
        n_outputs = len(model.layers[-1].b)
        grad = np.zeros((1, n_outputs), dtype=np.float64)
        grad[0, target] = 1.0

        # 3. Retropropagación desempaquetando solo grad_input (el primer elemento devuelto)
        for layer in reversed(model.layers):
            grad, _, _ = layer.backward(grad)

        saliencies.append(np.abs(grad.flatten()))
    return saliencies


def train_and_eval(
    data: DigitsData,
    config: dict[str, Any],
    epochs: int,
) -> tuple[MLP, dict[str, dict[str, float]]]:
    """Builds, trains and evaluates the best model configuration on a given DigitsData."""
    model = build_mlp(
        BEST_ARCH,
        BEST_HIDDEN_ACT,
        BEST_OUTPUT_ACT,
        BEST_OPTIMIZER,
        config["loss"],
        config,
        seed=42,
    )

    model.fit(
        data.train_x,
        data.train_y,
        batch_size=config["batch_size"],
        epochs=epochs,
        lr=BEST_LR,
        print_every=max(1, epochs // 5),
    )

    metrics = {}
    for name, x, y, ids in (
        ("train", data.train_x, data.train_y, data.train_ids),
        ("val", data.val_x, data.val_y, data.val_ids),
        ("test", data.test_x, data.test_y, data.test_ids),
    ):
        if x is None:
            continue
        probs = predict(model, x)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == ids))
        loss = float(model.loss_fn.compute(probs, y))
        metrics[name] = {"accuracy": acc, "loss": loss}

    return model, metrics


def run_ej3(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.set)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or config["max_epochs"]

    # 1. Cargar y entrenar el modelo base con Momentum (LR=0.1) en digits.csv
    print(f"=== Entrenando Modelo Base Ej2 ({BEST_OPTIMIZER.upper()}, LR={BEST_LR}) en digits.csv ===")
    ej2_data = load_dataset_by_path(config, DEFAULT_EJ2_DATASET_PATH)
    _, ej2_metrics = train_and_eval(ej2_data, config, epochs)

    # 2. Cargar y entrenar el modelo ampliado con Momentum (LR=0.1) en more_digits.csv
    print(f"=== Entrenando Modelo Ampliado Ej3 ({BEST_OPTIMIZER.upper()}, LR={BEST_LR}) en more_digits.csv ===")
    ej3_data = load_dataset_by_path(config, DEFAULT_EJ3_DATASET_PATH)
    ej3_model, ej3_metrics = train_and_eval(ej3_data, config, epochs)

    # 3. Matriz de Confusión sobre Test
    n_classes = len(ej3_data.labels)
    test_probs = predict(ej3_model, ej3_data.test_x)
    test_preds = np.argmax(test_probs, axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (ej3_data.test_ids, test_preds), 1)

    # 4. Prueba de Robustez al Ruido Gaussiano
    noise_sigmas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    noise_results = evaluate_noise_robustness(
        ej3_model, ej3_data.test_x, ej3_data.test_ids, noise_sigmas
    )

    # 5. Interpretabilidad (Campos Receptores y Mapas de Saliencia)
    first_layer_weights = ej3_model.layers[0].W.copy()
    sample_indices = np.random.default_rng(42).choice(len(ej3_data.test_x), size=8, replace=False)
    sample_saliencies = compute_saliency_maps(
        ej3_model, ej3_data.test_x[sample_indices], ej3_data.test_ids[sample_indices]
    )

    # 6. Exportar los resultados a JSON
    ej3_summary = {
        "architecture": BEST_ARCH,
        "optimizer": BEST_OPTIMIZER,
        "learning_rate": BEST_LR,
        "n_params": n_params(BEST_ARCH),
        "ej2_metrics": ej2_metrics,
        "ej3_metrics": ej3_metrics,
        "confusion": cm.tolist(),
        "noise_robustness": noise_results,
        "first_layer_weights": first_layer_weights.tolist(),
        "saliency_samples": [
            {
                "true": ej3_data.labels[ej3_data.test_ids[idx]],
                "pred": ej3_data.labels[test_preds[idx]],
                "pixels": ej3_data.raw_test[idx].tolist(),
                "saliency": sal.tolist(),
            }
            for idx, sal in zip(sample_indices, sample_saliencies)
        ],
        "labels": ej3_data.labels,
    }

    summary_file = out_dir / "ej3_results.json"
    summary_file.write_text(json.dumps(ej3_summary))
    print(f"\n[✓] Resultados del Ejercicio 3 guardados exitosamente en: {summary_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exercise 3 execution pipeline (Independent)")
    p.add_argument("-c", "--config", default="configs/ej2.toml")
    p.add_argument("--out-dir", default="metrics/ej3")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run_ej3(p.parse_args())