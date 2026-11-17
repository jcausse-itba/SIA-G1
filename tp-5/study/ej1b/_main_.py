from __future__ import annotations

import argparse
import csv
from html import escape
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tp_5.config.validator import AutoencoderConfig, DatasetBundle
from tp_5.training_report import TrainingReporter


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "study" / "ej1b" / "resultados"
DATASET = ROOT / "doc" / "font.h"
ARCHITECTURES = (
    [35, 32, 16, 2],
    [35, 48, 24, 2],
    [35, 64, 32, 2],
)
OPTIMIZERS = ("sgd", "momentum", "adam")
LEARNING_RATES = (0.001, 0.004, 0.01)
TRAINING_NOISE_LEVELS = (0.1, 0.2, 0.3)
LR_SCHEDULERS = ("constant", "cosine_annealing", "exponential_decay", "linear_decay")
EVALUATION_NOISE_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
SELECTION_NOISE_LEVELS = EVALUATION_NOISE_LEVELS[1:]
EVALUATION_REPEATS = 5
EVALUATION_SEED = 2026
EXPERIMENT_EPOCHS = 500
FINAL_EPOCHS = 1500
BATCH_SIZES = (4, 8, 16)
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MIN_LR_RATIOS = (0.01, 0.02, 0.05, 0.10, 0.20)
EXPONENTIAL_DECAY_RATES = (0.01, 0.02, 0.05, 0.10, 0.20)
MAX_PIXEL_ERRORS_WITHIN_TOLERANCE = 1
SEED = 42


def make_config(
    architecture: list[int],
    *,
    optimizer: str,
    learning_rate: float,
    noise_level: float,
    batch_size: int,
    max_epochs: int,
    lr_scheduler: str = "constant",
    min_lr_ratio: float = 0.02,
    decay_rate: float = 0.05,
    threshold: float = 0.5,
) -> AutoencoderConfig:
    return AutoencoderConfig(
        model_type="denoising_autoencoder",
        dataset_path=DATASET,
        encoder_layers=architecture,
        hidden_activation="tanh",
        latent_activation="linear",
        output_activation="sigmoid",
        loss="binary_cross_entropy",
        optimizer=optimizer,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        min_lr_ratio=min_lr_ratio,
        decay_rate=decay_rate,
        noise_type="bit_flip",
        noise_level=noise_level,
        eval_noise_levels=list(EVALUATION_NOISE_LEVELS),
        batch_size=batch_size,
        max_epochs=max_epochs,
        threshold=threshold,
        seed=SEED,
        output_dir=OUTPUT_DIR,
    )


def make_evaluation_sets(
    config: AutoencoderConfig,
    dataset: DatasetBundle,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    targets = np.tile(dataset.inputs, (EVALUATION_REPEATS, 1))
    evaluation_sets: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for level in EVALUATION_NOISE_LEVELS:
        noise = config.build_noise(override_level=level)
        rng = np.random.default_rng(EVALUATION_SEED + round(level * 1000))
        evaluation_sets[level] = (noise.corrupt(targets, rng), targets)
    return evaluation_sets


def evaluate_profile(
    model: Any,
    evaluation_sets: dict[float, tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    by_level: dict[float, dict[str, float]] = {}
    for level, (noisy_inputs, clean_targets) in evaluation_sets.items():
        summary = model.evaluate_reconstruction(
            noisy_inputs,
            X_target=clean_targets,
            threshold=threshold,
        )
        corrupted_per_pattern = np.sum(noisy_inputs != clean_targets, axis=1)
        pixel_errors = np.asarray(summary["pixel_errors"])
        patterns_within_tolerance = int(
            np.sum(pixel_errors <= MAX_PIXEL_ERRORS_WITHIN_TOLERANCE)
        )
        by_level[level] = {
            "corrupted_mean": float(np.mean(corrupted_per_pattern)),
            "reconstruction_mean": float(summary["mean_pixel_error"]),
            "reconstruction_std": float(np.std(pixel_errors)),
            "reconstruction_max": float(summary["max_pixel_error"]),
            "loss": float(summary["loss"]),
            "patterns_evaluated": int(summary["num_patterns"]),
            "patterns_within_tolerance": patterns_within_tolerance,
            "tolerance_pass_rate": float(
                patterns_within_tolerance / summary["num_patterns"]
            ),
        }

    level_means = [by_level[level]["reconstruction_mean"] for level in SELECTION_NOISE_LEVELS]
    level_pass_rates = [
        by_level[level]["tolerance_pass_rate"] for level in SELECTION_NOISE_LEVELS
    ]
    return {
        "by_level": by_level,
        "robustness_score": float(np.mean(level_means)),
        "robustness_tolerance_pass_rate": float(np.mean(level_pass_rates)),
        "worst_level_tolerance_pass_rate": float(np.min(level_pass_rates)),
        "clean_tolerance_pass_rate": by_level[0.0]["tolerance_pass_rate"],
        "clean_pattern_count": by_level[0.0]["patterns_evaluated"] // EVALUATION_REPEATS,
        "worst_level_mean_error": float(np.max(level_means)),
        "clean_mean_error": by_level[0.0]["reconstruction_mean"],
        "mean_loss": float(
            np.mean([by_level[level]["loss"] for level in SELECTION_NOISE_LEVELS])
        ),
    }


def train_experiment(
    config: AutoencoderConfig,
    dataset: DatasetBundle,
    evaluation_sets: dict[float, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    model = config.build_model(quantization_scale=dataset.quantization_scale)
    with redirect_stdout(io.StringIO()):
        history = model.fit(
            dataset.inputs,
            dataset.inputs,
            batch_size=config.batch_size,
            epochs=config.max_epochs,
            lr=config.learning_rate,
            print_every=config.max_epochs + 1,
        )
    return {
        "config": config,
        "model": model,
        "history": history,
        "epochs": len(history),
        "profile": evaluate_profile(model, evaluation_sets, config.threshold),
    }


def tune_threshold(
    experiment: dict[str, Any],
    evaluation_sets: dict[float, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    threshold_results = []
    for threshold in THRESHOLDS:
        profile = evaluate_profile(experiment["model"], evaluation_sets, threshold)
        threshold_results.append({"threshold": threshold, "profile": profile})

    selected_threshold = min(
        threshold_results,
        key=lambda item: (
            -item["profile"]["clean_tolerance_pass_rate"],
            -item["profile"]["robustness_tolerance_pass_rate"],
            item["profile"]["robustness_score"],
            abs(item["threshold"] - 0.5),
        ),
    )
    experiment["config"].threshold = selected_threshold["threshold"]
    experiment["profile"] = selected_threshold["profile"]
    experiment["threshold_results"] = threshold_results
    return experiment


def compare_configurations(
    dataset: DatasetBundle,
    evaluation_sets: dict[float, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    combinations = [
        (architecture, optimizer, learning_rate, noise_level, batch_size)
        for architecture in ARCHITECTURES
        for optimizer in OPTIMIZERS
        for learning_rate in LEARNING_RATES
        for noise_level in TRAINING_NOISE_LEVELS
        for batch_size in BATCH_SIZES
    ]
    experiments = []
    for index, (architecture, optimizer, learning_rate, noise_level, batch_size) in enumerate(
        combinations,
        start=1,
    ):
        if index == 1 or index % 25 == 0 or index == len(combinations):
            print(f"Entrenando combinación {index}/{len(combinations)}...")
        config = make_config(
            architecture,
            optimizer=optimizer,
            learning_rate=learning_rate,
            noise_level=noise_level,
            batch_size=batch_size,
            max_epochs=EXPERIMENT_EPOCHS,
        )
        experiment = train_experiment(config, dataset, evaluation_sets)
        experiments.append(tune_threshold(experiment, evaluation_sets))
    return experiments


def compare_lr_schedulers(
    best_config: AutoencoderConfig,
    dataset: DatasetBundle,
    evaluation_sets: dict[float, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    experiments = []
    for scheduler_name in LR_SCHEDULERS:
        if scheduler_name == "constant":
            scheduler_parameters = [(0.02, 0.05)]
        elif scheduler_name == "exponential_decay":
            scheduler_parameters = [(0.02, rate) for rate in EXPONENTIAL_DECAY_RATES]
        else:
            scheduler_parameters = [(ratio, 0.05) for ratio in MIN_LR_RATIOS]

        for min_lr_ratio, decay_rate in scheduler_parameters:
            choice = scheduler_choice(scheduler_name, min_lr_ratio, decay_rate)
            print(f"Evaluando {choice}...")
            config = make_config(
                best_config.encoder_layers,
                optimizer=best_config.optimizer,
                learning_rate=best_config.learning_rate,
                noise_level=best_config.noise_level,
                batch_size=best_config.batch_size,
                max_epochs=FINAL_EPOCHS,
                lr_scheduler=scheduler_name,
                min_lr_ratio=min_lr_ratio,
                decay_rate=decay_rate,
            )
            result = train_experiment(config, dataset, evaluation_sets)
            experiments.append(
                tune_threshold({**result, "choice": choice}, evaluation_sets)
            )
    return experiments


def scheduler_choice(scheduler_name: str, min_lr_ratio: float, decay_rate: float) -> str:
    if scheduler_name == "constant":
        return "constant"
    if scheduler_name == "exponential_decay":
        return f"{scheduler_name} (decay_rate={decay_rate:g})"
    return f"{scheduler_name} (min_lr_ratio={min_lr_ratio:g})"


def experiment_rank(experiment: dict[str, Any]) -> tuple[float, ...]:
    config = experiment["config"]
    profile = experiment["profile"]
    return (
        -profile["clean_tolerance_pass_rate"],
        -profile["robustness_tolerance_pass_rate"],
        profile["robustness_score"],
        profile["worst_level_mean_error"],
        profile["clean_mean_error"],
        profile["mean_loss"],
        float(parameter_count(config.encoder_layers)),
        float(config.batch_size),
        float(experiment["epochs"]),
    )


def parameter_count(encoder_layers: list[int]) -> int:
    layers = encoder_layers + list(reversed(encoder_layers[:-1]))
    return sum(source * target + target for source, target in zip(layers, layers[1:]))


def same_configuration(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_config = first["config"]
    second_config = second["config"]
    return (
        first_config.encoder_layers == second_config.encoder_layers
        and first_config.optimizer == second_config.optimizer
        and first_config.learning_rate == second_config.learning_rate
        and first_config.noise_level == second_config.noise_level
        and first_config.batch_size == second_config.batch_size
        and first_config.lr_scheduler == second_config.lr_scheduler
        and first_config.min_lr_ratio == second_config.min_lr_ratio
        and first_config.decay_rate == second_config.decay_rate
        and first_config.threshold == second_config.threshold
    )


def best_per_group(
    experiments: list[dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    best: dict[Any, dict[str, Any]] = {}
    for experiment in experiments:
        config = experiment["config"]
        if group_key == "architecture":
            group = tuple(config.encoder_layers)
            label = "-".join(map(str, config.encoder_layers))
        elif group_key == "optimizer":
            group = config.optimizer
            label = config.optimizer
        elif group_key == "learning_rate":
            group = config.learning_rate
            label = f"{config.learning_rate:g}"
        elif group_key == "batch_size":
            group = config.batch_size
            label = str(config.batch_size)
        else:
            group = config.noise_level
            label = f"{config.noise_level:g}"

        current = best.get(group)
        if current is None or experiment_rank(experiment) < experiment_rank(current):
            best[group] = {**experiment, "choice": label}
    return list(best.values())


def write_architecture_diagram(encoder_layers: list[int], path: Path) -> None:
    dimensions = encoder_layers + list(reversed(encoder_layers[:-1]))
    latent_index = len(encoder_layers) - 1
    last_index = len(dimensions) - 1
    positions = [
        np.linspace(0.12, 0.88, min(dimension, 8)).tolist()
        for dimension in dimensions
    ]
    fig = go.Figure()

    for layer_index in range(last_index):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        for left in positions[layer_index]:
            for right in positions[layer_index + 1]:
                x_values.extend([float(layer_index), float(layer_index + 1), None])
                y_values.extend([left, right, None])
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": "#b9c3cf", "width": 0.6},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for index, (dimension, ys) in enumerate(zip(dimensions, positions)):
        fig.add_trace(
            go.Scatter(
                x=[float(index)] * len(ys),
                y=ys,
                mode="markers",
                marker={"size": 10, "color": "#4c78a8"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if dimension > len(ys):
            fig.add_annotation(
                x=index,
                y=0.5,
                text=f"…<br>{dimension} neuronas",
                showarrow=False,
                font={"size": 11},
                bgcolor="rgba(255,255,255,0.82)",
            )

    for index, dimension in enumerate(dimensions):
        if index == 0:
            label = f"Entrada<br>{dimension} píxeles"
        elif index == latent_index:
            label = f"Latente<br>{dimension}D"
        elif index == last_index:
            label = f"Salida<br>{dimension} píxeles"
        elif index < latent_index:
            label = f"Encoder<br>{dimension}"
        else:
            label = f"Decoder<br>{dimension}"
        fig.add_annotation(x=index, y=0.02, text=label, showarrow=False, yanchor="top")

    fig.add_vrect(x0=0.4, x1=latent_index - 0.4, fillcolor="#dceaf7", opacity=0.2, line_width=0)
    fig.add_vrect(
        x0=latent_index - 0.3,
        x1=latent_index + 0.3,
        fillcolor="#f4d35e",
        opacity=0.24,
        line_width=0,
    )
    fig.add_vrect(
        x0=latent_index + 0.4,
        x1=last_index - 0.4,
        fillcolor="#e8f2df",
        opacity=0.22,
        line_width=0,
    )
    fig.update_layout(
        title=f"Arquitectura DAE seleccionada: {'–'.join(map(str, dimensions))}",
        template="plotly_white",
        height=580,
        showlegend=False,
        xaxis={"visible": False, "range": [-0.5, last_index + 0.5]},
        yaxis={"visible": False, "range": [-0.12, 1.0]},
        margin={"b": 90},
    )
    fig.write_html(path, include_plotlyjs="cdn")


def add_score_bars(
    fig: Any,
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
    color: str,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    profiles = [item["profile"] for item in rows]
    trace = go.Bar(
        x=[item["choice"] for item in rows],
        y=[profile["robustness_score"] for profile in profiles],
        marker_color=[
            "#2ca02c" if same_configuration(item, selected) else color
            for item in rows
        ],
        text=[f"{profile['robustness_score']:.2f}" for profile in profiles],
        textposition="outside",
        customdata=[
            [
                profile["worst_level_mean_error"],
                profile["clean_mean_error"],
                item["config"].noise_level,
                item["config"].learning_rate,
                item["config"].optimizer,
                "-".join(map(str, item["config"].encoder_layers)),
                item["epochs"],
                item["config"].batch_size,
            ]
            for item, profile in zip(rows, profiles)
        ],
        hovertemplate=(
            "%{x}<br>Error promedio ruidoso: %{y:.3f} px/patrón<br>"
            "Peor nivel (media): %{customdata[0]:.3f} px<br>"
            "Error limpio: %{customdata[1]:.3f} px<br>"
            "Ruido de entrenamiento: %{customdata[2]}<br>"
            "Learning rate: %{customdata[3]}<br>"
            "Optimizador: %{customdata[4]}<br>"
            "Encoder: %{customdata[5]}<br>"
            "Épocas: %{customdata[6]}<br>"
            "Batch size: %{customdata[7]}<extra></extra>"
        ),
    )
    if row is None or col is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=col)


def write_comparison_charts(
    experiments: list[dict[str, Any]],
    selected: dict[str, Any],
    output_dir: Path,
) -> None:
    architecture_rows = best_per_group(experiments, "architecture")
    fig_arch = go.Figure()
    add_score_bars(fig_arch, architecture_rows, selected, "#4c78a8")
    fig_arch.update_layout(
        title="Exploración de arquitecturas: mejor configuración observada por arquitectura",
        xaxis_title="Capas del encoder",
        yaxis_title="Error medio de reconstrucción en niveles ruidosos (px/patrón)",
        template="plotly_white",
        height=520,
    )
    fig_arch.write_html(output_dir / "exploracion_arquitecturas.html", include_plotlyjs="cdn")

    groups = (
        ("optimizer", "Optimizador", "#f58518"),
        ("learning_rate", "Learning rate", "#54a24b"),
        ("noise_level", "Ruido usado al entrenar", "#b279a2"),
        ("batch_size", "Batch size", "#eeca3b"),
    )
    fig_grid = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=tuple(title for _, title, _ in groups),
        horizontal_spacing=0.08,
    )
    for column, (key, _, color) in enumerate(groups, start=1):
        add_score_bars(
            fig_grid,
            best_per_group(experiments, key),
            selected,
            color,
            row=1,
            col=column,
        )
        fig_grid.update_xaxes(title_text=groups[column - 1][1], row=1, col=column)
        fig_grid.update_yaxes(
            title_text="Error medio (px/patrón)",
            row=1,
            col=column,
        )
    fig_grid.update_layout(
        title="Hiperparámetros: mejor combinación observada por grupo (verde = seleccionada)",
        template="plotly_white",
        height=560,
        showlegend=False,
    )
    fig_grid.write_html(output_dir / "optimizadores_hiperparametros.html", include_plotlyjs="cdn")


def write_lr_scheduler_charts(
    experiments: list[dict[str, Any]],
    selected: dict[str, Any],
    output_dir: Path,
) -> None:
    fig_scores = go.Figure()
    fig_scores.add_trace(
        go.Bar(
            x=[item["choice"] for item in experiments],
            y=[item["profile"]["robustness_score"] for item in experiments],
            marker_color=[
                "#2ca02c" if same_configuration(item, selected) else "#4c78a8"
                for item in experiments
            ],
            text=[f"{item['profile']['robustness_score']:.2f}" for item in experiments],
            textposition="outside",
            customdata=[
                [
                    item["profile"]["worst_level_mean_error"],
                    item["profile"]["clean_mean_error"],
                    item["profile"]["mean_loss"],
                    item["epochs"],
                    item["config"].threshold,
                    item["profile"]["clean_tolerance_pass_rate"],
                    item["profile"]["robustness_tolerance_pass_rate"],
                ]
                for item in experiments
            ],
            hovertemplate=(
                "Configuración: %{x}<br>Error promedio ruidoso: %{y:.3f} px/patrón<br>"
                "Peor nivel (media): %{customdata[0]:.3f} px<br>"
                "Error limpio: %{customdata[1]:.3f} px<br>"
                "BCE medio: %{customdata[2]:.5f}<br>"
                "Épocas: %{customdata[3]}<br>"
                "Umbral: %{customdata[4]:.2f}<br>"
                "Letras limpias dentro de tolerancia: %{customdata[5]:.1%}<br>"
                "Patrones ruidosos dentro de tolerancia: %{customdata[6]:.1%}<extra></extra>"
            ),
        )
    )
    fig_scores.update_layout(
        title="Barrido de scheduler y sus parámetros (verde = seleccionado)",
        xaxis_title="Scheduler y parámetro",
        yaxis_title="Error medio de reconstrucción en niveles ruidosos (px/patrón)",
        template="plotly_white",
        height=520,
    )
    fig_scores.write_html(output_dir / "lr_scheduler_comparison.html", include_plotlyjs="cdn")

    reference = experiments[0]["config"]
    epochs = np.arange(1, reference.max_epochs + 1)
    fig_rates = go.Figure()
    for experiment in experiments:
        config = experiment["config"]
        scheduler = config.build_scheduler()
        rates = [
            scheduler.get_lr(config.learning_rate, int(epoch), config.max_epochs)
            for epoch in epochs
        ]
        fig_rates.add_trace(
            go.Scatter(
                x=epochs,
                y=rates,
                mode="lines",
                name=experiment["choice"],
            )
        )
    fig_rates.update_layout(
        title="Curvas de learning rate para los algoritmos comparados",
        xaxis_title="Época",
        yaxis_title="Learning rate",
        yaxis_type="log",
        template="plotly_white",
        height=520,
    )
    fig_rates.write_html(output_dir / "lr_scheduler_curves.html", include_plotlyjs="cdn")


def write_lr_scheduler_csv(
    experiments: list[dict[str, Any]],
    selected: dict[str, Any],
    path: Path,
) -> None:
    fields = (
        "lr_scheduler",
        "learning_rate",
        "min_lr_ratio",
        "decay_rate",
        "batch_size",
        "threshold",
        "epochs",
        "robustness_score",
        "clean_tolerance_pass_rate",
        "robustness_tolerance_pass_rate",
        "worst_level_mean_error",
        "clean_mean_error",
        "mean_loss",
        "selected",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in experiments:
            config = item["config"]
            profile = item["profile"]
            writer.writerow(
                {
                    "lr_scheduler": config.lr_scheduler,
                    "learning_rate": config.learning_rate,
                    "min_lr_ratio": config.min_lr_ratio,
                    "decay_rate": config.decay_rate,
                    "batch_size": config.batch_size,
                    "threshold": config.threshold,
                    "epochs": item["epochs"],
                    "robustness_score": profile["robustness_score"],
                    "clean_tolerance_pass_rate": profile["clean_tolerance_pass_rate"],
                    "robustness_tolerance_pass_rate": profile[
                        "robustness_tolerance_pass_rate"
                    ],
                    "worst_level_mean_error": profile["worst_level_mean_error"],
                    "clean_mean_error": profile["clean_mean_error"],
                    "mean_loss": profile["mean_loss"],
                    "selected": same_configuration(item, selected),
                }
            )


def write_threshold_sweep(
    experiment: dict[str, Any],
    output_dir: Path,
) -> None:
    threshold_results = experiment["threshold_results"]
    selected_threshold = experiment["config"].threshold
    fields = (
        "threshold",
        "clean_patterns_within_tolerance",
        "clean_tolerance_pass_rate",
        "robustness_tolerance_pass_rate",
        "robustness_score",
        "selected",
    )
    with (output_dir / "threshold_comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in threshold_results:
            profile = item["profile"]
            clean_rate = profile["clean_tolerance_pass_rate"]
            writer.writerow(
                {
                    "threshold": item["threshold"],
                    "clean_patterns_within_tolerance": round(
                        clean_rate * profile["clean_pattern_count"]
                    ),
                    "clean_tolerance_pass_rate": clean_rate,
                    "robustness_tolerance_pass_rate": profile[
                        "robustness_tolerance_pass_rate"
                    ],
                    "robustness_score": profile["robustness_score"],
                    "selected": item["threshold"] == selected_threshold,
                }
            )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Letras limpias dentro de tolerancia", "Error en niveles ruidosos"),
    )
    thresholds = [item["threshold"] for item in threshold_results]
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=[
                item["profile"]["clean_tolerance_pass_rate"] * 100
                for item in threshold_results
            ],
            mode="lines+markers",
            name="Letras limpias aprobadas (%)",
            line={"color": "#4c78a8", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=[
                item["profile"]["robustness_tolerance_pass_rate"] * 100
                for item in threshold_results
            ],
            mode="lines+markers",
            name="Patrones ruidosos aprobados (%)",
            line={"color": "#f58518", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=[item["profile"]["robustness_score"] for item in threshold_results],
            mode="lines+markers",
            name="Error promedio ruidoso (px/patrón)",
            line={"color": "#54a24b", "width": 2},
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=selected_threshold,
        line_dash="dash",
        line_color="#e45756",
        annotation_text=f"seleccionado: {selected_threshold:g}",
        row=1,
        col=1,
    )
    fig.add_vline(
        x=selected_threshold,
        line_dash="dash",
        line_color="#e45756",
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Umbral de binarización", row=1, col=1)
    fig.update_xaxes(title_text="Umbral de binarización", row=1, col=2)
    fig.update_yaxes(title_text="Porcentaje", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Píxeles incorrectos por patrón", row=1, col=2)
    fig.update_layout(
        title="Barrido del umbral de binarización (tolerancia: hasta 1 píxel)",
        template="plotly_white",
        height=500,
    )
    fig.write_html(output_dir / "threshold_comparison.html", include_plotlyjs="cdn")


def write_experiment_csv(
    experiments: list[dict[str, Any]],
    selected: dict[str, Any],
    path: Path,
) -> None:
    fields = (
        "choice",
        "encoder_layers",
        "optimizer",
        "learning_rate",
        "lr_scheduler",
        "training_noise_level",
        "batch_size",
        "threshold",
        "epochs",
        "robustness_score",
        "clean_tolerance_pass_rate",
        "robustness_tolerance_pass_rate",
        "worst_level_mean_error",
        "clean_mean_error",
        "mean_loss",
        "selected",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in experiments:
            config = item["config"]
            profile = item["profile"]
            writer.writerow(
                {
                    "choice": (
                        f"{'-'.join(map(str, config.encoder_layers))} | "
                        f"{config.optimizer} | lr={config.learning_rate:g} | "
                        f"ruido={config.noise_level:g} | batch={config.batch_size}"
                    ),
                    "encoder_layers": "-".join(map(str, config.encoder_layers)),
                    "optimizer": config.optimizer,
                    "learning_rate": config.learning_rate,
                    "lr_scheduler": config.lr_scheduler,
                    "training_noise_level": config.noise_level,
                    "batch_size": config.batch_size,
                    "threshold": config.threshold,
                    "epochs": item["epochs"],
                    "robustness_score": profile["robustness_score"],
                    "clean_tolerance_pass_rate": profile["clean_tolerance_pass_rate"],
                    "robustness_tolerance_pass_rate": profile[
                        "robustness_tolerance_pass_rate"
                    ],
                    "worst_level_mean_error": profile["worst_level_mean_error"],
                    "clean_mean_error": profile["clean_mean_error"],
                    "mean_loss": profile["mean_loss"],
                    "selected": same_configuration(item, selected),
                }
            )


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value).__name__}")


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_winning_config(
    config: AutoencoderConfig,
    model_path: Path,
    trained_epochs: int,
    selected_score: float,
    profile: dict[str, Any],
    path: Path,
) -> None:
    sections = {
        "data": {"dataset_path": project_path(DATASET)},
        "model": {
            "model_type": config.model_type,
            "encoder_layers": config.encoder_layers,
            "decoder_layers": list(reversed(config.encoder_layers[:-1])),
            "hidden_activation": config.hidden_activation,
            "latent_activation": config.latent_activation,
            "output_activation": config.output_activation,
            "beta": config.beta,
        },
        "noise": {
            "noise_type": config.noise_type,
            "noise_level": config.noise_level,
            "eval_noise_levels": config.eval_noise_levels,
        },
        "training": {
            "loss": config.loss,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "lr_scheduler": config.lr_scheduler,
            "min_lr_ratio": config.min_lr_ratio,
            "decay_rate": config.decay_rate,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "threshold": config.threshold,
            "seed": config.seed,
        },
        "persistence": {
            "save_model_path": project_path(model_path),
            "output_dir": project_path(config.output_dir),
        },
        "study": {
            "trained_epochs": trained_epochs,
            "evaluation_repeats": EVALUATION_REPEATS,
            "selection_noise_levels": list(SELECTION_NOISE_LEVELS),
            "tolerance_max_pixel_errors": MAX_PIXEL_ERRORS_WITHIN_TOLERANCE,
            "clean_patterns_within_tolerance": round(
                profile["clean_tolerance_pass_rate"] * profile["clean_pattern_count"]
            ),
            "clean_tolerance_pass_rate": profile["clean_tolerance_pass_rate"],
            "robustness_tolerance_pass_rate": profile[
                "robustness_tolerance_pass_rate"
            ],
            "selected_robustness_score": selected_score,
            "final_robustness_score": profile["robustness_score"],
        },
    }
    lines = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in values.items())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_noise_profile(
    profile: dict[str, Any],
    selected_config: AutoencoderConfig,
    output_dir: Path,
) -> None:
    levels = sorted(profile["by_level"])
    corrupted = [profile["by_level"][level]["corrupted_mean"] for level in levels]
    reconstructed = [profile["by_level"][level]["reconstruction_mean"] for level in levels]
    deviation = [profile["by_level"][level]["reconstruction_std"] for level in levels]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=levels,
            y=corrupted,
            mode="lines+markers",
            name="Error de entrada por ruido",
            line={"color": "#f58518", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=levels,
            y=reconstructed,
            mode="lines+markers",
            name="Error después de reconstruir",
            error_y={"type": "data", "array": deviation, "visible": True},
            line={"color": "#4c78a8", "width": 2},
        )
    )
    fig.update_layout(
        title=(
            "Eliminación de ruido: "
            f"entrenamiento con bit-flip p={selected_config.noise_level:g}; "
            f"evaluación sobre {EVALUATION_REPEATS} corrupciones por patrón y nivel"
        ),
        xaxis_title="Probabilidad de bit-flip",
        yaxis_title="Píxeles incorrectos por patrón (media)",
        template="plotly_white",
        height=520,
    )
    fig.write_html(output_dir / "capacidad_denoising.html", include_plotlyjs="cdn")

    with (output_dir / "perfil_ruido.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "noise_level",
                "input_corrupted_mean",
                "reconstruction_error_mean",
                "reconstruction_error_std",
                "reconstruction_max_error",
                "patterns_within_tolerance",
                "tolerance_pass_rate",
            ),
        )
        writer.writeheader()
        for level in levels:
            metrics = profile["by_level"][level]
            writer.writerow(
                {
                    "noise_level": level,
                    "input_corrupted_mean": metrics["corrupted_mean"],
                    "reconstruction_error_mean": metrics["reconstruction_mean"],
                    "reconstruction_error_std": metrics["reconstruction_std"],
                    "reconstruction_max_error": metrics["reconstruction_max"],
                    "patterns_within_tolerance": metrics[
                        "patterns_within_tolerance"
                    ],
                    "tolerance_pass_rate": metrics["tolerance_pass_rate"],
                }
            )


def write_noise_character_grid(
    model: Any,
    dataset: DatasetBundle,
    evaluation_sets: dict[float, tuple[np.ndarray, np.ndarray]],
    threshold: float,
    output_dir: Path,
) -> Path:
    representative_indices = (1, 6, 11, 16, 21, 26)
    character_indices = np.asarray(
        [index for index in representative_indices if index < len(dataset.labels)]
        or np.linspace(
            0,
            len(dataset.labels) - 1,
            num=min(6, len(dataset.labels)),
            dtype=int,
        ),
        dtype=int,
    )
    characters = [str(dataset.labels[index]) for index in character_indices]
    levels = sorted(evaluation_sets)
    subplot_titles = [
        f"<b>p={level:g}</b>" if row_index == 0 else ""
        for row_index in range(3)
        for level in levels
    ]
    fig = make_subplots(
        rows=3,
        cols=len(levels),
        subplot_titles=subplot_titles,
        row_titles=("Original", "Ruidoso", "Reconstruido"),
        horizontal_spacing=0.035,
        vertical_spacing=0.07,
    )
    height, width = dataset.image_shape
    colorscale = [[0.0, "#f5f5f5"], [1.0, "#1b263b"]]
    traces_per_character = len(levels) * 3

    for character, data_index in zip(characters, character_indices):
        for col_index, level in enumerate(levels, start=1):
            noisy_inputs, clean_targets = evaluation_sets[level]
            original = dataset.inputs[data_index]
            noisy = noisy_inputs[data_index]
            reconstructed = model.evaluate_reconstruction(
                noisy[np.newaxis, :],
                X_target=clean_targets[data_index][np.newaxis, :],
                threshold=threshold,
            )["reconstructed_bin"][0]
            for row_index, image in enumerate((original, noisy, reconstructed), start=1):
                fig.add_trace(
                    go.Heatmap(
                        z=np.asarray(image).reshape(height, width),
                        zmin=0,
                        zmax=1,
                        colorscale=colorscale,
                        showscale=False,
                        hovertemplate=(
                            f"Ruido: {level:g}<br>Carácter: {escape(character)}"
                            "<br>Fila: %{y}<br>Columna: %{x}<br>Pixel: %{z}<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=col_index,
                )
                axis_index = (row_index - 1) * len(levels) + col_index
                x_axis = "x" if axis_index == 1 else f"x{axis_index}"
                fig.update_xaxes(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    row=row_index,
                    col=col_index,
                )
                fig.update_yaxes(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    autorange="reversed",
                    scaleanchor=x_axis,
                    scaleratio=1,
                    row=row_index,
                    col=col_index,
                )

    buttons = []
    for selected_index, character in enumerate(characters):
        visible = [
            character_index == selected_index
            for character_index in range(len(characters))
            for _ in range(traces_per_character)
        ]
        buttons.append(
            {
                "label": character,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": {"text": (
                        "Efecto visual del ruido por nivel de bit-flip"
                        f" — carácter <b>{escape(character)}</b>"
                    )}},
                ],
            }
        )

    fig.update_layout(
        title=(
            "Efecto visual del ruido por nivel de bit-flip"
            f" — carácter <b>{escape(characters[0])}</b>"
        ),
        template="plotly_white",
        height=650,
        width=1400,
        margin={"l": 130, "r": 30, "t": 120, "b": 30},
        updatemenus=[
            {
                "type": "dropdown",
                "active": 0,
                "x": 0.02,
                "y": 1.16,
                "buttons": buttons,
            }
        ],
    )
    output_path = output_dir / "efecto_ruido_caracteres.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudio reproducible del denoising autoencoder para 1b.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directorio donde se guardan los gráficos y resultados.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = make_config(
        ARCHITECTURES[0],
        optimizer="adam",
        learning_rate=LEARNING_RATES[0],
        noise_level=TRAINING_NOISE_LEVELS[0],
        batch_size=BATCH_SIZES[0],
        max_epochs=EXPERIMENT_EPOCHS,
    )
    dataset = dataset_config.build_dataset()
    evaluation_sets = make_evaluation_sets(dataset_config, dataset)

    experiments = compare_configurations(dataset, evaluation_sets)
    selected = min(experiments, key=experiment_rank)
    selected_config = selected["config"]
    print(
        "Mejor combinación según letras dentro de tolerancia y robustez: "
        f"encoder={selected_config.encoder_layers}, optimizer={selected_config.optimizer}, "
        f"lr={selected_config.learning_rate:g}, ruido_entrenamiento={selected_config.noise_level:g}; "
        f"batch_size={selected_config.batch_size}, umbral={selected_config.threshold:g}; "
        f"letras limpias dentro de tolerancia="
        f"{round(selected['profile']['clean_tolerance_pass_rate'] * selected['profile']['clean_pattern_count'])}/"
        f"{selected['profile']['clean_pattern_count']}; "
        f"error medio ruidoso={selected['profile']['robustness_score']:.3f} px/patrón."
    )
    write_comparison_charts(experiments, selected, args.output_dir)
    write_experiment_csv(experiments, selected, args.output_dir / "comparaciones.csv")

    scheduler_experiments = compare_lr_schedulers(selected_config, dataset, evaluation_sets)
    selected_scheduler = min(scheduler_experiments, key=experiment_rank)
    selected_scheduler_config = selected_scheduler["config"]
    print(
        "Mejor scheduler y parámetros con la configuración restante fija: "
        f"{selected_scheduler['choice']}, umbral={selected_scheduler_config.threshold:g}; "
        f"letras limpias dentro de tolerancia="
        f"{round(selected_scheduler['profile']['clean_tolerance_pass_rate'] * selected_scheduler['profile']['clean_pattern_count'])}/"
        f"{selected_scheduler['profile']['clean_pattern_count']}; "
        f"error medio={selected_scheduler['profile']['robustness_score']:.3f} px/patrón."
    )
    write_lr_scheduler_charts(scheduler_experiments, selected_scheduler, args.output_dir)
    write_lr_scheduler_csv(
        scheduler_experiments,
        selected_scheduler,
        args.output_dir / "lr_schedulers.csv",
    )

    final_config = make_config(
        selected_scheduler_config.encoder_layers,
        optimizer=selected_scheduler_config.optimizer,
        learning_rate=selected_scheduler_config.learning_rate,
        noise_level=selected_scheduler_config.noise_level,
        batch_size=selected_scheduler_config.batch_size,
        max_epochs=FINAL_EPOCHS,
        lr_scheduler=selected_scheduler_config.lr_scheduler,
        min_lr_ratio=selected_scheduler_config.min_lr_ratio,
        decay_rate=selected_scheduler_config.decay_rate,
        threshold=selected_scheduler_config.threshold,
    )
    final_result = tune_threshold(
        train_experiment(final_config, dataset, evaluation_sets),
        evaluation_sets,
    )
    if final_result["profile"]["robustness_score"] > selected_scheduler["profile"]["robustness_score"]:
        print(
            "Aviso: el reentrenamiento final obtuvo un error medio mayor que la prueba "
            "seleccionada; los gráficos muestran las métricas finales medidas."
        )

    final_profile = final_result["profile"]
    write_threshold_sweep(final_result, args.output_dir)
    model_path = args.output_dir / "modelo_ganador.pkl"
    config_path = args.output_dir / "configuracion_ganadora.toml"
    final_config.output_dir = args.output_dir
    final_config.save_model_path = model_path
    final_result["model"].save(
        model_path,
        metadata={
            "model_type": final_config.model_type,
            "dataset_path": str(DATASET),
            "encoder_layers": final_config.encoder_layers,
            "decoder_layers": list(reversed(final_config.encoder_layers[:-1])),
            "labels": dataset.labels,
            "noise_type": final_config.noise_type,
            "noise_level": final_config.noise_level,
            "batch_size": final_config.batch_size,
            "learning_rate": final_config.learning_rate,
            "optimizer": final_config.optimizer,
            "lr_scheduler": final_config.lr_scheduler,
            "min_lr_ratio": final_config.min_lr_ratio,
            "decay_rate": final_config.decay_rate,
            "threshold": final_config.threshold,
            "robustness_score": final_profile["robustness_score"],
            "clean_tolerance_pass_rate": final_profile[
                "clean_tolerance_pass_rate"
            ],
            "noise_profile": [
                {"noise_level": level, **metrics}
                for level, metrics in final_profile["by_level"].items()
            ],
        },
    )
    write_winning_config(
        final_config,
        model_path,
        final_result["epochs"],
        selected_scheduler["profile"]["robustness_score"],
        final_profile,
        config_path,
    )

    reporter = TrainingReporter(args.output_dir, model_name="ej1b")
    clean_summary = final_result["model"].evaluate_reconstruction(
        dataset.inputs,
        threshold=final_config.threshold,
    )
    reports = reporter.write_reports(
        loss_history=final_result["history"],
        eval_summary=clean_summary,
        dataset_inputs=dataset.inputs,
        labels=dataset.labels,
        image_shape=dataset.image_shape,
        model=final_result["model"],
    )
    write_noise_profile(final_profile, final_config, args.output_dir)
    write_noise_character_grid(
        final_result["model"],
        dataset,
        evaluation_sets,
        final_config.threshold,
        args.output_dir,
    )
    write_architecture_diagram(
        final_config.encoder_layers,
        args.output_dir / "arquitectura_propuesta.html",
    )

    print(
        f"Resultado final: error medio en ruido={final_profile['robustness_score']:.3f} px/patrón; "
        f"error limpio={final_profile['clean_mean_error']:.3f} px/patrón; "
        f"letras limpias dentro de tolerancia="
        f"{round(final_profile['clean_tolerance_pass_rate'] * final_profile['clean_pattern_count'])}/"
        f"{final_profile['clean_pattern_count']}; "
        f"umbral={final_config.threshold:g}; "
        f"máximo limpio={clean_summary['max_pixel_error']} px; "
        f"épocas={final_result['epochs']}."
    )
    print("Gráficos y resultados:")
    for path in sorted(args.output_dir.glob("*.html")):
        print(f"  - {path}")
    for name, path in reports.items():
        print(f"  - {name}: {path}")
    print(f"  - Comparaciones CSV: {args.output_dir / 'comparaciones.csv'}")
    print(f"  - Comparación de schedulers CSV: {args.output_dir / 'lr_schedulers.csv'}")
    print(f"  - Comparación de umbrales CSV: {args.output_dir / 'threshold_comparison.csv'}")
    print(f"  - Perfil por nivel de ruido CSV: {args.output_dir / 'perfil_ruido.csv'}")
    print(f"  - Configuración ganadora: {config_path}")
    print(f"  - Modelo ganador: {model_path}")


if __name__ == "__main__":
    main()
