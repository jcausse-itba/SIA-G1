from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tp_5.config.validator import AutoencoderConfig
from tp_5.training_report import TrainingReporter


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "study" / "ej1" / "resultados"
DATASET = ROOT / "doc" / "font.h"
ARCHITECTURE_CANDIDATES = (
    [35, 32, 16, 2],
    [35, 48, 24, 2],
    [35, 64, 32, 16, 2],
)
OPTIMIZER_CANDIDATES = ("sgd", "momentum", "adam")
LEARNING_RATE_CANDIDATES = (0.001, 0.004, 0.01)
EXPERIMENT_EPOCHS = 700
FINAL_EPOCHS = 1500
BATCH_SIZE = 8
SEED = 42


def make_config(
    architecture: list[int],
    *,
    optimizer: str,
    learning_rate: float,
    max_epochs: int = EXPERIMENT_EPOCHS,
) -> AutoencoderConfig:
    return AutoencoderConfig(
        dataset_path=DATASET,
        encoder_layers=architecture,
        hidden_activation="tanh",
        latent_activation="linear",
        output_activation="sigmoid",
        loss="binary_cross_entropy",
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        max_epochs=max_epochs,
        threshold=0.5,
        seed=SEED,
        output_dir=OUTPUT_DIR,
    )


def train_experiment(config: AutoencoderConfig) -> dict[str, Any]:
    dataset = config.build_dataset()
    model = config.build_model(quantization_scale=dataset.quantization_scale)

    def stop_when_learned(epoch: int, current_model: Any) -> bool:
        if epoch % 25 != 0:
            return False
        summary = current_model.evaluate_reconstruction(dataset.inputs, threshold=config.threshold)
        return summary["max_pixel_error"] <= 1

    history = model.fit(
        dataset.inputs,
        dataset.inputs,
        batch_size=config.batch_size,
        epochs=config.max_epochs,
        lr=config.learning_rate,
        print_every=config.max_epochs + 1,
        epoch_callback=stop_when_learned,
    )
    summary = model.evaluate_reconstruction(dataset.inputs, threshold=config.threshold)
    return {
        "config": config,
        "model": model,
        "dataset": dataset,
        "history": history,
        "summary": summary,
        "epochs": len(history),
    }


def compare_configurations() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_CANDIDATES:
        for optimizer in OPTIMIZER_CANDIDATES:
            for learning_rate in LEARNING_RATE_CANDIDATES:
                config = make_config(
                    architecture,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                )
                result = train_experiment(config)
                architecture_label = "-".join(map(str, architecture))
                experiments.append(
                    {
                        **result,
                        "choice": (
                            f"{architecture_label} | {optimizer} | lr={learning_rate:g}"
                        ),
                    }
                )
    return experiments


def experiment_rank(experiment: dict[str, Any]) -> tuple[float, ...]:
    config = experiment["config"]
    summary = experiment["summary"]
    return (
        float(summary["max_pixel_error"]),
        float(summary["mean_pixel_error"]),
        float(summary["loss"]),
        float(parameter_count(config.encoder_layers)),
        float(experiment["epochs"]),
    )


def parameter_count(encoder_layers: list[int]) -> int:
    layers = encoder_layers + list(reversed(encoder_layers[:-1]))
    return sum(source * target + target for source, target in zip(layers, layers[1:]))


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
        else:
            group = config.learning_rate
            label = f"{config.learning_rate:g}"

        current = best.get(group)
        if current is None or experiment_rank(experiment) < experiment_rank(current):
            best[group] = {**experiment, "choice": label}
    return list(best.values())


def write_architecture_diagram(encoder_layers: list[int], path: Path) -> None:
    dimensions = encoder_layers + list(reversed(encoder_layers[:-1]))
    latent_index = len(encoder_layers) - 1
    last_index = len(dimensions) - 1
    fig = go.Figure()
    positions: list[list[float]] = []
    for dimension in dimensions:
        count = min(dimension, 8)
        ys = np.linspace(0.12, 0.88, count).tolist()
        positions.append(ys)

    for layer_index in range(len(dimensions) - 1):
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for y_left in positions[layer_index]:
            for y_right in positions[layer_index + 1]:
                edge_x.extend([float(layer_index), float(layer_index + 1), None])
                edge_y.extend([y_left, y_right, None])
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "#b9c3cf", "width": 0.6},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for layer_index, (dimension, ys) in enumerate(zip(dimensions, positions)):
        x = float(layer_index)
        fig.add_trace(
            go.Scatter(
                x=[x] * len(ys),
                y=ys,
                mode="markers",
                marker={"size": 10, "color": "#4c78a8"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if dimension > len(ys):
            fig.add_annotation(
                x=x,
                y=0.5,
                text=f"…<br>{dimension} neuronas",
                showarrow=False,
                font={"size": 11, "color": "#444"},
                bgcolor="rgba(255,255,255,0.82)",
            )

    layer_labels = []
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
        layer_labels.append(label)

    for index, label in enumerate(layer_labels):
        fig.add_annotation(
            x=index,
            y=0.02,
            text=label,
            showarrow=False,
            yanchor="top",
            font={"size": 11},
        )

    fig.add_vrect(
        x0=0.4,
        x1=latent_index - 0.4,
        fillcolor="#dceaf7",
        opacity=0.2,
        line_width=0,
    )
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
        title=f"Arquitectura seleccionada: {'–'.join(map(str, dimensions))}",
        template="plotly_white",
        height=600,
        showlegend=False,
        xaxis={"visible": False, "range": [-0.5, last_index + 0.5]},
        yaxis={"visible": False, "range": [-0.12, 1.0]},
        margin={"b": 100},
    )
    fig.write_html(path, include_plotlyjs="cdn")


def write_comparison_charts(
    experiments: list[dict[str, Any]],
    selected: dict[str, Any],
    output_dir: Path,
) -> None:
    architecture_rows = best_per_group(experiments, "architecture")
    fig_arch = go.Figure()
    add_experiment_bars(fig_arch, architecture_rows, "#4c78a8", selected=selected)
    fig_arch.update_layout(
        title="Arquitecturas: mejor combinación observada por arquitectura",
        xaxis_title="Capas del encoder (entrada → espacio latente)",
        template="plotly_white",
        height=520,
    )
    fig_arch.update_yaxes(title_text="Máximo de píxeles incorrectos por patrón")
    fig_arch.add_hline(y=1, line_dash="dash", line_color="#bd4b45", annotation_text="Objetivo: ≤ 1 px")
    fig_arch.write_html(output_dir / "exploracion_arquitecturas.html", include_plotlyjs="cdn")

    optimizer_rows = best_per_group(experiments, "optimizer")
    rate_rows = best_per_group(experiments, "learning_rate")
    fig_opt = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Optimizador: mejor combinación observada",
            "Learning rate: mejor combinación observada",
        ),
        horizontal_spacing=0.16,
    )
    add_experiment_bars(fig_opt, optimizer_rows, "#f58518", selected=selected, row=1, col=1)
    add_experiment_bars(fig_opt, rate_rows, "#54a24b", selected=selected, row=1, col=2)
    fig_opt.update_layout(
        title="Optimizadores e hiperparámetros (barras verdes = configuración seleccionada)",
        template="plotly_white",
        height=520,
        showlegend=False,
    )
    fig_opt.update_yaxes(title_text="Máximo de píxeles incorrectos por patrón", row=1, col=1)
    fig_opt.update_yaxes(title_text="Máximo de píxeles incorrectos por patrón", row=1, col=2)
    fig_opt.add_hline(
        y=1,
        line_dash="dash",
        line_color="#bd4b45",
        annotation_text="Objetivo: ≤ 1 px",
        row=1,
        col=1,
    )
    fig_opt.add_hline(
        y=1,
        line_dash="dash",
        line_color="#bd4b45",
        annotation_text="Objetivo: ≤ 1 px",
        row=1,
        col=2,
    )
    fig_opt.write_html(output_dir / "optimizadores_hiperparametros.html", include_plotlyjs="cdn")


def add_experiment_bars(
    fig: Any,
    rows: list[dict[str, Any]],
    color: str,
    *,
    selected: dict[str, Any],
    row: int | None = None,
    col: int | None = None,
) -> None:
    summaries = [item["summary"] for item in rows]
    trace = go.Bar(
        x=[item["choice"] for item in rows],
        y=[summary["max_pixel_error"] for summary in summaries],
        marker_color=[
            "#2ca02c" if same_configuration(item, selected) else color
            for item in rows
        ],
        text=[f"{summary['max_pixel_error']} px" for summary in summaries],
        textposition="outside",
        customdata=[
            [
                summary["mean_pixel_error"],
                summary["loss"],
                summary["patterns_le_1_error"],
                item["epochs"],
                item["config"].learning_rate,
                item["config"].batch_size,
                "-".join(map(str, item["config"].encoder_layers)),
                item["config"].optimizer,
            ]
            for item, summary in zip(rows, summaries)
        ],
        hovertemplate=(
            "%{x}<br>Máximo de error: %{y} px<br>"
            "Error promedio: %{customdata[0]:.3f} px<br>"
            "BCE: %{customdata[1]:.5f}<br>"
            "Patrones con ≤1 px: %{customdata[2]}/32<br>"
            "Épocas: %{customdata[3]}<br>"
            "Learning rate: %{customdata[4]}<br>"
            "Batch size: %{customdata[5]}<br>"
            "Encoder: %{customdata[6]}<br>"
            "Optimizador: %{customdata[7]}<extra></extra>"
        ),
    )
    if row is None or col is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=col)


def same_configuration(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_config = first["config"]
    second_config = second["config"]
    return (
        first_config.encoder_layers == second_config.encoder_layers
        and first_config.optimizer == second_config.optimizer
        and first_config.learning_rate == second_config.learning_rate
    )


def write_experiment_csv(
    experiments: list[dict[str, Any]],
    selected: dict[str, Any],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "choice",
                "encoder_layers",
                "optimizer",
                "learning_rate",
                "batch_size",
                "epochs",
                "loss",
                "max_pixel_error",
                "mean_pixel_error",
                "patterns_le_1_error",
                "selected",
            ),
        )
        writer.writeheader()
        for item in experiments:
            summary = item["summary"]
            config = item["config"]
            writer.writerow(
                {
                    "choice": item["choice"],
                    "encoder_layers": "-".join(map(str, config.encoder_layers)),
                    "optimizer": config.optimizer,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "epochs": item["epochs"],
                    "loss": summary["loss"],
                    "max_pixel_error": summary["max_pixel_error"],
                    "mean_pixel_error": summary["mean_pixel_error"],
                    "patterns_le_1_error": summary["patterns_le_1_error"],
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
    summary: dict[str, Any],
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
        "training": {
            "loss": config.loss,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "lr_scheduler": config.lr_scheduler,
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
            "selected_max_pixel_error": summary["max_pixel_error"],
            "selected_mean_pixel_error": summary["mean_pixel_error"],
        },
    }
    lines = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in values.items())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_new_glyph(model: Any, inputs: np.ndarray, latent: np.ndarray, path: Path) -> None:
    lower = latent.min(axis=0)
    upper = latent.max(axis=0)
    span = np.maximum(upper - lower, 1.0)
    axes = [
        np.linspace(lower[dimension] - 0.15 * span[dimension], upper[dimension] + 0.15 * span[dimension], 41)
        for dimension in range(2)
    ]
    grid = np.array([(x, y) for x in axes[0] for y in axes[1]], dtype=np.float64)
    probabilities = model.decode(grid)
    binary = (probabilities >= 0.5).astype(np.float64)
    distances = np.array(
        [min(np.sum(image != known_image) for known_image in inputs) for image in binary],
        dtype=int,
    )
    ink_counts = binary.sum(axis=1)
    candidates = np.flatnonzero((distances >= 1) & (distances <= 6) & (ink_counts >= 5) & (ink_counts <= 22))
    if len(candidates) == 0:
        candidates = np.flatnonzero(distances >= 1)
    if len(candidates) == 0:
        raise RuntimeError("El decodificador no produjo ninguna muestra distinta del dataset.")

    candidate_scores = (
        np.abs(distances[candidates] - 2)
        + 0.08 * np.abs(ink_counts[candidates] - 12)
    )
    selected = candidates[int(np.argmin(candidate_scores))]
    probability_image = probabilities[selected].reshape(7, 5)
    binary_image = binary[selected].reshape(7, 5)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Probabilidades de píxel", "Glifo binarizado"),
    )
    fig.add_trace(
        go.Heatmap(z=probability_image, colorscale="Blues", zmin=0, zmax=1, showscale=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=binary_image,
            colorscale=[[0, "#f5f5f5"], [1, "#1b263b"]],
            zmin=0,
            zmax=1,
            showscale=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed")
    fig.update_layout(
        title=(
            "Glifo generado fuera del conjunto de entrenamiento "
            f"(z=({grid[selected, 0]:.2f}, {grid[selected, 1]:.2f}); "
            f"{int(ink_counts[selected])} píxeles activos; "
            f"distancia mínima al dataset: {distances[selected]} px)"
        ),
        template="plotly_white",
        height=420,
    )
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudio reproducible del autoencoder para la parte 1a.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directorio donde se guardan los gráficos y resultados.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Comparando arquitecturas, optimizadores y learning rates...")
    experiments = compare_configurations()
    selected = min(experiments, key=experiment_rank)
    selected_config = selected["config"]
    print(
        "Mejor combinación según error máximo, error promedio, BCE, "
        "cantidad de parámetros y épocas: "
        f"encoder={selected_config.encoder_layers}, optimizer={selected_config.optimizer}, "
        f"learning_rate={selected_config.learning_rate:g}, "
        f"máximo={selected['summary']['max_pixel_error']} px, "
        f"promedio={selected['summary']['mean_pixel_error']:.4f} px."
    )
    write_comparison_charts(experiments, selected, args.output_dir)
    write_experiment_csv(experiments, selected, args.output_dir / "comparaciones.csv")

    print("Reentrenando la combinación seleccionada para los informes finales...")
    final_config = make_config(
        selected_config.encoder_layers,
        optimizer=selected_config.optimizer,
        learning_rate=selected_config.learning_rate,
        max_epochs=FINAL_EPOCHS,
    )
    result = train_experiment(final_config)
    if result["summary"]["max_pixel_error"] > 1:
        raise RuntimeError(
            "El modelo final no alcanzó el objetivo de <=1 píxel de error por patrón "
            f"({result['summary']['max_pixel_error']} px)."
        )

    model_path = args.output_dir / "modelo_ganador.pkl"
    config_path = args.output_dir / "configuracion_ganadora.toml"
    final_config.output_dir = args.output_dir
    final_config.save_model_path = model_path
    result["model"].save(
        model_path,
        metadata={
            "model_type": final_config.model_type,
            "dataset_path": str(DATASET),
            "encoder_layers": final_config.encoder_layers,
            "decoder_layers": list(reversed(final_config.encoder_layers[:-1])),
            "labels": result["dataset"].labels,
            "max_pixel_error": result["summary"]["max_pixel_error"],
            "mean_pixel_error": result["summary"]["mean_pixel_error"],
        },
    )
    write_winning_config(
        final_config,
        model_path,
        result["epochs"],
        result["summary"],
        config_path,
    )

    reporter = TrainingReporter(args.output_dir, model_name="ej1")
    reports = reporter.write_reports(
        loss_history=result["history"],
        eval_summary=result["summary"],
        dataset_inputs=result["dataset"].inputs,
        labels=result["dataset"].labels,
        image_shape=result["dataset"].image_shape,
        model=result["model"],
    )
    latent = result["summary"]["latent_coords"]
    write_architecture_diagram(
        final_config.encoder_layers,
        args.output_dir / "arquitectura_propuesta.html",
    )
    write_new_glyph(
        result["model"],
        result["dataset"].inputs,
        latent,
        args.output_dir / "glifo_generado.html",
    )

    print(
        f"Resultado final: {result['summary']['patterns_le_1_error']}/"
        f"{result['summary']['num_patterns']} patrones con <=1 px de error; "
        f"máximo={result['summary']['max_pixel_error']} px; "
        f"épocas={result['epochs']}."
    )
    print("Gráficos y datos:")
    for path in sorted(args.output_dir.glob("*.html")):
        print(f"  - {path}")
    for name, path in reports.items():
        print(f"  - {name}: {path}")
    print(f"  - Comparaciones CSV: {args.output_dir / 'comparaciones.csv'}")
    print(f"  - Configuración ganadora: {config_path}")
    print(f"  - Modelo ganador: {model_path}")


if __name__ == "__main__":
    main()
