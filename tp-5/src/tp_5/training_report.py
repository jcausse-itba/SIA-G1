from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .engine.models.autoencoder import NOISE_REGISTRY, Autoencoder, VariationalAutoencoder


def _tile_characters(
    flat_images: NDArray[np.float64],
    image_shape: tuple[int, int],
    cols: int,
    gap: int = 1,
) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    """Tiles flat binary images into a 2D canvas separated by NaN borders."""
    h, w = image_shape
    n = len(flat_images)
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (h + gap) - gap, cols * (w + gap) - gap), np.nan)
    origins: list[tuple[int, int]] = []
    for idx, img in enumerate(flat_images):
        r, c = divmod(idx, cols)
        y0, x0 = r * (h + gap), c * (w + gap)
        canvas[y0 : y0 + h, x0 : x0 + w] = img.reshape(h, w)
        origins.append((x0, y0))
    return canvas, origins


def write_training_report(
    loss_history: Sequence[float],
    eval_summary: dict[str, Any],
    dataset_inputs: NDArray[np.float64],
    labels: list[str],
    image_shape: tuple[int, int],
    model: Autoencoder | VariationalAutoencoder,
    output_dir: str | Path,
    exercise: str = "1a",
) -> dict[str, Path]:
    """Generates all interactive HTML charts for TP5 Autoencoder exercises."""
    out = Path(output_dir) / f"ex_{exercise}"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    pixel_errors = eval_summary["pixel_errors"]
    latent_coords = eval_summary["latent_coords"]
    reconstructed_bin = eval_summary["reconstructed_bin"]

    # -------------------------------------------------------------------------
    # 1. Training Loss & Per-Character Pixel Error (1.a.2)
    # -------------------------------------------------------------------------
    fig_metrics = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Curva de Aprendizaje (Loss vs Época)",
            f"Error en Píxeles por Carácter (Máx = {eval_summary['max_pixel_error']} px | "
            f"<= 1 px: {eval_summary['patterns_le_1_error']}/{eval_summary['num_patterns']})",
        ),
        vertical_spacing=0.18,
    )
    fig_metrics.add_trace(
        go.Scatter(
            x=np.arange(1, len(loss_history) + 1),
            y=list(loss_history),
            mode="lines",
            name="Training Loss",
            line={"color": "#278276", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig_metrics.update_yaxes(title_text="Loss (log)", type="log", row=1, col=1)
    fig_metrics.update_xaxes(title_text="Época", row=1, col=1)

    bar_colors = ["#54a24b" if err <= 1 else "#bd4b45" for err in pixel_errors]
    fig_metrics.add_trace(
        go.Bar(
            x=labels,
            y=pixel_errors,
            marker_color=bar_colors,
            name="Píxeles incorrectos",
            text=pixel_errors,
            textposition="outside",
        ),
        row=2,
        col=1,
    )
    fig_metrics.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#bd4b45",
        annotation_text="Tolerancia máxima (1 píxel)",
        row=2,
        col=1,
    )
    fig_metrics.update_yaxes(title_text="Píxeles Incorrectos (de 35)", range=[0, max(3, int(pixel_errors.max()) + 1)], row=2, col=1)
    fig_metrics.update_xaxes(title_text="Carácter (font.h)", type="category", row=2, col=1)
    fig_metrics.update_layout(title_text=f"TP5 Ejercicio {exercise}: Desempeño de Entrenamiento", template="plotly_white", height=780)

    metrics_path = out / "training_and_pixel_errors.html"
    fig_metrics.write_html(metrics_path, include_plotlyjs="cdn")
    written["metrics"] = metrics_path

    # -------------------------------------------------------------------------
    # 2. Original vs Reconstructed 7x5 Bitmaps Grid (1.a.2)
    # -------------------------------------------------------------------------
    cols = 8
    orig_canvas, orig_origins = _tile_characters(dataset_inputs, image_shape, cols=cols, gap=2)
    recon_canvas, _ = _tile_characters(reconstructed_bin, image_shape, cols=cols, gap=2)

    fig_recon = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Patrones Originales (font.h)", "Patrones Reconstruidos por el Autoencoder"),
        horizontal_spacing=0.06,
    )
    colorscale = [[0.0, "#f5f5f5"], [1.0, "#1b263b"]]
    fig_recon.add_trace(go.Heatmap(z=orig_canvas, colorscale=colorscale, showscale=False, hoverinfo="skip"), row=1, col=1)
    fig_recon.add_trace(go.Heatmap(z=recon_canvas, colorscale=colorscale, showscale=False, hoverinfo="skip"), row=1, col=2)

    h, w = image_shape
    for idx, (x0, y0) in enumerate(orig_origins):
        err = pixel_errors[idx]
        lbl_color = "#278276" if err == 0 else ("#f58518" if err == 1 else "#bd4b45")
        for col_idx in (1, 2):
            fig_recon.add_annotation(
                x=x0 + w / 2 - 0.5,
                y=y0 - 0.7,
                text=f"<b>{labels[idx]}</b> ({err}px)" if col_idx == 2 else f"<b>{labels[idx]}</b>",
                showarrow=False,
                font={"size": 10, "color": lbl_color if col_idx == 2 else "#333"},
                row=1,
                col=col_idx,
            )

    fig_recon.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_recon.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed", scaleanchor="x", scaleratio=1)
    fig_recon.update_layout(title_text=f"Comparación Visual de los {len(labels)} Caracteres (7x5)", template="plotly_white", height=620)

    recon_path = out / "reconstructions.html"
    fig_recon.write_html(recon_path, include_plotlyjs="cdn")
    written["reconstructions"] = recon_path

    # -------------------------------------------------------------------------
    # 3. 2D Latent Space Scatter Plot (Requirement 1.a.3)
    # -------------------------------------------------------------------------
    if latent_coords.shape[1] >= 2:
        fig_latent = go.Figure()
        fig_latent.add_trace(
            go.Scatter(
                x=latent_coords[:, 0],
                y=latent_coords[:, 1],
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont={"size": 14, "color": "#1b263b"},
                marker={
                    "size": 12,
                    "color": pixel_errors,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Error (px)"},
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate="Carácter: <b>%{text}</b><br>z1: %{x:.3f}<br>z2: %{y:.3f}<extra></extra>",
            )
        )
        fig_latent.update_layout(
            title_text="Representación de los Caracteres en el Espacio Latente 2D (z1, z2)",
            xaxis_title="Dimensión Latente z1",
            yaxis_title="Dimensión Latente z2",
            template="plotly_white",
            height=650,
        )
        latent_path = out / "latent_space_2d.html"
        fig_latent.write_html(latent_path, include_plotlyjs="cdn")
        written["latent_space"] = latent_path

        # ---------------------------------------------------------------------
        # 4. New Character Generation & Latent Interpolation (Requirement 1.a.4 & 2.c)
        # ---------------------------------------------------------------------
        idx_a, idx_b = 1, min(2, len(labels) - 1)  # e.g., 'a' and 'b'
        z_a, z_b = latent_coords[idx_a], latent_coords[idx_b]
        alphas = np.linspace(0.0, 1.0, 7)
        interp_z = np.array([(1.0 - a) * z_a + a * z_b for a in alphas], dtype=np.float64)
        # Also include centroid of top-4 nearest neighbors to center for a novel hybrid character
        centroid_z = np.mean(latent_coords, axis=0, keepdims=True)
        all_gen_z = np.vstack([interp_z, centroid_z])

        gen_probs = model.decode(all_gen_z)
        gen_bin = (gen_probs >= 0.5).astype(np.float64)

        gen_canvas_prob, gen_origins = _tile_characters(gen_probs, image_shape, cols=len(all_gen_z), gap=2)
        gen_canvas_bin, _ = _tile_characters(gen_bin, image_shape, cols=len(all_gen_z), gap=2)

        fig_gen = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"Interpolación Continua en Espacio Latente ('{labels[idx_a]}' → '{labels[idx_b]}') + Centroide (Probabilidades)",
                "Nuevos Caracteres Generados (Binarizados a 7x5)",
            ),
            vertical_spacing=0.15,
        )
        fig_gen.add_trace(go.Heatmap(z=gen_canvas_prob, colorscale="Blues", showscale=False), row=1, col=1)
        fig_gen.add_trace(go.Heatmap(z=gen_canvas_bin, colorscale=colorscale, showscale=False), row=2, col=1)

        step_labels = [f"α={a:.2f}" for a in alphas] + ["Centroide"]
        for idx, (x0, y0) in enumerate(gen_origins):
            fig_gen.add_annotation(
                x=x0 + w / 2 - 0.5,
                y=y0 - 0.7,
                text=step_labels[idx],
                showarrow=False,
                font={"size": 11},
                row=1,
                col=1,
            )

        fig_gen.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_gen.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
        fig_gen.update_layout(
            title_text="Generación de Nuevas Letras desde el Espacio Latente 2D",
            template="plotly_white",
            height=520,
        )
        gen_path = out / "generated_letters.html"
        fig_gen.write_html(gen_path, include_plotlyjs="cdn")
        written["generated_letters"] = gen_path

    return written


def write_denoising_study_report(
    model: Autoencoder,
    clean_inputs: NDArray[np.float64],
    labels: list[str],
    image_shape: tuple[int, int],
    noise_type: str,
    noise_levels: list[float],
    output_dir: str | Path,
) -> Path:
    """Evaluates and plots Denoising Autoencoder performance across multiple distortion levels (Ex 1.b.2)."""
    out = Path(output_dir) / "ex_1b"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    noise_cls = NOISE_REGISTRY[noise_type.lower()]
    mean_Input_diff, mean_recon_err = [], []

    for lvl in noise_levels:
        corrupter = noise_cls(level=lvl) if noise_type != "none" else noise_cls()
        noisy_X = corrupter.corrupt(clean_inputs, rng)
        input_corrupted_px = float(np.mean(np.sum(np.abs((noisy_X >= 0.5).astype(float) - clean_inputs), axis=1)))
        res = model.evaluate_reconstruction(noisy_X, X_target=clean_inputs)
        mean_Input_diff.append(input_corrupted_px)
        mean_recon_err.append(res["mean_pixel_error"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"Nivel {lvl:g}" for lvl in noise_levels],
            y=mean_Input_diff,
            name="Píxeles distorsionados en la entrada (antes del DAE)",
            marker_color="#f58518",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[f"Nivel {lvl:g}" for lvl in noise_levels],
            y=mean_recon_err,
            name="Píxeles incorrectos tras reconstrucción (salida del DAE)",
            marker_color="#4c78a8",
        )
    )
    fig.update_layout(
        barmode="group",
        title_text=f"Capacidad de Eliminación de Ruido ({noise_type}) del Denoising Autoencoder",
        xaxis_title="Nivel de Distorsión",
        yaxis_title="Promedio de Píxeles Incorrectos (de 35)",
        template="plotly_white",
        height=520,
    )
    study_path = out / "denoising_study.html"
    fig.write_html(study_path, include_plotlyjs="cdn")
    return study_path
