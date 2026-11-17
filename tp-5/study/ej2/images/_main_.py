from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageFile
from plotly.subplots import make_subplots

from tp_5.config.validator import DatasetBundle
from tp_5.engine.activation_functions.identity import Identity
from tp_5.engine.activation_functions.tanh import Tanh
from tp_5.engine.loss_functions.elbo import ELBOLoss
from tp_5.engine.loss_functions.kl_divergence import KLDivergence
from tp_5.engine.loss_functions.mse import MeanSquaredError
from tp_5.engine.models.autoencoder import Autoencoder
from tp_5.engine.models.mlp import MLP
from tp_5.engine.models.variational_autoencoder import VariationalAutoencoder
from tp_5.engine.optimizers.adam import Adam
from tp_5.engine.schedulers.constant import Constant


ROOT = Path(__file__).resolve().parents[3]
BMP_DIR = ROOT / "code_dataset" / "images"
OUTPUT_DIR = Path(__file__).resolve().parent

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 18
CHANNELS = 3
INPUT_DIM = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS
LATENT_DIM = 2
ENCODER_LAYERS = (INPUT_DIM, 128, 32, LATENT_DIM)
DECODER_LAYERS = (32, 128, INPUT_DIM)

LEARNING_RATE = 0.001
KL_WEIGHT = 0.00001
HIDDEN_ACTIVATION = "tanh"
LATENT_ACTIVATION = "linear"
OUTPUT_ACTIVATION = "linear"
OPTIMIZER = "adam"
RECONSTRUCTION_LOSS = "mean_squared_error"
BATCH_SIZE = 8
MAX_EPOCHS = 2500
SEED = 42

DECODER_GRID_SIDE = 9
PRIOR_CANDIDATES = 64
DISPLAYED_GENERATIONS = 16
INTERPOLATION_STEPS = 9

CATEGORY_COLORS = {
    "draws": "#4c78a8",
    "fractal-of-fractals": "#54a24b",
    "hybrid": "#f58518",
}
COLORSCALE = [[0.0, "#f5f5f5"], [1.0, "#1b263b"]]


def category_from_filename(filename: str) -> str:
    stem = re.sub(r"^\d+-", "", Path(filename).stem)
    if stem.startswith("fractal-of-fractals-"):
        return "fractal-of-fractals"
    return stem.split("-", maxsplit=1)[0]


def read_bmp(path: Path) -> tuple[np.ndarray, bool]:
    def decode() -> np.ndarray:
        with Image.open(path) as image:
            image.load()
            resized = image.convert("RGB").resize(
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                Image.Resampling.LANCZOS,
            )
            return np.asarray(resized, dtype=np.float64) / 255.0

    try:
        return decode(), False
    except OSError as error:
        if "truncated" not in str(error).casefold():
            raise RuntimeError(f"No se pudo leer el BMP {path}.") from error

    previous_truncated_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return decode(), True
    except OSError as error:
        raise RuntimeError(f"No se pudo recuperar el BMP truncado {path}.") from error
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated_setting


def load_bmp_dataset() -> tuple[DatasetBundle, list[str], list[bool]]:
    files = sorted(BMP_DIR.glob("*.bmp"))
    if not files:
        raise FileNotFoundError(f"No hay imágenes BMP en {BMP_DIR}.")

    vectors = []
    labels = []
    categories = []
    recovered_truncated = []
    for path in files:
        image, was_truncated = read_bmp(path)
        vectors.append(image.reshape(-1))
        labels.append(path.name)
        categories.append(category_from_filename(path.name))
        recovered_truncated.append(was_truncated)

    dataset = DatasetBundle(
        inputs=np.vstack(vectors),
        labels=labels,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        quantization_scale=1.0,
    )
    return dataset, categories, recovered_truncated


def write_dataset_manifest(
    dataset: DatasetBundle,
    categories: list[str],
    recovered_truncated: list[bool],
) -> None:
    path = OUTPUT_DIR / "dataset_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("filename", "category", "truncated_bmp_recovered"),
        )
        writer.writeheader()
        for label, category, was_truncated in zip(
            dataset.labels,
            categories,
            recovered_truncated,
        ):
            writer.writerow(
                {
                    "filename": label,
                    "category": category,
                    "truncated_bmp_recovered": was_truncated,
                }
            )

    recovered = [label for label, flag in zip(dataset.labels, recovered_truncated) if flag]
    if recovered:
        print(
            f"[AVISO] Se recuperaron {len(recovered)} BMP truncados "
            "leyendo su contenido disponible: "
            + ", ".join(recovered)
        )


def build_autoencoder() -> Autoencoder:
    np.random.seed(SEED)
    return Autoencoder(
        encoder_layers=list(ENCODER_LAYERS),
        decoder_layers=list(DECODER_LAYERS),
        hidden_activation=Tanh(),
        latent_activation=Identity(),
        output_activation=Identity(),
        loss_function=MeanSquaredError(),
        optimizer=Adam,
        lr_scheduler=Constant(),
        quantization_scale=1.0,
    )


def build_vae() -> VariationalAutoencoder:
    np.random.seed(SEED)
    return VariationalAutoencoder(
        encoder_layers=list(ENCODER_LAYERS),
        decoder_layers=list(DECODER_LAYERS),
        hidden_activation=Tanh(),
        latent_activation=Identity(),
        output_activation=Identity(),
        loss_function=MeanSquaredError(),
        optimizer=Adam,
        kl_loss=KLDivergence(),
        kl_weight=KL_WEIGHT,
        lr_scheduler=Constant(),
        quantization_scale=1.0,
        seed=SEED,
    )


def train_autoencoder(
    model: Autoencoder,
    inputs: np.ndarray,
) -> list[float]:
    print(f"Entrenando AE: {MAX_EPOCHS} épocas, batch size {BATCH_SIZE}...")
    return model.fit(
        inputs,
        inputs,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
        print_every=500,
    )


def train_vae(
    model: VariationalAutoencoder,
    inputs: np.ndarray,
) -> tuple[list[float], dict[str, list[float]]]:
    components: dict[str, list[float]] = {
        "reconstruction": [],
        "kl": [],
        "weighted_kl": [],
        "evaluation_total": [],
    }

    def capture_components(_epoch: int, trained_model: MLP) -> None:
        if _epoch != len(components["reconstruction"]) + 1:
            raise RuntimeError("Las épocas del callback VAE no son consecutivas.")
        if not isinstance(trained_model, VariationalAutoencoder):
            raise TypeError("El callback de componentes requiere un VAE.")
        variational_layer = trained_model.variational_layer
        loss_fn = trained_model.loss_fn
        if not isinstance(loss_fn, ELBOLoss):
            raise TypeError("El VAE debe utilizar una pérdida ELBO.")
        was_training = variational_layer.training
        variational_layer.training = False
        try:
            prediction = trained_model.forward(inputs)
            reconstruction = loss_fn.reconstruction_loss.compute(prediction, inputs)
            kl = loss_fn.kl_loss.compute(variational_layer.mu, variational_layer.std)
        finally:
            variational_layer.training = was_training

        weighted_kl = KL_WEIGHT * kl
        components["reconstruction"].append(float(reconstruction))
        components["kl"].append(float(kl))
        components["weighted_kl"].append(float(weighted_kl))
        components["evaluation_total"].append(float(reconstruction + weighted_kl))

    print(f"Entrenando VAE: {MAX_EPOCHS} épocas, batch size {BATCH_SIZE}...")
    history = model.fit(
        inputs,
        inputs,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
        print_every=500,
        epoch_callback=capture_components,
    )
    if len(components["reconstruction"]) != len(history):
        raise RuntimeError("No se registraron componentes de pérdida para cada época.")
    return history, components


def model_parameters(model: Any) -> int:
    return sum(layer.W.size + layer.b.size for layer in model.layers)


def reconstruction_metrics(model: Any, inputs: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if isinstance(model, VariationalAutoencoder):
        model.variational_layer.training = False
    predictions = np.clip(model.forward(inputs), 0.0, 1.0)
    difference = predictions - inputs
    mse = float(np.mean(np.square(difference)))
    mae = float(np.mean(np.abs(difference)))
    psnr = float(10.0 * np.log10(1.0 / max(mse, np.finfo(float).tiny)))
    return predictions, {"mse": mse, "mae": mae, "psnr": psnr}


def to_rgb_image(vector: np.ndarray) -> np.ndarray:
    return np.rint(
        np.clip(vector.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), 0.0, 1.0) * 255
    ).astype(np.uint8)


def tile_images(
    images: list[np.ndarray],
    *,
    columns: int,
    highlight_index: int | None = None,
    gap: int = 3,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    rows = (len(images) + columns - 1) // columns
    canvas_height = rows * IMAGE_HEIGHT + (rows - 1) * gap
    canvas_width = columns * IMAGE_WIDTH + (columns - 1) * gap
    canvas = np.full((canvas_height, canvas_width, CHANNELS), 245, dtype=np.uint8)
    centers = []

    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        top = row * (IMAGE_HEIGHT + gap)
        left = column * (IMAGE_WIDTH + gap)
        tile = to_rgb_image(image).copy()
        if index == highlight_index:
            tile[0, :, :] = (255, 35, 35)
            tile[-1, :, :] = (255, 35, 35)
            tile[:, 0, :] = (255, 35, 35)
            tile[:, -1, :] = (255, 35, 35)
        canvas[top : top + IMAGE_HEIGHT, left : left + IMAGE_WIDTH] = tile
        centers.append((left + IMAGE_WIDTH / 2, top + IMAGE_HEIGHT / 2))
    return canvas, centers


def write_dataset_preview(dataset: DatasetBundle, categories: list[str]) -> Path:
    category_order = sorted(set(categories))
    columns = 4
    titles = []
    figures = []
    for category in category_order:
        indices = np.flatnonzero(np.asarray(categories) == category)
        selected = np.linspace(
            0,
            len(indices) - 1,
            num=min(columns, len(indices)),
            dtype=int,
        )
        row_indices = indices[selected]
        titles.extend(
            Path(dataset.labels[index]).stem[:26] for index in row_indices
        )
        titles.extend([""] * (columns - len(row_indices)))
        figures.extend(dataset.inputs[index] for index in row_indices)
        figures.extend(
            np.zeros(INPUT_DIM, dtype=np.float64)
            for _ in range(columns - len(row_indices))
        )

    fig = make_subplots(
        rows=len(category_order),
        cols=columns,
        subplot_titles=titles,
        row_titles=category_order,
        horizontal_spacing=0.025,
        vertical_spacing=0.12,
    )
    for index, image in enumerate(figures):
        row, column = divmod(index, columns)
        fig.add_trace(go.Image(z=to_rgb_image(image)), row=row + 1, col=column + 1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
    fig.update_layout(
        title="Muestras BMP del dataset, agrupadas por categoría",
        template="plotly_white",
        height=250 * len(category_order),
        width=1250,
        margin={"l": 190, "r": 30, "t": 80, "b": 30},
    )
    path = OUTPUT_DIR / "dataset_muestras.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_architecture_diagram() -> Path:
    fig = go.Figure()

    def node(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label: str,
        color: str,
        border: str = "#52606d",
    ) -> None:
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line={"color": border, "width": 2},
            fillcolor=color,
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=label,
            showarrow=False,
            font={"size": 13, "color": "#17202a"},
        )

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#52606d",
        )

    node(0.02, 0.42, 0.17, 0.60, "Imagen RGB<br>32 × 18 × 3", "#e8eef5")
    node(0.22, 0.42, 0.39, 0.60, "Encoder<br>1728 → 128 → 32", "#dceeff")
    node(0.45, 0.69, 0.59, 0.84, "μ<br>2 valores", "#d7f0e2")
    node(0.45, 0.34, 0.59, 0.49, "σ<br>2 valores", "#d7f0e2")
    node(0.64, 0.11, 0.79, 0.27, "ε ~ N(0, I)", "#e8eef5")
    node(
        0.63,
        0.49,
        0.80,
        0.68,
        "Reparametrización<br>z = μ + σ ⊙ ε",
        "#ffe0b2",
        "#ef6c00",
    )
    node(0.84, 0.42, 1.00, 0.60, "Decoder<br>2 → 32 → 128 → 1728", "#dceeff")
    node(1.04, 0.42, 1.18, 0.60, "Imagen RGB<br>reconstruida", "#e8eef5")

    arrow(0.17, 0.51, 0.22, 0.51)
    arrow(0.39, 0.53, 0.45, 0.76)
    arrow(0.39, 0.49, 0.45, 0.42)
    arrow(0.59, 0.76, 0.63, 0.63)
    arrow(0.59, 0.42, 0.63, 0.54)
    arrow(0.715, 0.27, 0.715, 0.49)
    arrow(0.80, 0.58, 0.84, 0.51)
    arrow(1.00, 0.51, 1.04, 0.51)

    fig.add_annotation(
        x=0.50,
        y=0.15,
        text="Regularización: KL(q(z|x) || N(0, I))",
        showarrow=False,
        font={"size": 12, "color": "#3949ab"},
    )
    fig.update_xaxes(visible=False, range=[0, 1.20], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_layout(
        title="Arquitectura VAE para imágenes BMP",
        template="plotly_white",
        height=560,
        width=1500,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    path = OUTPUT_DIR / "arquitectura_vae.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_vae_training_curves(
    history: list[float],
    components: dict[str, list[float]],
) -> Path:
    epochs = np.arange(1, len(history) + 1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=epochs, y=history, mode="lines", name="ELBO total de entrenamiento"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=components["evaluation_total"],
            mode="lines",
            name="Total: reconstrucción + β·KL (z=μ)",
            line={"dash": "dash"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=components["reconstruction"],
            mode="lines",
            name="Reconstrucción (dataset completo, z=μ)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=components["weighted_kl"],
            mode="lines",
            name="β·KL (contribución a ELBO)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=components["kl"],
            mode="lines",
            name="KL sin ponderar",
            line={"color": "#f58518"},
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Época")
    fig.update_yaxes(title_text="Pérdida / contribución ponderada", secondary_y=False)
    fig.update_yaxes(title_text="KL sin ponderar", secondary_y=True)
    fig.update_layout(
        title=f"Entrenamiento VAE: MSE + β·KL (β={KL_WEIGHT:g})",
        hovermode="x unified",
        template="plotly_white",
        height=650,
    )
    path = OUTPUT_DIR / "entrenamiento_vae.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_latent_spaces(
    dataset: DatasetBundle,
    categories: list[str],
    autoencoder: Autoencoder,
    vae: VariationalAutoencoder,
) -> Path:
    ae_latent = autoencoder.encode(dataset.inputs)
    vae_mu, _ = vae.encode_distribution(dataset.inputs)
    category_array = np.asarray(categories)
    ordered_categories = sorted(set(categories))
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("AE", "VAE (μ posterior)"),
        horizontal_spacing=0.12,
    )
    for category_index, category in enumerate(ordered_categories):
        selected = category_array == category
        color = CATEGORY_COLORS.get(category, f"hsl({category_index * 67},55%,45%)")
        for column, latent in ((1, ae_latent), (2, vae_mu)):
            fig.add_trace(
                go.Scatter(
                    x=latent[selected, 0],
                    y=latent[selected, 1],
                    mode="markers",
                    name=category,
                    legendgroup=category,
                    showlegend=column == 1,
                    text=np.asarray(dataset.labels)[selected],
                    marker={"color": color, "size": 8, "opacity": 0.8},
                    hovertemplate=(
                        f"Categoría: {category}<br>Archivo: %{{text}}"
                        "<br>z₁: %{x:.3f}<br>z₂: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    fig.update_xaxes(title_text="z₁", row=1, col=1)
    fig.update_xaxes(title_text="z₁", row=1, col=2)
    fig.update_yaxes(title_text="z₂", row=1, col=1)
    fig.update_yaxes(title_text="z₂", row=1, col=2)
    fig.update_layout(
        title="Espacios latentes: codificaciones del AE y μ del VAE",
        template="plotly_white",
        height=650,
        width=1250,
        legend_title_text="Categoría",
    )
    path = OUTPUT_DIR / "espacio_latente_ae_vae.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_decoder_grid(vae: VariationalAutoencoder) -> Path:
    coordinates = np.linspace(-2.0, 2.0, DECODER_GRID_SIDE)
    latent_grid = np.asarray(
        [[x, y] for y in coordinates[::-1] for x in coordinates],
        dtype=np.float64,
    )
    decoded = np.clip(vae.decode(latent_grid), 0.0, 1.0)
    canvas, centers = tile_images(
        list(decoded),
        columns=DECODER_GRID_SIDE,
        gap=2,
    )
    fig = go.Figure(go.Image(z=canvas))
    fig.update_xaxes(
        tickmode="array",
        tickvals=[centers[index][0] for index in range(DECODER_GRID_SIDE)],
        ticktext=[f"{value:.1f}" for value in coordinates],
        title_text="z₁",
        showgrid=False,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[centers[row * DECODER_GRID_SIDE][1] for row in range(DECODER_GRID_SIDE)],
        ticktext=[f"{value:.1f}" for value in coordinates[::-1]],
        title_text="z₂",
        autorange="reversed",
        showgrid=False,
    )
    fig.update_layout(
        title="Grilla del decoder VAE al recorrer z₁ y z₂",
        template="plotly_white",
        width=1050,
        height=750,
    )
    path = OUTPUT_DIR / "grilla_decoder_latente.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_random_generations(
    dataset: DatasetBundle,
    vae: VariationalAutoencoder,
) -> tuple[Path, dict[str, Any]]:
    rng = np.random.default_rng(SEED + 1)
    latent_samples = rng.standard_normal((PRIOR_CANDIDATES, LATENT_DIM))
    generated = np.clip(vae.decode(latent_samples), 0.0, 1.0)

    best_index = 0
    best_distance = float("inf")
    best_nearest_image = ""
    for index, sample in enumerate(generated):
        distances = np.mean(np.square(dataset.inputs - sample), axis=1)
        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] < best_distance:
            best_index = index
            best_distance = float(distances[nearest_index])
            best_nearest_image = dataset.labels[nearest_index]

    displayed_indices = [
        index for index in range(PRIOR_CANDIDATES) if index != best_index
    ][: DISPLAYED_GENERATIONS - 1]
    displayed_indices.append(best_index)
    canvas, _ = tile_images(
        [generated[index] for index in displayed_indices],
        columns=4,
        highlight_index=DISPLAYED_GENERATIONS - 1,
    )
    fig = go.Figure(go.Image(z=canvas))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
    fig.add_annotation(
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        text=(
            f"Resaltada: mejor coincidencia entre {PRIOR_CANDIDATES} muestras del prior; "
            f"más cercana a {best_nearest_image} (MSE={best_distance:.5f})."
        ),
        showarrow=False,
        font={"size": 12},
    )
    fig.update_layout(
        title="Muestras aleatorias de z ~ N(0, I); borde rojo = muestra elegida",
        template="plotly_white",
        width=900,
        height=650,
        margin={"l": 30, "r": 30, "t": 80, "b": 80},
    )
    path = OUTPUT_DIR / "muestras_aleatorias.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path, {
        "sample_index": best_index,
        "z": latent_samples[best_index].tolist(),
        "nearest_training_image": best_nearest_image,
        "nearest_training_mse": best_distance,
    }


def write_interpolation(
    dataset: DatasetBundle,
    categories: list[str],
    vae: VariationalAutoencoder,
) -> Path:
    source_indices = []
    for category in ("draws", "hybrid"):
        try:
            source_indices.append(categories.index(category))
        except ValueError as error:
            raise ValueError(f"No hay imágenes de la categoría {category!r}.") from error

    means, _ = vae.encode_distribution(dataset.inputs[source_indices])
    interpolation = np.linspace(means[0], means[1], INTERPOLATION_STEPS)
    decoded = np.clip(vae.decode(interpolation), 0.0, 1.0)
    canvas, centers = tile_images(list(decoded), columns=INTERPOLATION_STEPS, gap=2)
    fractions = np.linspace(0.0, 1.0, INTERPOLATION_STEPS)

    fig = go.Figure(go.Image(z=canvas))
    fig.update_xaxes(
        tickmode="array",
        tickvals=[center[0] for center in centers],
        ticktext=[f"{value:.2f}" for value in fractions],
        title_text="Fracción de interpolación",
        showgrid=False,
    )
    fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed")
    fig.update_layout(
        title=(
            "Interpolación entre "
            f"{dataset.labels[source_indices[0]]} y {dataset.labels[source_indices[1]]}"
        ),
        template="plotly_white",
        width=1250,
        height=350,
    )
    path = OUTPUT_DIR / "interpolacion_latente.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_reconstruction_comparison(
    dataset: DatasetBundle,
    ae_reconstructions: np.ndarray,
    vae_reconstructions: np.ndarray,
) -> Path:
    selected_indices = np.linspace(
        0,
        len(dataset.labels) - 1,
        num=min(6, len(dataset.labels)),
        dtype=int,
    )
    columns = len(selected_indices)
    titles = [
        Path(dataset.labels[index]).stem[:24] if row == 0 else ""
        for row in range(3)
        for index in selected_indices
    ]
    fig = make_subplots(
        rows=3,
        cols=columns,
        subplot_titles=titles,
        row_titles=("Original", "AE", "VAE"),
        horizontal_spacing=0.025,
        vertical_spacing=0.08,
    )
    for col, index in enumerate(selected_indices, start=1):
        for row, image in enumerate(
            (
                dataset.inputs[index],
                ae_reconstructions[index],
                vae_reconstructions[index],
            ),
            start=1,
        ):
            fig.add_trace(go.Image(z=to_rgb_image(image)), row=row, col=col)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
    fig.update_layout(
        title="Reconstrucciones sobre imágenes del dataset: AE frente a VAE",
        template="plotly_white",
        width=1450,
        height=650,
        margin={"l": 110, "r": 20, "t": 80, "b": 25},
    )
    path = OUTPUT_DIR / "reconstrucciones_ae_vae.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_comparison_table(
    ae_metrics: dict[str, float],
    vae_metrics: dict[str, float],
    ae_history: list[float],
    vae_history: list[float],
    ae: Autoencoder,
    vae: VariationalAutoencoder,
) -> Path:
    metric_names = (
        "Dimensión de entrada",
        "Dimensión latente",
        "MSE de reconstrucción",
        "MAE de reconstrucción",
        "PSNR de reconstrucción (dB)",
        "Última pérdida de entrenamiento",
        "Parámetros entrenables",
        "β·KL",
    )
    ae_values = (
        str(INPUT_DIM),
        str(LATENT_DIM),
        f"{ae_metrics['mse']:.6f}",
        f"{ae_metrics['mae']:.6f}",
        f"{ae_metrics['psnr']:.2f}",
        f"{ae_history[-1]:.6f}",
        f"{model_parameters(ae):,}",
        "—",
    )
    vae_values = (
        str(INPUT_DIM),
        str(LATENT_DIM),
        f"{vae_metrics['mse']:.6f}",
        f"{vae_metrics['mae']:.6f}",
        f"{vae_metrics['psnr']:.2f}",
        f"{vae_history[-1]:.6f}",
        f"{model_parameters(vae):,}",
        f"{KL_WEIGHT:g}",
    )
    fig = go.Figure(
        go.Table(
            header={
                "values": ["Métrica", "Autoencoder (AE)", "Autoencoder variacional (VAE)"],
                "fill_color": "#2a3990",
                "font": {"color": "white", "size": 14},
                "align": "left",
            },
            cells={
                "values": [metric_names, ae_values, vae_values],
                "fill_color": [
                    [
                        "#eef2f8" if index % 2 == 0 else "#ffffff"
                        for index in range(len(metric_names))
                    ]
                    for _ in range(3)
                ],
                "align": "left",
                "height": 34,
            },
        )
    )
    fig.update_layout(
        title="Comparación AE vs VAE (reconstrucción sobre el dataset de entrenamiento)",
        template="plotly_white",
        height=500,
    )
    path = OUTPUT_DIR / "comparacion_ae_vae.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_training_summary(
    dataset: DatasetBundle,
    categories: list[str],
    ae_metrics: dict[str, float],
    vae_metrics: dict[str, float],
    ae_history: list[float],
    vae_history: list[float],
    selected_generation: dict[str, Any],
) -> None:
    path = OUTPUT_DIR / "resumen.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("metric", "ae", "vae"),
        )
        writer.writeheader()
        rows = (
            ("dataset_images", len(dataset.labels), len(dataset.labels)),
            ("dataset_categories", len(set(categories)), len(set(categories))),
            ("training_epochs", len(ae_history), len(vae_history)),
            ("learning_rate", LEARNING_RATE, LEARNING_RATE),
            ("batch_size", BATCH_SIZE, BATCH_SIZE),
            ("mse", ae_metrics["mse"], vae_metrics["mse"]),
            ("mae", ae_metrics["mae"], vae_metrics["mae"]),
            ("psnr", ae_metrics["psnr"], vae_metrics["psnr"]),
            ("final_training_loss", ae_history[-1], vae_history[-1]),
            ("latent_dimension", LATENT_DIM, LATENT_DIM),
            ("kl_weight", 0.0, KL_WEIGHT),
            ("selected_sample_index", "", selected_generation["sample_index"]),
            ("selected_sample_nearest_image", "", selected_generation["nearest_training_image"]),
            ("selected_sample_nearest_mse", "", selected_generation["nearest_training_mse"]),
        )
        for metric, ae_value, vae_value in rows:
            writer.writerow({"metric": metric, "ae": ae_value, "vae": vae_value})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, categories, recovered_truncated = load_bmp_dataset()
    write_dataset_manifest(dataset, categories, recovered_truncated)
    print(f"Dataset: {len(dataset.labels)} BMP; categorías={dict(Counter(categories))}")
    print(
        f"Configuración: encoder={list(ENCODER_LAYERS)}, decoder={list(DECODER_LAYERS)}, "
        f"lr={LEARNING_RATE}, kl_weight={KL_WEIGHT}, optimizer={OPTIMIZER}, "
        f"batch_size={BATCH_SIZE}, epochs={MAX_EPOCHS}."
    )

    autoencoder = build_autoencoder()
    ae_history = train_autoencoder(autoencoder, dataset.inputs)
    vae = build_vae()
    vae_history, vae_components = train_vae(vae, dataset.inputs)

    ae_reconstructions, ae_metrics = reconstruction_metrics(autoencoder, dataset.inputs)
    vae_reconstructions, vae_metrics = reconstruction_metrics(vae, dataset.inputs)

    write_dataset_preview(dataset, categories)
    write_architecture_diagram()
    write_vae_training_curves(vae_history, vae_components)
    write_latent_spaces(dataset, categories, autoencoder, vae)
    write_decoder_grid(vae)
    _, selected_generation = write_random_generations(dataset, vae)
    write_interpolation(dataset, categories, vae)
    write_reconstruction_comparison(
        dataset,
        ae_reconstructions,
        vae_reconstructions,
    )
    write_comparison_table(
        ae_metrics,
        vae_metrics,
        ae_history,
        vae_history,
        autoencoder,
        vae,
    )
    write_training_summary(
        dataset,
        categories,
        ae_metrics,
        vae_metrics,
        ae_history,
        vae_history,
        selected_generation,
    )
    print(f"Gráficos y métricas generados en {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
