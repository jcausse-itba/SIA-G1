"""
Module to handle QR codes task (2.1) for Hopfield networks.

Exposes a single entry-point `run_qr_task()` which is used to initiate the task.
"""

__all__ = ["QR_SHAPE", "load_qr_patterns", "run_qr_task"]

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from .analyze import AnalysisParams, build_orthogonal_sequence, run_full_analysis
from .hopfield import HopfieldNetwork
from .plots import DIVERGING, out_dir, save_csv, save_fig

QR_SHAPE = (37, 37)
N_PIXELS = QR_SHAPE[0] * QR_SHAPE[1]  # 1369 neuronas

QR_ORDER = ("food", "wine", "beverages", "dessert", "specials")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")

QUIET_ZONE = 4
UPSCALE = 10
RANDOM_STARTS = 100

Decoder = Callable[[np.ndarray], str]


# ============================================================================
# Carga de datos
# ============================================================================
def load_qr_patterns(folder: str | Path) -> Tuple[List[str], np.ndarray]:
    """Carga los QR de `folder`, los lleva a 37x37 (si hiciera falta) y los binariza a {-1, +1}."""
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT]
    if not files:
        raise FileNotFoundError(f"No hay imágenes {IMAGE_EXT} en {folder}")

    def order(p: Path) -> Tuple[int, str]:
        stem = p.stem.lower()
        return next((i for i, n in enumerate(QR_ORDER) if stem.startswith(n)), len(QR_ORDER)), stem

    names: List[str] = []
    pats: List[np.ndarray] = []
    for path in sorted(files, key=order):
        img = Image.open(path).convert("L")
        if img.size != (QR_SHAPE[1], QR_SHAPE[0]):
            img = img.resize((QR_SHAPE[1], QR_SHAPE[0]), Image.Resampling.BOX)
        pats.append(np.where(np.asarray(img) < 128, 1, -1).astype(np.int8).ravel())
        names.append(path.stem.lower().removesuffix("_qr").removesuffix("_menu").capitalize())
    return names, np.stack(pats, axis=0)


def _make_decoder(patterns: np.ndarray) -> Optional[Tuple[Decoder, List[str]]]:
    """Lector de QR (OpenCV) para comprobar si un estado realmente se puede escanear."""
    try:
        import cv2
    except ImportError:
        print("[qr] opencv no instalado: no se verifica la decodificación "
              "(pip install opencv-python-headless).")
        return None

    detector = cv2.QRCodeDetector()

    def decode(vec: np.ndarray) -> str:
        img = np.where(np.asarray(vec).reshape(QR_SHAPE) > 0, 0, 255).astype(np.uint8)
        img = np.pad(img, QUIET_ZONE, constant_values=255)
        img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
        try:
            return detector.detectAndDecode(img)[0]
        except cv2.error:
            return ""

    refs = [decode(p) for p in patterns]
    if not all(refs):
        print("[qr] OpenCV no pudo leer los QR originales: se omite la verificación de decodificación.")
        return None
    return decode, refs


# ============================================================================
# Helpers de gráficos específicos de QR
# ============================================================================
def _style_image_axes(fig: go.Figure, n_subplots: int) -> None:
    for k in range(1, n_subplots + 1):
        sfx = "" if k == 1 else str(k)
        fig.layout[f"xaxis{sfx}"].update(showticklabels=False, showgrid=False, zeroline=False)
        fig.layout[f"yaxis{sfx}"].update(
            showticklabels=False, showgrid=False, zeroline=False,
            autorange="reversed", scaleanchor=f"x{sfx}", scaleratio=1,
        )


def analyze_common_structure(net: HopfieldNetwork, out: str) -> float:
    """Mapa de qué píxeles son idénticos en todos los QR."""
    same = (net.patterns == net.patterns[0]).all(axis=0)
    frac_same = float(same.mean())
    consensus = net.patterns.astype(np.float64).mean(axis=0).reshape(QR_SHAPE)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Promedio de los QR por píxel (±1 = idéntico en todos)",
                        f"Píxeles idénticos en los {net.n_patterns} QR: {frac_same:.1%}"),
        horizontal_spacing=0.08,
    )
    fig.add_heatmap(z=consensus, zmin=-1, zmax=1, colorscale=DIVERGING, hoverinfo="z",
                    colorbar={"title": "media", "len": 0.8, "x": 0.45}, row=1, col=1)
    fig.add_heatmap(z=same.reshape(QR_SHAPE).astype(float), zmin=0, zmax=1,
                    colorscale=[[0.0, "#f5f6fa"], [1.0, "#1e8449"]], showscale=False,
                    hoverinfo="z", row=1, col=2)
    _style_image_axes(fig, 2)
    fig.update_layout(title="Estructura común entre los QR almacenados", height=520, width=1100)
    save_fig(fig, out, "estructura_comun")
    return frac_same


# ============================================================================
# Limitaciones del caso de uso
# ============================================================================
def analyze_limitations(
    net: HopfieldNetwork,
    df_stored: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_summary: pd.DataFrame,
    frac_same: float,
    out: str,
) -> pd.DataFrame:
    """Tabla con los números que respaldan las limitaciones del caso de uso."""
    p = net.n_patterns
    real_side = QR_SHAPE[0] * UPSCALE
    real_n = real_side ** 2
    expected = float(np.sqrt(2.0 / (np.pi * N_PIXELS)))
    n_fixed = int(df_stored["is_fixed_point"].sum())
    unstable_total = int(df_stored["unstable_pixels"].sum()) if "unstable_pixels" in df_stored.columns else 0
    ok_levels = df_summary.loc[df_summary["recovery_rate"] >= 0.9, "noise_level"]
    max_ok: float | str = float(ok_levels.max()) if len(ok_levels) else "ninguno"

    items = [
        ("neuronas", N_PIXELS, f"{QR_SHAPE[0]}×{QR_SHAPE[1]} píxeles, una neurona por módulo"),
        ("pesos_en_W", N_PIXELS ** 2, f"{N_PIXELS ** 2 * 8 / 1e6:.1f} MB en float64"),
        ("pesos_en_W_con_QR_real", real_n ** 2,
         f"con {real_side}×{real_side} px (10 px/módulo): {real_n ** 2 * 8 / 1e9:.0f} GB -> no escala"),
        ("carga_alpha", round(p / N_PIXELS, 4),
         "muy por debajo de 0.138 (capacidad teórica); "
         "el problema no es la cantidad sino la correlación entre los QR"),
        ("solapamiento_medio_QR", round(float(df_pairs['abs_overlap'].mean()), 3),
         f"patrones aleatorios independientes: ≈{expected:.3f} -> los QR están muy correlacionados"),
        ("pixeles_identicos_en_todos_los_QR", round(frac_same, 4),
         "posición, alineación, timing, versión/formato y el prefijo común de la URL"),
        ("qr_puntos_fijos", f"{n_fixed}/{p}",
         f"puntos fijos estables: {n_fixed}/{p} ({unstable_total} píxeles inestables en total)"),
        ("ruido_max_con_recuperacion>=90%", max_ok, "nivel más alto de ruido con ≥90% de recuperación exacta"),
    ]
    df_lim = pd.DataFrame(items, columns=["metric", "value", "comment"])
    save_csv(df_lim, out, "limitaciones")
    return df_lim


# ============================================================================
# Entry point
# ============================================================================
def run_qr_task(cfg: Dict[str, Any]) -> None:
    """Punto de entrada principal para el ejercicio 2.1 (recuperación de códigos QR deteriorados)."""
    sub = "qr" if cfg.get("task") == "all" else ""
    out = out_dir(cfg, sub)
    names, patterns = load_qr_patterns(cfg["dataset_path"])
    net = HopfieldNetwork(patterns, names)
    params = AnalysisParams.from_cfg(cfg)

    pool = {name: patterns[i] for i, name in enumerate(names)}
    ortho_seq = build_orthogonal_sequence(names, pool, max_p=len(names))
    capacity_sequences = {
        "QR del menú (orden natural)": names,
        "Secuencia optimizada (ortogonalidad)": ortho_seq,
    }

    # Ejecutar el análisis completo mediante analyze.py (incluyendo capacidad aleatoria recomendada para pocos patrones)
    results = run_full_analysis(
        net=net,
        params=params,
        out=out,
        shape=QR_SHAPE,
        pool=pool,
        capacity_sequences=capacity_sequences,
        item_label="código QR",
        tag="qr",
        include_random_capacity=True,
    )

    # Análisis estructural y limitaciones específicos de QR
    frac_same = analyze_common_structure(net, out)
    analyze_limitations(net, results.stored, results.orthogonality.pairs, results.noise_summary, frac_same, out)

    print(f"\n[qr] Análisis de códigos QR completado con éxito. Resultados guardados en: {out}/")