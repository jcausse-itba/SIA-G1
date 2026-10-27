import argparse
from typing import Any, Dict

from commons.config import (
    build_base_parser, csv_header, grid_shape, load_and_merge_config,
    non_negative_float, nullable_float, nullable_int, positive_float,
    positive_int, require, require_columns, require_file, validate_common,
)

# Columnas que identifican al creador: nunca son features (constante del dataset).
ID_COLUMNS = ("creator_id", "name")

# Defaults justificados en el EDA (ver output/eda/derived_checks.json y correlacion_agrupada.html):
#  - price_ars = 1500 * price_usd                        (lineal exacta)
#  - total_engagement_per_1000 = likes+comments+saves+shares  (lineal exacta)
#  - organic_posts_pct = 100 - sponsored_posts_pct       (lineal exacta)
#  - estimated_argentina_reach = reach * arg_pct / 100   (producto de dos features; se usa en el punto 4)
#  - average_impressions, average_video_views            (r >= 0.987 con average_reach)
DEFAULT_EXCLUDED = (
    "price_ars", "total_engagement_per_1000", "organic_posts_pct",
    "estimated_argentina_reach", "average_impressions", "average_video_views",
)
# Conteos / montos con cola derecha: se les aplica log1p antes de estandarizar.
DEFAULT_LOG = (
    "followers", "average_reach", "average_impressions", "average_video_views",
    "estimated_argentina_reach", "price_usd",
)

TASKS = ("eda", "pca", "kohonen", "sweep", "oja", "campaign", "all")
CAMPAIGNS = ("football", "dogs", "makeup", "art")


def build_parser() -> argparse.ArgumentParser:
    p = build_base_parser("TP4 Ej.1 - Campañas de marketing (PCA, Kohonen, Oja)", prog="tp4-marketing")

    # ---- Qué parte del enunciado correr
    p.add_argument_group("Pipeline").add_argument(
        "--task", choices=TASKS, default="all",
        help="Punto del enunciado: 1) eda/pca, 2) kohonen/sweep, 3) oja, 4) campaign.")

    # ---- 1. Exploración y selección de variables
    d = p.add_argument_group("Data & feature selection (punto 1)")
    d.add_argument("--dataset-path", type=str, default="data/creadores.csv",
                   help="CSV de creadores de contenido.")
    d.add_argument("--exclude-features", nargs="*", default=list(DEFAULT_EXCLUDED), metavar="COL",
                   help="Columnas descartadas tras el EDA (redundantes/derivadas).")
    d.add_argument("--log-features", nargs="*", default=list(DEFAULT_LOG), metavar="COL",
                   help="Columnas a las que se aplica log1p antes de estandarizar "
                        "(las excluidas se ignoran).")

    # ---- 1. PCA
    pc = p.add_argument_group("PCA (punto 1)")
    pc.add_argument("--n-components", type=nullable_int, default=None,
                    help="Componentes a conservar ('none' = todas, para ver el scree plot).")

    # ---- 2. Kohonen
    k = p.add_argument_group("Kohonen (punto 2)")
    k.add_argument("--grid-rows", type=positive_int, default=5, help="Filas de la grilla.")
    k.add_argument("--grid-cols", type=positive_int, default=5, help="Columnas de la grilla.")
    k.add_argument("--iterations", type=positive_int, default=5000,
                   help="Iteraciones de entrenamiento (una muestra por iteración).")
    k.add_argument("--init-method", choices=("random", "sample"), default="sample",
                   help="Pesos iniciales: aleatorios o copias de muestras reales.")
    k.add_argument("--initial-lr", type=positive_float, default=0.5,
                   help="Learning rate inicial (decae linealmente).")
    k.add_argument("--final-lr", type=positive_float, default=0.001,
                   help="Learning rate final.")
    k.add_argument("--initial-radius", type=nullable_float, default=None,
                   help="Radio de vecindad inicial ('none' = max(filas, columnas)/2; decae linealmente).")
    k.add_argument("--final-radius", type=positive_float, default=1.0,
                   help="Radio de vecindad final.")
    k.add_argument("--n-runs", type=positive_int, default=10,
                   help="Corridas independientes por configuración (varía la inicialización).")
    k.add_argument("--n-regions", type=positive_int, default=5,
                   help="Regiones de la grilla (clustering jerárquico de los pesos) para interpretar.")

    # ---- 2. Barrido de hiperparámetros
    s = p.add_argument_group("Kohonen sweep (punto 2)")
    s.add_argument("--sweep-grids", type=grid_shape, nargs="+",
                   default=[(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (10, 10)],
                   metavar="RxC", help="Grillas a probar (ej. 5x5 7x7).")
    s.add_argument("--sweep-lrs", type=positive_float, nargs="+", default=[0.1, 0.5, 0.9],
                   help="Learning rates iniciales a probar.")
    s.add_argument("--sweep-radii", type=nullable_float, nargs="+", default=[None],
                   help="Radios iniciales a probar ('none' = max(filas, columnas)/2).")
    s.add_argument("--sweep-iterations", type=positive_int, nargs="+", default=None,
                   help="Iteraciones a probar (default: solo --iterations).")

    # ---- 3. Oja
    o = p.add_argument_group("Oja (punto 3)")
    o.add_argument("--oja-lr", type=positive_float, default=0.0001, help="Learning rate de Oja.")
    o.add_argument("--oja-epochs", type=positive_int, default=500, help="Épocas de Oja.")

    # ---- 4. Decisión de campaña
    c = p.add_argument_group("Campaign decision (punto 4)")
    c.add_argument("--campaign", choices=CAMPAIGNS, default="football",
                   help="Temática del cliente (usa la columna topic_<x>_pct). Con --task all se corren las cuatro.")
    c.add_argument("--objective", choices=("reach", "discovery", "depth", "all"), default="all",
                   help="Objetivo: alcance masivo, descubrimiento o profundidad ('all' = compara los tres).")
    c.add_argument("--min-topic-pct", type=non_negative_float, default=30.0,
                   help="Mínimo de topic_<campaña>_pct para que un creador sea elegible.")
    c.add_argument("--top-n", type=positive_int, default=5,
                   help="Cantidad de creadores a seleccionar / mostrar en los rankings.")
    return p


def validate_config(cfg: Dict[str, Any]) -> None:
    validate_common(cfg)

    require_file(cfg["dataset_path"], "Dataset")
    header = csv_header(cfg["dataset_path"])
    excluded = cfg["exclude_features"] or []
    require_columns(header, excluded, "--exclude-features")
    require_columns(header, cfg["log_features"] or [], "--log-features")
    require(not set(excluded) & set(ID_COLUMNS), "--exclude-features cannot include identifier columns.")
    require(not set(cfg["log_features"] or []) & set(ID_COLUMNS), "--log-features cannot include identifier columns.")
    for camp in CAMPAIGNS:
        require_columns(header, [f"topic_{camp}_pct"], "--campaign")
    require(cfg["min_topic_pct"] <= 100, "--min-topic-pct must be in [0, 100].")

    n_features = len(header) - len(ID_COLUMNS) - len(set(excluded))
    require(n_features >= 2, "At least 2 features are needed.")
    if cfg["n_components"] is not None:
        require(1 <= cfg["n_components"] <= n_features,
                f"--n-components must be in [1, {n_features}] (number of selected features).")

    if cfg["initial_radius"] is not None:
        require(cfg["initial_radius"] > 0, "--initial-radius must be > 0 or 'none'.")
        require(cfg["final_radius"] <= cfg["initial_radius"], "--final-radius must be <= --initial-radius.")
    for r in cfg["sweep_radii"]:
        require(r is None or r >= cfg["final_radius"], "--sweep-radii must be >= --final-radius (or 'none').")
    require(cfg["final_lr"] <= cfg["initial_lr"], "--final-lr must be <= --initial-lr.")
    require(all(lr >= cfg["final_lr"] for lr in cfg["sweep_lrs"]), "--sweep-lrs must be >= --final-lr.")
    require(cfg["n_regions"] <= cfg["grid_rows"] * cfg["grid_cols"],
            "--n-regions cannot exceed the number of neurons.")


def parse_config(argv=None) -> Dict[str, Any]:
    cfg = load_and_merge_config(build_parser(), argv)
    validate_config(cfg)
    return cfg
