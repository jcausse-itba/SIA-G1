import argparse
from typing import Any, Dict

from commons.config import (
    build_base_parser, csv_header, load_and_merge_config, nullable_float,
    nullable_int, positive_float, positive_int, require, require_columns,
    require_file, validate_common,
)

# Columnas que identifican al creador: nunca son features (constante del dataset).
ID_COLUMNS = ("creator_id", "name")


def build_parser() -> argparse.ArgumentParser:
    p = build_base_parser("TP4 Ej.1 - Campañas de marketing (PCA, Kohonen, Oja)", prog="tp4-marketing")

    # ---- Qué parte del enunciado correr
    p.add_argument_group("Pipeline").add_argument(
        "--task", choices=("eda", "pca", "kohonen", "oja", "campaign", "all"), default="all",
        help="Punto del enunciado: 1) eda/pca, 2) kohonen, 3) oja, 4) campaign.")

    # ---- 1. Exploración y selección de variables
    d = p.add_argument_group("Data & feature selection (punto 1)")
    d.add_argument("--dataset-path", type=str, default="datos/creadores.csv",
                   help="CSV de creadores de contenido.")
    d.add_argument("--exclude-features", nargs="+", default=None, metavar="COL",
                   help="Columnas descartadas tras el EDA (redundantes/derivadas).")

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
    k.add_argument("--final-lr", type=positive_float, default=0.01,
                   help="Learning rate final.")
    k.add_argument("--initial-radius", type=nullable_float, default=None,
                   help="Radio de vecindad inicial ('none' = max(filas, columnas)/2; decae linealmente).")
    k.add_argument("--final-radius", type=positive_float, default=1.0,
                   help="Radio de vecindad final.")
    k.add_argument("--n-runs", type=positive_int, default=10,
                   help="Corridas independientes por configuración (varía la inicialización).")

    # ---- 3. Oja
    o = p.add_argument_group("Oja (punto 3)")
    o.add_argument("--oja-lr", type=positive_float, default=1e-3, help="Learning rate de Oja.")
    o.add_argument("--oja-epochs", type=positive_int, default=500, help="Épocas de Oja.")

    # ---- 4. Decisión de campaña
    c = p.add_argument_group("Campaign decision (punto 4)")
    c.add_argument("--campaign", choices=("football", "dogs", "makeup", "art"), default="football",
                   help="Temática del cliente (usa la columna topic_<x>_pct).")
    c.add_argument("--objective", choices=("reach", "discovery", "depth"), default=None,
                   help="Objetivo: alcance masivo, descubrimiento o profundidad ('none' = compara los tres).")
    c.add_argument("--top-n", type=positive_int, default=10,
                   help="Cantidad de creadores a seleccionar / mostrar en los rankings.")
    return p


def validate_config(cfg: Dict[str, Any]) -> None:
    validate_common(cfg)

    require_file(cfg["dataset_path"], "Dataset")
    header = csv_header(cfg["dataset_path"])
    excluded = cfg["exclude_features"] or []
    require_columns(header, excluded, "--exclude-features")
    require(not set(excluded) & set(ID_COLUMNS), "--exclude-features cannot include identifier columns.")
    require_columns(header, [f"topic_{cfg['campaign']}_pct"], "--campaign")

    n_features = len(header) - len(ID_COLUMNS) - len(set(excluded))
    require(n_features >= 2, "At least 2 features are needed.")
    if cfg["n_components"] is not None:
        require(1 <= cfg["n_components"] <= n_features,
                f"--n-components must be in [1, {n_features}] (number of selected features).")

    if cfg["initial_radius"] is not None:
        require(cfg["final_radius"] <= cfg["initial_radius"], "--final-radius must be <= --initial-radius.")
    require(cfg["final_lr"] <= cfg["initial_lr"], "--final-lr must be <= --initial-lr.")


def parse_config(argv=None) -> Dict[str, Any]:
    cfg = load_and_merge_config(build_parser(), argv)
    validate_config(cfg)
    return cfg