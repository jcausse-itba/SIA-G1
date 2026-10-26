import argparse
import os
from pathlib import Path
from typing import Any, Dict

from commons.config import (
    build_base_parser, load_and_merge_config, positive_int, require,
    require_dir, unit_float, validate_common,
)


def build_parser(data_path: Path) -> argparse.ArgumentParser:
    p = build_base_parser("TP4 Ej.2 - Memoria asociativa (Hopfield)", prog="tp4-hopfield")

    p.add_argument_group("Pipeline").add_argument(
        "--task", choices=("letters", "qr", "all"), default="all",
        help="letters: caso simple 5x5 (2.2). qr: códigos QR 37x37 (2.1).")

    # ---- Patrones a almacenar
    d = p.add_argument_group("Patterns")
    d.add_argument("--letters", nargs="+", default=["A", "E", "J", "O"], metavar="LETTER",
                   help="Letras (matrices 5x5) a almacenar.")
    d.add_argument("--dataset-path", type=str, default=str(data_path),
                   help="Carpeta con las imágenes de los QR originales.")

    # ---- Red
    n = p.add_argument_group("Hopfield network")
    n.add_argument("--rule", choices=("hebb", "pseudoinverse"), default="hebb",
               help="Regla de aprendizaje: hebb (regla clásica) o pseudoinverse (proyección).")
    n.add_argument("--update-mode", choices=("synchronous", "asynchronous"), default="asynchronous",
                   help="Actualización de neuronas: todas a la vez o una por vez.")
    n.add_argument("--max-iterations", type=positive_int, default=100,
                   help="Tope de iteraciones (barridos) si la red no converge.")

    # ---- Ruido
    r = p.add_argument_group("Noise")
    r.add_argument("--type", choices=("perlin", "finger", "white", "flip"), default="white", 
                   help="Función de ruido")
    r.add_argument("--noise-levels", nargs="+", type=unit_float,
                   default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50], metavar="P",
                   help="Fracción de píxeles invertidos en cada consulta.")
    r.add_argument("--trials", type=positive_int, default=20,
                   help="Consultas ruidosas independientes por nivel de ruido y patrón.")
    r.add_argument("--spurious-noise", type=unit_float, default=0.4,
                   help="Nivel de ruido del patrón 'muy ruidoso' usado para buscar estados espurios.")
    return p


def validate_config(cfg: Dict[str, Any]) -> None:
    validate_common(cfg)

    if cfg["task"] in ("letters", "all"):
        letters = [l.upper() for l in cfg["letters"]]
        require(all(len(l) == 1 and l.isascii() and l.isalpha() for l in letters),
                "--letters must be single letters A-Z.")
        require(len(set(letters)) == len(letters), "--letters contains duplicates.")
        require(len(letters) >= 2, "Store at least 2 letters.")

    if cfg["task"] in ("qr", "all"):
        require_dir(cfg["dataset_path"], "QR dataset")
        imgs = [f for f in os.listdir(cfg["dataset_path"])
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        require(len(imgs) >= 2, f"Need at least 2 QR images in {cfg['dataset_path']}.")


def parse_config(data_path: Path, argv=None) -> Dict[str, Any]:
    cfg = load_and_merge_config(build_parser(data_path), argv)
    validate_config(cfg)
    return cfg
