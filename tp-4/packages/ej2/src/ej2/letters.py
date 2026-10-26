"""Module to handle letters task (2.2) for Hopfield networks.

Exposes a single entry-point `run_letters_task()` which is used to initiate the task.
"""

__all__ = ["LETTERS_5X5", "get_letter_patterns", "run_letters_task"]

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .analyze import AnalysisParams, build_orthogonal_sequence, run_full_analysis
from .hopfield import HopfieldNetwork
from .plots import out_dir

LETTER_SHAPE = (5, 5)
N_PIXELS = 25

# Diccionario completo A-Z en grillas de 5x5 con {-1, +1}.
# La letra "J" respeta exactamente la matriz de ejemplo de la página 6 del enunciado.
LETTERS_5X5: Dict[str, np.ndarray] = {
    "A": np.array([
        [-1, -1,  1, -1, -1],
        [-1,  1, -1,  1, -1],
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "B": np.array([
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "C": np.array([
        [-1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [-1,  1,  1,  1,  1],
    ], dtype=np.int8),
    "D": np.array([
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "E": np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ], dtype=np.int8),
    "F": np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
    ], dtype=np.int8),
    "G": np.array([
        [-1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1,  1],
    ], dtype=np.int8),
    "H": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "I": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ], dtype=np.int8),
    "J": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [ 1, -1, -1,  1, -1],
        [ 1,  1,  1, -1, -1],
    ], dtype=np.int8),
    "K": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1,  1, -1],
        [ 1,  1,  1, -1, -1],
        [ 1, -1, -1,  1, -1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "L": np.array([
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "M": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1,  1, -1,  1,  1],
        [ 1, -1,  1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "N": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1,  1, -1, -1,  1],
        [ 1, -1,  1, -1,  1],
        [ 1, -1, -1,  1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "O": np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "P": np.array([
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
    ], dtype=np.int8),
    "Q": np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1,  1, -1,  1],
        [ 1, -1, -1,  1, -1],
        [-1,  1,  1, -1,  1],
    ], dtype=np.int8),
    "R": np.array([
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1,  1, -1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "S": np.array([
        [-1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [-1,  1,  1,  1, -1],
        [-1, -1, -1, -1,  1],
        [ 1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "T": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
    ], dtype=np.int8),
    "U": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
    ], dtype=np.int8),
    "V": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
    ], dtype=np.int8),
    "W": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1,  1, -1,  1],
        [ 1,  1, -1,  1,  1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "X": np.array([
        [ 1, -1, -1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1,  1, -1,  1, -1],
        [ 1, -1, -1, -1,  1],
    ], dtype=np.int8),
    "Y": np.array([
        [ 1, -1, -1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
    ], dtype=np.int8),
    "Z": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1,  1, -1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ], dtype=np.int8),
}


def get_letter_patterns(letters: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """Devuelve la lista de nombres en mayúscula y la matriz (P, 25) de patrones en {-1, +1}."""
    names = [ch.upper() for ch in letters]
    for name in names:
        if name not in LETTERS_5X5:
            raise ValueError(f"Letra {name!r} no disponible en LETTERS_5X5.")
    patterns = np.stack([LETTERS_5X5[name].ravel() for name in names], axis=0)
    return names, patterns


def run_letters_task(cfg: Dict[str, Any]) -> None:
    """Punto de entrada principal para el ejercicio 2.2 (reconocimiento de letras 5x5)."""
    sub = "letras" if cfg.get("task") == "all" else ""
    out = out_dir(cfg, sub)
    names, patterns = get_letter_patterns(cfg["letters"])
    net = HopfieldNetwork(patterns, names)
    params = AnalysisParams.from_cfg(cfg)

    # Pool de patrones para el análisis de capacidad generalizado
    pool = {name: mat.ravel() for name, mat in LETTERS_5X5.items()}
    
    # Secuencias de capacidad: baja correlación vs alta correlación
    ortho_seq = build_orthogonal_sequence(cfg["letters"], pool, max_p=10)
    correlated_raw = ["E", "F", "P", "B", "R", "C", "G", "S", "H", "K"]
    correlated_seq = [ch for ch in correlated_raw if ch in pool][:10]

    capacity_sequences = {
        "Configuradas + baja correlación": ortho_seq,
        "Letras parecidas (alta correlación)": correlated_seq,
    }

    # Ejecutar el análisis completo mediante la generalización de analyze.py
    run_full_analysis(
        net=net,
        params=params,
        out=out,
        shape=LETTER_SHAPE,
        pool=pool,
        capacity_sequences=capacity_sequences,
        item_label="letra",
        tag="letters",
        include_random_capacity=False,
    )