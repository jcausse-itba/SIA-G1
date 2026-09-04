"""
Visualización de métricas del AG.

Funciones independientes, cada una recibe `history` (lista de dicts
con claves: generation, best, mean, std, worst) y guarda/muestra el gráfico.

Uso típico desde el engine o desde un script de análisis:

    from tp_2.metrics.plotting import (
        plot_fitness_curve,
        plot_diversity,
        plot_comparison,
        save_metrics_csv,
    )
    plot_fitness_curve(engine.history, path="out/fitness.png")
    save_metrics_csv(engine.history, path="out/metrics.csv")
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # sin GUI — funciona en servidores y sin DISPLAY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _extract(history: List[Dict], key: str) -> np.ndarray:
    return np.array([h[key] for h in history])


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Curva de fitness (mejor / media ± std / peor)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fitness_curve(
    history: List[Dict],
    path: str = "metrics/fitness_curve.png",
    title: str = "Evolución del Fitness",
    show: bool = False,
) -> None:
    """
    Grafica la evolución del fitness a lo largo de las generaciones.

    Muestra:
      - Línea del mejor individuo por generación.
      - Línea de la media poblacional.
      - Banda sombreada de ±1 std alrededor de la media.
      - Línea del peor individuo (referencia inferior).
    """
    gens  = _extract(history, "generation")
    best  = _extract(history, "best")
    mean  = _extract(history, "mean")
    std   = _extract(history, "std")
    worst = _extract(history, "worst")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(gens, best,  color="#2196F3", lw=2,   label="Mejor")
    ax.plot(gens, mean,  color="#FF9800", lw=1.5, label="Media", linestyle="--")
    ax.fill_between(gens, mean - std, mean + std,
                    alpha=0.2, color="#FF9800", label="±1 std")
    ax.plot(gens, worst, color="#9E9E9E", lw=1,   label="Peor",  linestyle=":")

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    _ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[metrics] Curva de fitness guardada en: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Diversidad poblacional (std del fitness como proxy)
# ─────────────────────────────────────────────────────────────────────────────

def plot_diversity(
    history: List[Dict],
    path: str = "metrics/diversity.png",
    title: str = "Diversidad Poblacional (std del fitness)",
    show: bool = False,
) -> None:
    """
    Grafica la desviación estándar del fitness por generación como proxy
    de diversidad: std alto → población diversa; std bajo → convergencia.
    """
    gens = _extract(history, "generation")
    std  = _extract(history, "std")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, std, color="#4CAF50", lw=1.5)
    ax.fill_between(gens, 0, std, alpha=0.15, color="#4CAF50")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Desviación estándar del fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    _ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[metrics] Diversidad guardada en: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Comparación entre múltiples corridas / configuraciones
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
]


def plot_comparison(
    runs: Dict[str, List[Dict]],
    path: str = "metrics/comparison.png",
    title: str = "Comparación de configuraciones",
    metric: str = "best",
    show: bool = False,
) -> None:
    """
    Superpone la curva `metric` de múltiples corridas en un mismo gráfico.

    Parameters
    ----------
    runs   : dict {etiqueta: history}.  Cada history es una lista de dicts.
    metric : qué métrica graficar ("best", "mean", "std", "worst").
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    for idx, (label, history) in enumerate(runs.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        gens  = _extract(history, "generation")
        vals  = _extract(history, metric)
        ax.plot(gens, vals, color=color, lw=1.8, label=label)

    ax.set_xlabel("Generación")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    _ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[metrics] Comparación guardada en: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Resumen final en barra (último fitness por corrida)
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_bar(
    runs: Dict[str, List[Dict]],
    path: str = "metrics/final_bar.png",
    title: str = "Fitness final por configuración",
    show: bool = False,
) -> None:
    """
    Barras del fitness del mejor individuo al final de cada corrida.
    Útil para comparar resultados de experimentos de forma compacta.
    """
    labels = list(runs.keys())
    finals = [max(h["best"] for h in history) for history in runs.values()]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, finals, color=colors, edgecolor="white", width=0.6)

    # Anotar valor encima de cada barra
    for bar, val in zip(bars, finals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002 * max(finals),
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Fitness final (mejor individuo)")
    ax.set_title(title)
    ax.set_ylim(0, max(finals) * 1.12)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    _ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[metrics] Barra final guardada en: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Exportar CSV con todas las métricas
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(
    history: List[Dict],
    path: str = "metrics/metrics.csv",
) -> None:
    """
    Guarda el history completo como CSV para análisis externo
    (Excel, R, pandas, etc.).
    """
    if not history:
        return
    _ensure_dir(path)
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"[metrics] CSV guardado en: {path}") 