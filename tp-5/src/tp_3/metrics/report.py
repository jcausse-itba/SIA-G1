"""One Plotly .html per chart for the fraud exercise, written into a folder."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .metrics import (
    DEFAULT_POINTS,
    EpochMetricsTracker,
    area_under,
    binarize,
    threshold_sweep,
)

COLORS = {
    "loss": "#54a24b", "accuracy": "#4c78a8", "precision": "#f58518",
    "recall": "#278276", "f1": "#b279a2", "tpr": "#278276", "fpr": "#bd4b45",
}
EPOCH_CHARTS = (
    ("loss", "Loss"), ("accuracy", "Accuracy"), ("precision", "Precision"),
    ("recall", "Recall"), ("f1", "F1 score"), ("tpr", "TPR (tasa de verdaderos positivos)"),
    ("fpr", "FPR (tasa de falsos positivos)"),
)


def _save(fig: go.Figure, path: Path, title: str, xtitle: str, ytitle: str, unit_y: bool = True) -> None:
    fig.update_layout(title_text=title, template="plotly_white", height=550,
                      xaxis_title=xtitle, yaxis_title=ytitle, hovermode="x unified")
    if unit_y:
        fig.update_yaxes(range=[-0.02, 1.02])
    fig.write_html(path, include_plotlyjs="directory",
                   config={"responsive": True, "displaylogo": False})


def write_fraud_metrics_report(
    tracker: EpochMetricsTracker,
    val_targets: np.ndarray,
    val_scores: np.ndarray,
    output_dir: str | Path,
    n_points: int = DEFAULT_POINTS,
) -> dict[str, float]:
    """Writes one .html per chart into `output_dir`; returns best-F1 threshold and ROC AUC."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = tracker.records
    epochs = rec["epoch"]
    subtitle = (f"(fraude real = prob. BigModel ≥ {tracker.fraud_cutoff}, "
                f"umbral = {tracker.threshold}, primeras {len(epochs)} épocas)")

    # ---- Per-epoch charts: train (dotted) vs validation (solid)
    for key, label in EPOCH_CHARTS:
        fig = go.Figure()
        for split, name, dash in (("train", "Train", "dot"), ("val", "Validación", None)):
            fig.add_trace(go.Scatter(
                x=epochs, y=rec[f"{split}_{key}"], mode="lines", name=name,
                line={"color": COLORS[key], "dash": dash, "width": 2},
            ))
        _save(fig, out / f"{key}.html", f"{label} por época {subtitle}", "Época", label,
              unit_y=key != "loss")

    # ---- Threshold charts (validation set)
    actual = binarize(val_targets, tracker.fraud_cutoff)
    scores = np.clip(np.asarray(val_scores).ravel(), 0.0, 1.0)
    sweep = threshold_sweep(actual, scores, n_points)
    best = int(np.argmax(sweep["f1"]))
    best_threshold, best_f1 = float(sweep["threshold"][best]), float(sweep["f1"][best])
    roc_auc = area_under(sweep["fpr"], sweep["tpr"])

    def vlines(fig: go.Figure) -> None:
        fig.add_vline(x=best_threshold, line_dash="dash", line_color="#888",
                      annotation_text=f"mejor F1 @ {best_threshold:.2f}")
        fig.add_vline(x=tracker.threshold, line_dash="dot", line_color="#bd4b45",
                      annotation_text=f"umbral config {tracker.threshold}", annotation_position="bottom right")

    fig = go.Figure()
    for m in ("accuracy", "precision", "recall", "f1"):
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[m], mode="lines", name=m,
                                 line={"color": COLORS[m], "width": 2}))
    vlines(fig)
    _save(fig, out / "threshold_metrics.html", "Accuracy / Precision / Recall / F1 vs umbral (validación)",
          "Umbral", "Valor")

    fig = go.Figure()
    for m in ("tpr", "fpr"):
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[m], mode="lines", name=m.upper(),
                                 line={"color": COLORS[m], "width": 2}))
    vlines(fig)
    _save(fig, out / "threshold_rates.html", "TPR y FPR vs umbral (validación)", "Umbral", "Tasa")

    hover = [f"umbral {t:.2f}" for t in sweep["threshold"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["fpr"], y=sweep["tpr"], mode="lines+markers", name="ROC",
                             marker={"size": 4}, line={"color": COLORS["tpr"]}, text=hover))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                             line={"dash": "dash", "color": "#999"}))
    _save(fig, out / "roc.html", f"Curva ROC (AUC≈{roc_auc:.3f})", "FPR", "TPR")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["recall"], y=sweep["precision"], mode="lines+markers",
                             name="Precision-Recall", marker={"size": 4},
                             line={"color": COLORS["precision"]}, text=hover))
    _save(fig, out / "precision_recall.html", "Curva Precision-Recall", "Recall", "Precision")

    return {"best_threshold": best_threshold, "best_f1": best_f1, "roc_auc": roc_auc}