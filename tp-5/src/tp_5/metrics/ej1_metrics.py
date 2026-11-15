"""Exercise 1 charts. Reads the files written by tp_3/ej1_study.py and writes one .html per chart.

  compare_*   linear vs non-linear (all samples)
  design_*    dataset and split design
  epoch_*     per-epoch curves of the chosen model (mean +/- std across folds)
  thr_*       threshold analysis on pooled out-of-fold predictions
  final_*     recommended threshold: confusion matrix, metrics table, predicted vs actual
  robust_*    how solid the conclusions are (per fold, bootstrap, fraud cutoff, calibration)
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .metrics import area_under

COLORS = {"loss": "#54a24b", "accuracy": "#4c78a8", "precision": "#f58518", "recall": "#278276",
          "f1": "#b279a2", "tpr": "#278276", "fpr": "#bd4b45"}
EPOCH_CHARTS = (("loss", "Loss"), ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                ("f1", "F1 score"), ("tpr", "TPR (tasa de verdaderos positivos)"), ("fpr", "FPR (tasa de falsos positivos)"))
LINEAR_C, NONLINEAR_C = "#bd4b45", "#4c78a8"


# ---------------------------------------------------------------------------- helpers
def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _save(fig: go.Figure, path: Path, title: str, xtitle: str | None = None, ytitle: str | None = None,
          height: int = 550, unit_y: bool = False, hover: str = "x unified") -> None:
    fig.update_layout(title_text=title, template="plotly_white", height=height, hovermode=hover)
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    if unit_y:
        fig.update_yaxes(range=[-0.02, 1.02])
    fig.write_html(path, include_plotlyjs="directory", config={"responsive": True, "displaylogo": False})


def _div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def sweep_counts(actual: np.ndarray, scores: np.ndarray, thresholds: np.ndarray):
    """tp, fp, fn, tn for every threshold (prediction = score >= threshold), vectorised by sorting."""
    pos, neg = np.sort(scores[actual]), np.sort(scores[~actual])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    return tp.astype(float), fp.astype(float), len(pos) - tp.astype(float), len(neg) - fp.astype(float)


def rates(tp, fp, fn, tn) -> dict[str, np.ndarray]:
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"precision": precision, "recall": recall, "tpr": recall, "fpr": _div(fp, fp + tn),
            "f1": _div(2 * precision * recall, precision + recall),
            "accuracy": _div(tp + tn, tp + fp + fn + tn)}


def _at(actual, scores, threshold) -> dict[str, float]:
    r = rates(*sweep_counts(actual, scores, np.array([threshold])))
    return {k: float(v[0]) for k, v in r.items()}


def _reg(actual, pred) -> dict[str, float]:
    res = pred - actual
    var = float(np.sum((actual - actual.mean()) ** 2))
    return {"MAE": float(np.mean(np.abs(res))), "RMSE": float(np.sqrt(np.mean(res ** 2))),
            "R2": float(1 - np.sum(res ** 2) / var) if var > 0 else float("nan")}


# --------------------------------------------------------------------------- compare
def plot_compare(out: Path, data: dict) -> list[Path]:
    models = data["models"]
    names = {m["role"]: f"{m['model_type']} ({m['activation']})" for m in models}
    colors = {"chosen": NONLINEAR_C, "other": LINEAR_C}

    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Scatter(x=np.arange(1, len(m["loss_history"]) + 1), y=m["loss_history"], mode="lines",
                                 name=names[m["role"]], line={"color": colors[m["role"]], "width": 2}))
    fig.update_yaxes(type="log")
    _save(fig, out / "compare_loss.html", f"Loss por época, todas las muestras ({data['n_samples']:,}), escala log",
          "Época", "Loss (MSE, log)")

    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[f"{names[m['role']]}  R²={m['metrics']['R2']:.3f}" for m in models])
    lo = min(min(m["sample_actual"] + m["sample_pred"]) for m in models)
    hi = max(max(m["sample_actual"] + m["sample_pred"]) for m in models)
    for i, m in enumerate(models, start=1):
        fig.add_trace(go.Scatter(x=m["sample_actual"], y=m["sample_pred"], mode="markers", opacity=0.45,
                                 marker={"size": 4, "color": colors[m["role"]]}, name=names[m["role"]],
                                 showlegend=False), row=1, col=i)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#444"},
                                 name="Predicción perfecta", showlegend=(i == 1)), row=1, col=i)
        for level in (0, 1):  # a probability must stay inside [0, 1]
            fig.add_hline(y=level, line_dash="dot", line_color="#999", row=1, col=i)
        fig.update_xaxes(title_text="Probabilidad BigModel (real)", row=1, col=i)
    fig.update_yaxes(title_text="Predicción TinyModel", row=1, col=1)
    _save(fig, out / "compare_pred_vs_actual.html", "Predicho vs real (todas las muestras; líneas punteadas = 0 y 1)",
          height=520, hover="closest")

    fig = go.Figure(go.Table(
        header={"values": ["Modelo", "Loss final", "MAE", "RMSE", "R²", "% predicciones fuera de [0,1]"],
                "fill_color": "#eaeaea", "align": "left"},
        cells={"values": [[names[m["role"]] for m in models], [f"{m['loss_history'][-1]:.5f}" for m in models],
                          [f"{m['metrics']['MAE']:.4f}" for m in models], [f"{m['metrics']['RMSE']:.4f}" for m in models],
                          [f"{m['metrics']['R2']:.4f}" for m in models],
                          [f"{100 * m['share_outside_unit']:.2f}%" for m in models]], "align": "left"}))
    _save(fig, out / "compare_table.html", "Lineal vs no lineal (todas las muestras)", height=260)
    return [out / f"compare_{n}.html" for n in ("loss", "pred_vs_actual", "table")]


# ---------------------------------------------------------------------------- design
def plot_design(out: Path, d: dict) -> list[Path]:
    cutoff, edges, counts = d["cutoff"], np.array(d["hist"]["edges"]), np.array(d["hist"]["counts"])
    fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges), marker_color="#4c78a8"))
    fig.add_vline(x=cutoff, line_dash="dash", line_color="#bd4b45",
                  annotation_text=f"corte de fraude = {cutoff} ({100 * d['positive_rate']:.2f}% positivos)")
    fig.update_yaxes(type="log")
    _save(fig, out / "design_target_hist.html", f"Distribución de la probabilidad de BigModel ({d['n_rows']:,} filas)",
          "Probabilidad de fraude (BigModel)", "Cantidad de transacciones (log)")

    color = {"train": "#4c78a8", "val": "#f58518", "unused": "#cccccc"}
    label = {"train": "Entrenamiento", "val": "Validación", "unused": "No usado"}
    fig, seen = go.Figure(), set()
    for row_name, roles in d["roles"].items():
        for chunk, role in zip(d["chunks"], roles):
            fig.add_trace(go.Bar(
                y=[row_name], x=[chunk["width_pct"]], base=[chunk["start_pct"]], orientation="h",
                marker={"color": color[role], "line": {"color": "white", "width": 1}},
                name=label[role], legendgroup=role, showlegend=role not in seen,
                hovertemplate=f"{label[role]}<br>{chunk['n']:,} filas<br>{chunk['positives']:,} positivos "
                              f"({100 * chunk['rate']:.2f}%)<extra></extra>"))
            seen.add(role)
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(autorange="reversed")
    cv = "k-fold (entrena también con tramos futuros)" if d["cv"] == "kfold" else "forward chaining (solo pasado)"
    _save(fig, out / "design_folds.html", f"Esquema de folds sobre el orden cronológico: {cv}",
          "Posición en el orden cronológico (%)", None, height=130 + 60 * len(d["roles"]), hover="closest")

    fig = go.Figure(go.Bar(x=[f"Tramo {c['chunk'] + 1}" for c in d["chunks"]],
                           y=[100 * c["rate"] for c in d["chunks"]], marker_color="#4c78a8",
                           text=[f"{c['positives']:,}/{c['n']:,}" for c in d["chunks"]], textposition="outside"))
    fig.add_hline(y=100 * d["positive_rate"], line_dash="dash", line_color="#bd4b45",
                  annotation_text=f"global {100 * d['positive_rate']:.2f}%")
    _save(fig, out / "design_chunk_positive_rate.html", f"% de fraude (prob. ≥ {cutoff}) por tramo cronológico",
          None, "% de positivos", hover="closest")
    return [out / f"design_{n}.html" for n in ("target_hist", "folds", "chunk_positive_rate")]


# ----------------------------------------------------------------------------- study
def _load_study(out: Path):
    meta = json.loads((out / "study.json").read_text())
    arrays = np.load(out / "study.npz")
    folds = [{**f, "scores": np.clip(arrays[f"scores_{f['fold']}"], 0.0, 1.0), "targets": arrays[f"targets_{f['fold']}"]}
             for f in meta["folds"]]
    return meta, folds


def analyze(meta: dict, folds: list[dict], recommend: str, cost_ratios, bootstrap: int, cutoffs) -> dict[str, Any]:
    cutoff = meta["fraud_cutoff"]
    grid = np.linspace(0.0, 1.0, max(2, int(meta.get("metrics_points", 100))))
    scores = np.concatenate([f["scores"] for f in folds])
    targets = np.concatenate([f["targets"] for f in folds])
    fold_id = np.concatenate([np.full(len(f["scores"]), f["fold"]) for f in folds])
    actual = targets >= cutoff
    n = len(actual)
    if actual.sum() == 0 or actual.all():
        raise ValueError(f"With fraud_cutoff={cutoff} there are no positives or no negatives; pick another cutoff.")

    tp, fp, fn, tn = sweep_counts(actual, scores, grid)
    r = rates(tp, fp, fn, tn)
    cost = {float(ratio): (ratio * fn + fp) / n * 1000 for ratio in cost_ratios}
    thr_f1 = float(grid[np.argmax(r["f1"])])
    thr_cost = {ratio: float(grid[np.argmin(c)]) for ratio, c in cost.items()}
    if recommend == "f1":
        thr, rule = thr_f1, "máximo F1"
    elif recommend.startswith("cost:"):
        ratio = float(recommend.split(":")[1])
        thr, rule = float(grid[np.argmin(ratio * fn + fp)]), f"mínimo costo (FN:FP = {ratio:g}:1)"
    else:
        thr, rule = float(recommend), "umbral fijado a mano"

    own, lofo, per_fold = [], [], []
    for f in folds:
        a_i = f["targets"] >= cutoff
        own.append(float(grid[np.argmax(rates(*sweep_counts(a_i, f["scores"], grid))["f1"])]) if a_i.any() else None)
        mask = fold_id != f["fold"]
        lofo.append(float(grid[np.argmax(rates(*sweep_counts(actual[mask], scores[mask], grid))["f1"])]))
        per_fold.append({**_at(a_i, f["scores"], thr), **_reg(f["targets"], f["scores"]), "positives": int(a_i.sum())})

    rng, pred_pos = np.random.default_rng(0), scores >= thr
    boots = {k: [] for k in ("precision", "recall", "f1")}
    for _ in range(max(0, bootstrap)):
        idx = rng.integers(0, n, n)
        a, p = actual[idx], pred_pos[idx]
        t, f_p, f_n = float(np.sum(a & p)), float(np.sum(~a & p)), float(np.sum(a & ~p))
        pr, rc = (t / (t + f_p) if t + f_p else 0.0), (t / (t + f_n) if t + f_n else 0.0)
        boots["precision"].append(pr); boots["recall"].append(rc)
        boots["f1"].append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items() if v}

    sens = []
    for c in cutoffs:
        act_c = targets >= c
        if act_c.sum() == 0 or act_c.all():
            continue
        rc_ = rates(*sweep_counts(act_c, scores, grid))
        sens.append({"cutoff": float(c), "positive_rate": float(act_c.mean()), "best_f1": float(rc_["f1"].max()),
                     "best_threshold": float(grid[np.argmax(rc_["f1"])]), "auc": area_under(rc_["fpr"], rc_["tpr"]),
                     "f1_at_recommended": _at(act_c, scores, thr)["f1"]})

    bins = np.linspace(0, 1, 11)
    which = np.digitize(scores, bins[1:-1])
    calib = [{"mean_pred": float(scores[which == b].mean()), "mean_actual": float(targets[which == b].mean()),
              "count": int((which == b).sum())} for b in range(10) if (which == b).any()]

    return {"cutoff": cutoff, "n": n, "positives": int(actual.sum()), "grid": grid, "rates": r,
            "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "cost": cost, "thr_f1": thr_f1, "thr_cost": thr_cost,
            "threshold": thr, "rule": rule, "config_threshold": meta["threshold"], "own_thr": own, "lofo_thr": lofo,
            "per_fold": per_fold, "pooled": {**_at(actual, scores, thr), **_reg(targets, scores)}, "ci": ci,
            "boots": boots, "sens": sens, "calib": calib, "scores": scores, "targets": targets, "actual": actual,
            "auc": area_under(r["fpr"], r["tpr"])}


def plot_epochs(out: Path, meta: dict, folds: list[dict]) -> list[Path]:
    epochs = folds[0]["records"]["epoch"]
    written = []
    for key, label in EPOCH_CHARTS:
        fig = go.Figure()
        color = COLORS[key]
        val = np.array([f["records"][f"val_{key}"] for f in folds], dtype=float)
        train = np.array([f["records"][f"train_{key}"] for f in folds], dtype=float)
        m, s = val.mean(axis=0), val.std(axis=0)
        fig.add_trace(go.Scatter(x=epochs, y=m + s, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=epochs, y=m - s, mode="lines", line={"width": 0}, fill="tonexty",
                                 fillcolor=_rgba(color, 0.2), name="± desvío (folds)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=epochs, y=m, mode="lines", name="Validación (media)", line={"color": color, "width": 2}))
        fig.add_trace(go.Scatter(x=epochs, y=train.mean(axis=0), mode="lines", name="Train (media)",
                                 line={"color": color, "width": 2, "dash": "dot"}))
        _save(fig, out / f"epoch_{key}.html",
              f"{label} por época (primeras {len(epochs)} épocas; media de {len(folds)} folds, umbral {meta['threshold']}, "
              f"corte {meta['fraud_cutoff']})", "Época", label, unit_y=key != "loss")
        written.append(out / f"epoch_{key}.html")
    return written


def plot_thresholds(out: Path, A: dict) -> list[Path]:
    g, r = A["grid"], A["rates"]

    def marks(fig):
        fig.add_vline(x=A["threshold"], line_dash="dash", line_color="#444", annotation_text=f"recomendado {A['threshold']:.2f}")
        if abs(A["config_threshold"] - A["threshold"]) > 1e-9:
            fig.add_vline(x=A["config_threshold"], line_dash="dot", line_color="#bd4b45",
                          annotation_text=f"config {A['config_threshold']}", annotation_position="bottom right")

    fig = go.Figure()
    for m in ("accuracy", "precision", "recall", "f1"):
        fig.add_trace(go.Scatter(x=g, y=r[m], mode="lines", name=m, line={"color": COLORS[m], "width": 2}))
    marks(fig)
    _save(fig, out / "thr_metrics.html", "Accuracy / Precision / Recall / F1 vs umbral (predicciones fuera de muestra)",
          "Umbral", "Valor", unit_y=True)

    fig = go.Figure()
    for m in ("tpr", "fpr"):
        fig.add_trace(go.Scatter(x=g, y=r[m], mode="lines", name=m.upper(), line={"color": COLORS[m], "width": 2}))
    marks(fig)
    _save(fig, out / "thr_rates.html", "TPR y FPR vs umbral", "Umbral", "Tasa", unit_y=True)

    hover = [f"umbral {t:.2f}" for t in g]
    fig = go.Figure([go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines+markers", name="ROC", text=hover,
                                marker={"size": 4}, line={"color": COLORS["tpr"]}),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line={"dash": "dash", "color": "#999"})])
    _save(fig, out / "thr_roc.html", f"Curva ROC (AUC≈{A['auc']:.3f})", "FPR", "TPR", hover="closest")
    base = A["positives"] / A["n"]
    fig = go.Figure([go.Scatter(x=r["recall"], y=r["precision"], mode="lines+markers", name="Precision-Recall", text=hover,
                                marker={"size": 4}, line={"color": COLORS["precision"]})])
    fig.add_hline(y=base, line_dash="dash", line_color="#999", annotation_text=f"azar = {base:.3f}")
    _save(fig, out / "thr_pr.html", "Curva Precision-Recall", "Recall", "Precision", hover="closest")

    fig = go.Figure()
    for ratio, c in A["cost"].items():
        i = int(np.argmin(c))
        fig.add_trace(go.Scatter(x=g, y=c, mode="lines", name=f"FN:FP = {ratio:g}:1"))
        fig.add_trace(go.Scatter(x=[g[i]], y=[c[i]], mode="markers", marker={"symbol": "star", "size": 12},
                                 showlegend=False, hovertemplate=f"mínimo @ {g[i]:.2f}<extra></extra>"))
    marks(fig)
    _save(fig, out / "thr_cost.html", "Costo esperado por 1000 transacciones vs umbral (unidad = costo de un falso positivo)",
          "Umbral", "Costo por 1000 transacciones")

    c = A["counts"]
    fig = go.Figure(go.Scatter(x=g, y=100 * (c["tp"] + c["fp"]) / A["n"], mode="lines", line={"color": "#4c78a8", "width": 2}))
    marks(fig)
    _save(fig, out / "thr_alerts.html", "Transacciones marcadas como fraude vs umbral (carga de revisión)",
          "Umbral", "% de transacciones marcadas")
    return [out / f"thr_{n}.html" for n in ("metrics", "rates", "roc", "pr", "cost", "alerts")]


def plot_final(out: Path, A: dict) -> list[Path]:
    thr = A["threshold"]
    tp, fp, fn, tn = (float(np.sum(A["actual"] & (A["scores"] >= thr))), float(np.sum(~A["actual"] & (A["scores"] >= thr))),
                      float(np.sum(A["actual"] & (A["scores"] < thr))), float(np.sum(~A["actual"] & (A["scores"] < thr))))
    z = np.array([[tn, fp], [fn, tp]])
    pct = z / z.sum(axis=1, keepdims=True)
    fig = go.Figure(go.Heatmap(z=pct, x=["Predicho: no fraude", "Predicho: fraude"], y=["Real: no fraude", "Real: fraude"],
                               colorscale="Blues", zmin=0, zmax=1, showscale=False,
                               text=[[f"{int(z[i, j]):,}<br>{100 * pct[i, j]:.1f}% de la fila" for j in range(2)] for i in range(2)],
                               texttemplate="%{text}"))
    fig.update_yaxes(autorange="reversed")
    _save(fig, out / "final_confusion_matrix.html", f"Matriz de confusión al umbral {thr:.2f} (fuera de muestra, {A['rule']})",
          height=450, hover="closest")

    pf = A["per_fold"]
    ms = lambda k: f"{np.mean([f[k] for f in pf]):.4f} ± {np.std([f[k] for f in pf]):.4f}"
    ci = lambda k: f"[{A['ci'][k][0]:.4f}, {A['ci'][k][1]:.4f}]" if k in A["ci"] else "—"
    rows = [("MAE (prob.)", "MAE"), ("RMSE (prob.)", "RMSE"), ("R² (prob.)", "R2"), ("Precision", "precision"),
            ("Recall (TPR)", "recall"), ("F1", "f1"), ("Accuracy", "accuracy"), ("FPR", "fpr")]
    fig = go.Figure(go.Table(
        header={"values": ["Métrica", "Global (fuera de muestra)", "Media ± desvío entre folds", "IC 95% bootstrap"],
                "fill_color": "#eaeaea", "align": "left"},
        cells={"values": [[a for a, _ in rows], [f"{A['pooled'][k]:.4f}" for _, k in rows],
                          [ms(k) for _, k in rows], [ci(k) for _, k in rows]], "align": "left"}))
    _save(fig, out / "final_metrics_table.html", f"Métricas finales (umbral {thr:.2f}, corte de fraude {A['cutoff']})", height=420)

    idx = np.sort(np.random.default_rng(0).choice(A["n"], min(3000, A["n"]), replace=False))
    fig = go.Figure([go.Scatter(x=A["targets"][idx], y=A["scores"][idx], mode="markers", opacity=0.45,
                                marker={"size": 4, "color": NONLINEAR_C}, name="Muestras"),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#444"}, name="Predicción perfecta")])
    fig.add_vline(x=A["cutoff"], line_dash="dot", line_color="#999")
    fig.add_hline(y=thr, line_dash="dot", line_color="#999")
    _save(fig, out / "final_pred_vs_actual.html", "TinyModel vs BigModel (fuera de muestra; líneas = corte real y umbral)",
          "Probabilidad BigModel (real)", "Predicción TinyModel", hover="closest")
    return [out / f"final_{n}.html" for n in ("confusion_matrix", "metrics_table", "pred_vs_actual")]


def plot_robust(out: Path, A: dict, folds: list[dict]) -> list[Path]:
    k = len(folds)
    labels = [f"Fold {i + 1}" for i in range(k)]
    fig = go.Figure()
    for m in ("precision", "recall", "f1"):
        vals = [f[m] for f in A["per_fold"]]
        fig.add_trace(go.Scatter(x=[m] * k, y=vals, mode="markers", marker={"size": 10, "color": COLORS[m], "opacity": 0.6},
                                 name=m, text=labels, hovertemplate="%{text}: %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[m], y=[np.mean(vals)], mode="markers", marker={"size": 16, "symbol": "diamond", "color": "#222"},
                                 error_y={"type": "data", "array": [np.std(vals)], "visible": True}, showlegend=False,
                                 hovertemplate=f"media {np.mean(vals):.4f} ± {np.std(vals):.4f}<extra></extra>"))
    _save(fig, out / "robust_fold_metrics.html", f"Precision / Recall / F1 por fold al umbral {A['threshold']:.2f} (rombo = media ± desvío)",
          None, "Valor", unit_y=True, hover="closest")

    fig = go.Figure([go.Bar(x=labels, y=A["own_thr"], name="Mejor F1 con ese fold", marker_color="#4c78a8"),
                     go.Scatter(x=labels, y=A["lofo_thr"], mode="markers", name="Mejor F1 sin ese fold",
                                marker={"symbol": "diamond", "size": 12, "color": "#f58518"})])
    fig.add_hline(y=A["threshold"], line_dash="dash", line_color="#444", annotation_text=f"recomendado {A['threshold']:.2f}")
    _save(fig, out / "robust_fold_thresholds.html", "¿Es estable el umbral? Mejor umbral por F1 en cada fold",
          None, "Umbral", unit_y=True, hover="closest")

    if A["boots"]["f1"]:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Precision", "Recall", "F1"), horizontal_spacing=0.06)
        for col, m in enumerate(("precision", "recall", "f1"), start=1):
            fig.add_trace(go.Histogram(x=A["boots"][m], nbinsx=40, marker_color=COLORS[m], showlegend=False), row=1, col=col)
            for edge in A["ci"][m]:
                fig.add_vline(x=edge, line_dash="dash", line_color="#444", row=1, col=col)
            fig.update_xaxes(title_text=f"IC95% [{A['ci'][m][0]:.3f}, {A['ci'][m][1]:.3f}]", row=1, col=col)
        _save(fig, out / "robust_bootstrap.html", f"Bootstrap ({len(A['boots']['f1'])} remuestreos) al umbral {A['threshold']:.2f}",
              height=420, hover="closest")

    s = A["sens"]
    if s:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        xs = [f"{d['cutoff']:g}" for d in s]
        fig.add_trace(go.Scatter(x=xs, y=[d["best_f1"] for d in s], mode="lines+markers", name="Mejor F1",
                                 text=[f"mejor umbral {d['best_threshold']:.2f}" for d in s], line={"color": COLORS["f1"]}), secondary_y=False)
        fig.add_trace(go.Scatter(x=xs, y=[d["f1_at_recommended"] for d in s], mode="lines+markers", name="F1 al umbral recomendado",
                                 line={"color": COLORS["f1"], "dash": "dot"}), secondary_y=False)
        fig.add_trace(go.Scatter(x=xs, y=[d["auc"] for d in s], mode="lines+markers", name="AUC ROC",
                                 line={"color": COLORS["accuracy"]}), secondary_y=False)
        fig.add_trace(go.Bar(x=xs, y=[100 * d["positive_rate"] for d in s], name="% positivos", opacity=0.25,
                             marker_color="#888"), secondary_y=True)
        fig.update_yaxes(title_text="F1 / AUC", range=[0, 1.02], secondary_y=False)
        fig.update_yaxes(title_text="% de positivos", secondary_y=True)
        _save(fig, out / "robust_cutoff_sensitivity.html", "Sensibilidad al corte que define 'fraude real'",
              "Corte sobre la probabilidad de BigModel")

    cal = A["calib"]
    fig = go.Figure([go.Scatter(x=[c["mean_pred"] for c in cal], y=[c["mean_actual"] for c in cal], mode="lines+markers",
                                marker={"size": [min(40, 8 + 30 * c["count"] / A["n"] * 3) for c in cal]}, name="TinyModel",
                                text=[f"{c['count']:,} muestras" for c in cal]),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#999"}, name="Calibración perfecta")])
    _save(fig, out / "robust_calibration.html", "Calibración respecto de BigModel (10 intervalos de predicción)",
          "Probabilidad predicha (media del intervalo)", "Probabilidad BigModel (media del intervalo)", hover="closest")
    return [out / f"robust_{n}.html" for n in ("fold_metrics", "fold_thresholds", "bootstrap", "cutoff_sensitivity", "calibration")]


# ------------------------------------------------------------------------------ entry
def write_ej1_plots(out_dir: str | Path, recommend: str = "f1", cost_ratios=(1, 5, 10, 50), bootstrap: int = 1000,
                    cutoffs=(0.3, 0.4, 0.5, 0.6, 0.7)) -> dict[str, Any]:
    out = Path(out_dir)
    written: list[Path] = []
    summary: dict[str, Any] = {}
    if (out / "design.json").exists():
        written += plot_design(out, json.loads((out / "design.json").read_text()))
    if (out / "compare.json").exists():
        written += plot_compare(out, json.loads((out / "compare.json").read_text()))
    if (out / "study.json").exists():
        meta, folds = _load_study(out)
        A = analyze(meta, folds, recommend, cost_ratios, bootstrap, cutoffs)
        written += plot_epochs(out, meta, folds) + plot_thresholds(out, A) + plot_final(out, A) + plot_robust(out, A, folds)
        summary = {"threshold": A["threshold"], "rule": A["rule"], "threshold_best_f1": A["thr_f1"],
                   "threshold_by_cost_ratio": {f"{k:g}": v for k, v in A["thr_cost"].items()},
                   "fraud_cutoff": A["cutoff"], "positives": A["positives"], "n": A["n"], "auc": A["auc"],
                   "pooled": A["pooled"], "ci95": A["ci"], "per_fold": A["per_fold"],
                   "fold_best_thresholds": A["own_thr"], "fold_thresholds_without_fold": A["lofo_thr"],
                   "cutoff_sensitivity": A["sens"]}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        p = A["pooled"]
        print(f"\nUmbral recomendado: {A['threshold']:.2f} ({A['rule']}); mejor F1 en {A['thr_f1']:.2f}; "
              f"por costo: { {f'{k:g}:1': round(v, 2) for k, v in A['thr_cost'].items()} }")
        print(f"Global: precision={p['precision']:.4f} recall={p['recall']:.4f} F1={p['f1']:.4f} "
              f"| MAE={p['MAE']:.4f} RMSE={p['RMSE']:.4f} R2={p['R2']:.4f} | AUC≈{A['auc']:.4f}")
        if A["ci"]:
            print("IC95% bootstrap:", {k: [round(x, 4) for x in v] for k, v in A["ci"].items()})
    print("Charts written:", ", ".join(w.name for w in written))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild exercise 1 charts from the stored results")
    ap.add_argument("--out-dir", default="metrics/ej1")
    ap.add_argument("--recommend", default="f1", help="'f1', 'cost:<FN/FP ratio>' or a number")
    ap.add_argument("--cost-ratios", type=float, nargs="+", default=[1, 5, 10, 50])
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    write_ej1_plots(a.out_dir, a.recommend, a.cost_ratios, a.bootstrap)