"""Exercise 2 charts for the evaluation protocol and the best model (reads protocol.json / best.json).

  protocol_scheme.html        how digits.csv and digits_test.csv are used
  protocol_class_balance.html % of samples per digit in train / validation / test
  protocol_examples.html      example digits
  best_summary_table.html     hyper-parameters + accuracy / loss on train, validation and test
  best_confusion.html         confusion matrix (test = 'production' when available)
  best_per_class.html         precision / recall per digit
  best_misclassified.html     test digits the best model gets wrong (most confident errors first)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

SPLIT_COLORS = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}
SPLIT_NAMES = {"train": "Train", "val": "Validación", "test": "Test (producción)"}


def _save(fig: go.Figure, path: Path, title: str, height: int = 520, **layout) -> None:
    fig.update_layout(title_text=title, template="plotly_white", height=height, **layout)
    fig.write_html(path, include_plotlyjs="directory", config={"responsive": True, "displaylogo": False})


def _mosaic(images: list[np.ndarray], side: int, cols: int, gap_x: int = 3, gap_y: int = 3):
    """Tiles square images into one array (NaN in the gaps). Returns the canvas and each tile's (x0, y0)."""
    rows = int(np.ceil(len(images) / cols))
    canvas = np.full((rows * (side + gap_y) - gap_y, cols * (side + gap_x) - gap_x), np.nan)
    origins = []
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        y0, x0 = r * (side + gap_y), c * (side + gap_x)
        canvas[y0:y0 + side, x0:x0 + side] = np.asarray(img, dtype=float).reshape(side, side)
        origins.append((x0, y0))
    return canvas, origins


def _heat(canvas: np.ndarray) -> go.Heatmap:
    return go.Heatmap(z=canvas, colorscale=[[0, "white"], [1, "black"]], showscale=False, hoverinfo="skip")


def _image_axes(fig: go.Figure) -> None:
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, constrain="domain")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed",
                     scaleanchor="x", scaleratio=1)


# ---------------------------------------------------------------------------- protocol
def plot_protocol(out: Path, p: dict) -> list[Path]:
    sizes, ratio = p["sizes"], p["split_ratio"]
    fig = go.Figure()
    total = sizes["train"] + sizes["val"]
    rows = [(f"{p['dataset']}", [("train", sizes["train"], "Train: ajuste de pesos"),
                                 ("val", sizes["val"], "Validación: elegir hiperparámetros y modelo")])]
    if sizes["test"]:
        rows.append((p["test_dataset"] or "digits_test.csv",
                     [("test", sizes["test"], "Test: una sola evaluación del mejor modelo")]))
    seen = set()
    for name, parts in rows:
        base_total = sum(n for _, n, _ in parts)
        start = 0.0
        for key, n, text in parts:
            width = 100 * n / base_total
            fig.add_trace(go.Bar(y=[name], x=[width], base=[start], orientation="h", name=SPLIT_NAMES[key],
                                 legendgroup=key, showlegend=key not in seen, marker_color=SPLIT_COLORS[key],
                                 text=[f"{text}<br>{n:,} muestras"], textposition="inside", insidetextanchor="middle",
                                 textfont={"color": "white"}, hoverinfo="skip"))
            seen.add(key)
            start += width
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[0, 100], title_text="% del archivo")
    fig.update_yaxes(autorange="reversed")
    _save(fig, out / "protocol_scheme.html",
          f"Protocolo de evaluación (split {ratio:.0%} / {1 - ratio:.0%}; el test no se usa para decidir nada)",
          height=360, margin={"l": 140})

    fig = go.Figure()
    labels = p["labels"]
    for key in ("train", "val", "test"):
        counts = p["counts"].get(key)
        if not counts:
            continue
        counts = np.array(counts)
        fig.add_trace(go.Bar(x=labels, y=100 * counts / counts.sum(), name=SPLIT_NAMES[key], marker_color=SPLIT_COLORS[key],
                             customdata=counts, hovertemplate="Dígito %{x}: %{y:.2f}% (%{customdata:,} muestras)<extra></extra>"))
    fig.add_hline(y=100 / len(labels), line_dash="dash", line_color="#888", annotation_text="reparto uniforme")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Dígito", type="category")
    fig.update_yaxes(title_text="% de muestras del conjunto")
    _save(fig, out / "protocol_class_balance.html", "Balance de clases por conjunto")

    side = p["side"]
    cols = max(len(e) for e in p["examples"])
    # tile order: one column per digit, `cols` examples down each column
    ordered = [p["examples"][c][k] for k in range(cols) for c in range(len(p["examples"])) if k < len(p["examples"][c])]
    canvas, origins = _mosaic(ordered, side, cols=len(p["examples"]))
    fig = go.Figure(_heat(canvas))
    fig.update_xaxes(tickmode="array", showticklabels=True, side="top",
                     tickvals=[o[0] + side / 2 for o in origins[:len(p["examples"])]], ticktext=labels)
    _image_axes(fig)
    fig.update_xaxes(showticklabels=True)
    _save(fig, out / "protocol_examples.html", "Ejemplos de cada dígito (conjunto de entrenamiento)", height=420)
    return [out / f"protocol_{n}.html" for n in ("scheme", "class_balance", "examples")]


# ------------------------------------------------------------------------------- best
def plot_best(out: Path, b: dict) -> list[Path]:
    run, acc, loss, labels = b["run"], b["accuracy"], b["loss"], b["labels"]
    eval_set = b["eval_set"]
    where = SPLIT_NAMES[eval_set]
    fmt = lambda v: "—" if v is None else f"{v:.4f}"
    rows = [("Corrida", f"{run['label']} ({run['experiment']})"),
            ("Arquitectura", " → ".join(map(str, run["architecture"])) + f"  ({run['n_params']:,} parámetros)"),
            ("Activación (oculta / salida)", f"{run['hidden_act']} / {run['output_act']}"),
            ("Optimizador / learning rate", f"{run['optimizer']} / {run['lr']:g}"),
            ("Batch / épocas / pérdida", f"{run['batch_size']} / {run['epochs']} / {run['loss']}"),
            ("Escalado de entradas", run["scaling"])]
    for split in ("train", "val", "test"):
        if split in acc:
            rows.append((f"Accuracy {SPLIT_NAMES[split]}", fmt(acc[split])))
            rows.append((f"Loss {SPLIT_NAMES[split]}", fmt(loss[split])))
    if "test" in acc:
        rows.append(("Brecha train − test", f"{acc['train'] - acc['test']:+.4f}"))
        rows.append(("Brecha validación − test", f"{acc['val'] - acc['test']:+.4f}"))
    fig = go.Figure(go.Table(header={"values": ["", "Mejor modelo"], "fill_color": "#eaeaea", "align": "left"},
                             cells={"values": [[a for a, _ in rows], [v for _, v in rows]], "align": "left"}))
    _save(fig, out / "best_summary_table.html", "Mejor modelo (elegido por accuracy de validación)",
          height=140 + 30 * len(rows))

    cm = np.array(b["confusion"])
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=pct, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1,
                               text=cm.astype(str), texttemplate="%{text}", colorbar={"title": "% de la fila"}))
    fig.update_xaxes(title_text="Predicho", type="category")
    fig.update_yaxes(title_text="Real", type="category", autorange="reversed")
    _save(fig, out / "best_confusion.html",
          f"Matriz de confusión — {where}, accuracy = {acc[eval_set]:.4f}", height=650, hovermode="closest")

    fig = go.Figure([
        go.Scatter(x=labels, y=b["precision"], mode="markers", name="Precision", marker={"size": 12, "color": "#f58518"}),
        go.Scatter(x=labels, y=b["recall"], mode="markers", name="Recall", marker={"size": 12, "symbol": "diamond", "color": "#278276"})])
    lo = min(min(b["precision"]), min(b["recall"]))
    fig.update_yaxes(title_text="Valor", range=[max(0.0, lo - 0.03), 1.005])
    fig.update_xaxes(title_text="Dígito", type="category")
    _save(fig, out / "best_per_class.html", f"Precision y recall por dígito — {where} (eje vertical ampliado)",
          hovermode="x unified")

    written = [out / f"best_{n}.html" for n in ("summary_table", "confusion", "per_class")]
    mis = b["misclassified"]
    if mis:
        side = int(round(np.sqrt(len(mis[0]["pixels"]))))
        cols = min(8, len(mis))
        canvas, origins = _mosaic([m["pixels"] for m in mis], side, cols, gap_x=3, gap_y=9)
        fig = go.Figure(_heat(canvas))
        for m, (x0, y0) in zip(mis, origins):
            fig.add_annotation(x=x0 + side / 2, y=y0 + side + 4.5, text=f"real {m['true']} → pred {m['pred']}",
                               showarrow=False, font={"size": 11, "color": "#bd4b45"})
        _image_axes(fig)
        _save(fig, out / "best_misclassified.html",
              f"Errores del mejor modelo en {where} (los más seguros primero)", height=130 + 140 * int(np.ceil(len(mis) / cols)))
        written.append(out / "best_misclassified.html")
    return written


def write_ej2_extra_plots(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    written: list[Path] = []
    if (out / "protocol.json").exists():
        written += plot_protocol(out, json.loads((out / "protocol.json").read_text()))
    if (out / "best.json").exists():
        written += plot_best(out, json.loads((out / "best.json").read_text()))
    print("Generated plots:", ", ".join(p.name for p in written))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild the protocol / best-model charts of exercise 2")
    ap.add_argument("--out-dir", default="metrics/ej2")
    write_ej2_extra_plots(ap.parse_args().out_dir)