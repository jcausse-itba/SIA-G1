import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ACC, LOSS = "#4c78a8", "#f58518"
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#bd4b45", "#278276", "#9d755d", "#7f7f7f"]


def _stats(values: list) -> tuple[float | None, float]:
    vals = [v for v in values if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, 0.0)


def _groups(runs: list[dict], experiment: str) -> list[tuple[Any, list[dict]]]:
    """Group experiment runs by x (sorted if x is numeric, insertion order otherwise)."""
    grouped: dict[Any, list[dict]] = {}
    for run in runs:
        if run["experiment"] == experiment:
            grouped.setdefault(run["x"], []).append(run)
    keys = list(grouped)
    if keys and all(isinstance(k, (int, float)) for k in keys):
        keys.sort()
    return [(k, grouped[k]) for k in keys]


def _summary(group: list[dict], key: str) -> tuple[float | None, float]:
    return _stats([r[key] for r in group])


def _save(fig: go.Figure, path: Path, title: str, height: int = 560) -> None:
    fig.update_layout(
        title_text=title,
        template="plotly_white",
        height=height,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    fig.write_html(path, include_plotlyjs="directory", config={"responsive": True, "displaylogo": False})


def _dual_axis(
    xs: list,
    groups: list[list[dict]],
    xtitle: str,
    path: Path,
    title: str,
    log_x: bool = False,
    bars: bool = False,
    hover: list[str] | None = None,
) -> None:
    """Accuracy on left axis, loss on right axis; dashed for training, solid for validation."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    traces = (("val", "Validation", None), ("train", "Training", "dash"))
    for split, name, dash in traces:
        for metric, label, color, secondary in (
            ("acc", "Accuracy", ACC, False),
            ("loss", "Loss", LOSS, True),
        ):
            stats = [_summary(g, f"final_{split}_{metric}") for g in groups]
            ys, errs = [s[0] for s in stats], [s[1] for s in stats]
            common = dict(
                x=xs,
                y=ys,
                name=f"{label} ({name})",
                text=hover,
                error_y={"type": "data", "array": errs, "visible": any(e > 0 for e in errs)},
            )
            if bars and not secondary:
                trace = go.Bar(**common, marker_color=color, opacity=1.0 if split == "val" else 0.45)
            else:
                trace = go.Scatter(
                    **common,
                    mode="lines+markers" if not bars else "markers",
                    line={"color": color, "dash": dash, "width": 2},
                    marker={"color": color, "size": 9, "symbol": "diamond" if bars else "circle"},
                )
            fig.add_trace(trace, secondary_y=secondary)
    fig.update_xaxes(title_text=xtitle, type="log" if log_x else None)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.02], secondary_y=False)
    fig.update_yaxes(title_text="Loss", rangemode="tozero", secondary_y=True)
    _save(fig, path, title)


def _split_metrics_plot(
    xs: list,
    groups: list[list[dict]],
    xtitle: str,
    path: Path,
    title: str,
    hover: list[str] | None = None,
) -> None:
    """Genera dos gráficos separados en subplots para Accuracy y Loss."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Accuracy vs Optimizer", "Loss vs Optimizer"),
        horizontal_spacing=0.12,
    )

    traces = (("val", "Validation", 1.0), ("train", "Training", 0.5))

    for split, name, opacity in traces:
        # Subplot Accuracy
        stats_acc = [_summary(g, f"final_{split}_acc") for g in groups]
        ys_acc, errs_acc = [s[0] for s in stats_acc], [s[1] for s in stats_acc]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys_acc,
                name=f"Accuracy ({name})",
                marker_color=ACC,
                opacity=opacity,
                error_y={"type": "data", "array": errs_acc, "visible": any(e > 0 for e in errs_acc)},
                text=hover,
            ),
            row=1,
            col=1,
        )

        # Subplot Loss
        stats_loss = [_summary(g, f"final_{split}_loss") for g in groups]
        ys_loss, errs_loss = [s[0] for s in stats_loss], [s[1] for s in stats_loss]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys_loss,
                name=f"Loss ({name})",
                marker_color=LOSS,
                opacity=opacity,
                error_y={"type": "data", "array": errs_loss, "visible": any(e > 0 for e in errs_loss)},
                text=hover,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text=xtitle, row=1, col=1)
    fig.update_xaxes(title_text=xtitle, row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(title_text="Loss", rangemode="tozero", row=1, col=2)

    fig.update_layout(barmode="group")
    _save(fig, path, title)


def _curves(groups: list[tuple[Any, list[dict]]], path: Path, title: str) -> None:
    """Learning curves (first seed for each configuration): loss and accuracy per epoch."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Loss per epoch", "Accuracy per epoch"),
        horizontal_spacing=0.08,
    )
    for i, (_, group) in enumerate(groups):
        run, color = group[0], PALETTE[i % len(PALETTE)]
        epochs = [r["epoch"] for r in run["history"]]
        for split, dash in (("val", None), ("train", "dot")):
            for col, metric in ((1, "loss"), (2, "acc")):
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=[r[f"{split}_{metric}"] for r in run["history"]],
                        mode="lines",
                        name=f"{run['label']}" + (" (training)" if split == "train" else ""),
                        legendgroup=run["label"],
                        showlegend=(col == 1),
                        line={"color": color, "dash": dash, "width": 2},
                    ),
                    row=1,
                    col=col,
                )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.02], row=1, col=2)
    _save(fig, path, title, height=520)


def write_ej2_plots(out_dir: str | Path, results: dict | None = None) -> list[Path]:
    """Genera los gráficos del ejercicio 2.

    Los resultados pueden pasarse directamente con `results` (dict con claves
    "runs" y "meta"); si es None, se leen de `<out_dir>/results.json`.
    """
    out = Path(out_dir)
    stored = results if results is not None else json.loads((out / "results.json").read_text())
    runs, meta = stored["runs"], stored["meta"]
    note = f"{meta.get('epochs')} epochs, batch {meta.get('batch_size')}, {meta.get('seeds')} seed(s)"
    written: list[Path] = []

    groups = _groups(runs, "lr")
    if groups:
        _dual_axis(
            [k for k, _ in groups],
            [g for _, g in groups],
            "Learning rate",
            out / "lr_sweep.html",
            f"Accuracy and Loss vs Learning Rate ({meta.get('base_optimizer')}; {note})",
            log_x=True,
        )
        _curves(groups, out / "lr_curves.html", "Learning curves by learning rate")
        written += [out / "lr_sweep.html", out / "lr_curves.html"]

    groups = _groups(runs, "optimizer")
    if groups:
        _split_metrics_plot(
            [k for k, _ in groups],
            [g for _, g in groups],
            "Optimizer",
            out / "optimizer_sweep.html",
            f"Accuracy and Loss vs Optimizer (lr={meta.get('base_lr')}; {note})",
        )
        _curves(groups, out / "optimizer_curves.html", "Learning curves by optimizer")
        written += [out / "optimizer_sweep.html", out / "optimizer_curves.html"]

    groups = _groups(runs, "activation")
    if groups:
        _dual_axis(
            [k for k, _ in groups],
            [g for _, g in groups],
            "Activation",
            out / "activation_sweep.html",
            f"Accuracy (bars) and Loss (diamonds) vs Activation ({meta.get('base_optimizer')}, "
            f"lr={meta.get('base_lr')}; {note})",
            bars=True,
        )
        _curves(groups, out / "activation_curves.html", "Learning curves by activation")
        written += [out / "activation_sweep.html", out / "activation_curves.html"]

    groups = _groups(runs, "depth")
    if groups:
        hover = [f"{g[0]['architecture']} ({g[0]['n_params']:,} parameters)" for _, g in groups]
        _dual_axis(
            [k for k, _ in groups],
            [g for _, g in groups],
            "Hidden Layers (N)",
            out / "depth_sweep.html",
            f"Accuracy and Loss vs Depth (width: {meta.get('width_strategy')}, "
            f"{meta.get('base_optimizer')}, lr={meta.get('base_lr')}; {note})",
            hover=hover,
        )
        _curves(groups, out / "depth_curves.html", "Learning curves by depth")
        written += [out / "depth_sweep.html", out / "depth_curves.html"]

    groups = _groups(runs, "grid")
    if groups or any(r["experiment"] == "grid" for r in runs):
        grid = [r for r in runs if r["experiment"] == "grid"]
        optimizers = list(dict.fromkeys(r["optimizer"] for r in grid))
        lrs = sorted({r["lr"] for r in grid})
        z = [
            [
                _stats([r["final_val_acc"] for r in grid if r["optimizer"] == o and r["lr"] == lr])[0]
                for lr in lrs
            ]
            for o in optimizers
        ]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=[f"{lr:g}" for lr in lrs],
                y=optimizers,
                zmin=0,
                zmax=1,
                colorscale="Viridis",
                text=[[f"{v:.3f}" if v is not None else "" for v in row] for row in z],
                texttemplate="%{text}",
                colorbar={"title": "Val. Accuracy"},
            )
        )
        fig.update_xaxes(title_text="Learning rate", type="category")
        fig.update_yaxes(title_text="Optimizer")
        _save(
            fig,
            out / "lr_optimizer_grid.html",
            f"Validation Accuracy: learning rate × optimizer ({note})",
        )
        written.append(out / "lr_optimizer_grid.html")

    if runs:  # Confusion matrix for overall best run
        best = max(runs, key=lambda r: r["final_val_acc"] if r["final_val_acc"] is not None else -1)
        labels, cm = meta["labels"], np.array(best["confusion"])
        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm.astype(str),
                texttemplate="%{text}",
                colorbar={"title": "Samples"},
            )
        )
        fig.update_xaxes(title_text="Predicted", type="category")
        fig.update_yaxes(title_text="Actual", type="category", autorange="reversed")
        _save(
            fig,
            out / "confusion_matrix.html",
            f"Confusion Matrix (validation) — best run: {best['label']} "
            f"[{best['experiment']}], accuracy={best['final_val_acc']:.4f}",
            height=650,
        )
        written.append(out / "confusion_matrix.html")

    print("Generated plots:", ", ".join(p.name for p in written))
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct Exercise 2 plots from results.json")
    parser.add_argument("--out-dir", default="metrics/ej2", help="Metrics output directory")
    write_ej2_plots(parser.parse_args().out_dir)