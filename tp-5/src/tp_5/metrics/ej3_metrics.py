"""Generates plot HTML files for Exercise 3 slides.

Outputs:
1. ej3_comparison.html      - Bar chart comparing Ej2 vs Ej3 Accuracy & Target (>98%).
2. ej3_noise_robustness.html- Accuracy vs Noise Level curve.
3. ej3_confusion_matrix.html- Confusion matrix on test set.
4. ej3_receptive_fields.html- First layer weights visualization (receptive fields).
5. ej3_saliency_maps.html   - Saliency maps for model interpretability.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _save(fig: go.Figure, path: Path, title: str, height: int = 520) -> None:
    fig.update_layout(title_text=title, template="plotly_white", height=height)
    fig.write_html(path, include_plotlyjs="directory", config={"responsive": True, "displaylogo": False})


def generate_ej3_plots(ej2_dir: Path, ej3_dir: Path) -> list[Path]:
    written = []
    ej3_file = ej3_dir / "ej3_results.json"
    if not ej3_file.exists():
        print(f"[!] {ej3_file} not found. Run ej3_sweep.py first.")
        return written

    data3 = json.loads(ej3_file.read_text())
    labels = data3["labels"]

    # 1. Comparison Plot (Ej2 vs Ej3)
    ej2_best_file = ej2_dir / "best.json"
    if ej2_best_file.exists():
        data2 = json.loads(ej2_best_file.read_text())
        ej2_acc = data2.get("accuracy", {}).get("test", data2.get("accuracy", {}).get("val", 0.0))
    else:
        ej2_acc = data3.get("ej2_metrics", {}).get("test", {}).get("accuracy", 0.0)

    ej3_acc = data3["ej3_metrics"]["test"]["accuracy"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Ex 2 (Base)", "Ex 3 (More Data)"], y=[ej2_acc, ej3_acc],
                         marker_color=["#4c78a8", "#54a24b"], text=[f"{ej2_acc:.4f}", f"{ej3_acc:.4f}"],
                         textposition="auto"))
    fig.add_hline(y=0.98, line_dash="dash", line_color="#bd4b45",
                  annotation_text="Target Goal (98%)", annotation_position="top left")
    fig.update_yaxes(range=[0.8, 1.0], title_text="Test Accuracy")
    _save(fig, ej3_dir / "ej3_comparison.html", "Performance Improvement: Exercise 2 vs Exercise 3")
    written.append(ej3_dir / "ej3_comparison.html")

    # 2. Robustness to Noise
    noise_data = data3["noise_robustness"]
    sigmas = [d["sigma"] for d in noise_data]
    accs = [d["accuracy"] for d in noise_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sigmas, y=accs, mode="lines+markers", line=dict(color="#f58518", width=3),
                             marker=dict(size=8)))
    fig.update_xaxes(title_text="Gaussian Noise Level (σ)")
    fig.update_yaxes(title_text="Test Accuracy", range=[0, 1.02])
    _save(fig, ej3_dir / "ej3_noise_robustness.html", "Test Accuracy under Gaussian Noise")
    written.append(ej3_dir / "ej3_noise_robustness.html")

    # 3. Confusion Matrix
    cm = np.array(data3["confusion"])
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", text=cm.astype(str),
                               texttemplate="%{text}",
                               colorbar=dict(title=dict(text="Samples", side="top"))))
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="True", autorange="reversed")
    _save(fig, ej3_dir / "ej3_confusion_matrix.html", f"Exercise 3 Confusion Matrix (Acc: {ej3_acc:.4f})")
    written.append(ej3_dir / "ej3_confusion_matrix.html")

    # 4. Receptive Fields (First Layer Weights)
    weights = np.array(data3["first_layer_weights"])  # (n_inputs, n_hidden)
    side = int(np.sqrt(weights.shape[0]))
    n_neurons = min(16, weights.shape[1])

    rows = int(np.ceil(n_neurons / 4))
    fig = make_subplots(rows=rows, cols=4, subplot_titles=[f"Neuron {i+1}" for i in range(n_neurons)])
    
    w_min, w_max = weights.min(), weights.max()

    for idx in range(n_neurons):
        r, c = idx // 4 + 1, idx % 4 + 1
        w_img = weights[:, idx].reshape(side, side)
        
        show_scale = (idx == n_neurons - 1)
        
        fig.add_trace(
            go.Heatmap(
                z=w_img,
                colorscale="Viridis",
                zmin=w_min,
                zmax=w_max,
                showscale=show_scale,
                colorbar=dict(
                    title=dict(
                        text="Weight Value (W)<br><sub>Yellow: Positive (Excitatory)<br>Purple: Negative (Inhibitory)</sub>",
                        side="top",
                    ),
                    len=0.8,
                ) if show_scale else None,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=r, col=c)

    _save(fig, ej3_dir / "ej3_receptive_fields.html", "Receptive Fields (1st Hidden Layer Weights)", height=680)
    written.append(ej3_dir / "ej3_receptive_fields.html")

    # 5. Saliency Maps
    saliency_samples = data3.get("saliency_samples", [])
    if saliency_samples:
        n_samples = len(saliency_samples)
        fig = make_subplots(
            rows=2,
            cols=n_samples,
            subplot_titles=[f"True: {s['true']} | Pred: {s['pred']}" for s in saliency_samples] + [""] * n_samples,
        )

        for i, sample in enumerate(saliency_samples):
            px = np.array(sample["pixels"]).reshape(side, side)
            sal = np.array(sample["saliency"]).reshape(side, side)

            show_scale_px = (i == n_samples - 1)
            show_scale_sal = (i == n_samples - 1)

            # Row 1: Original Image
            fig.add_trace(
                go.Heatmap(
                    z=px,
                    colorscale="gray",
                    showscale=show_scale_px,
                    colorbar=dict(
                        title=dict(
                            text="Pixel Intensity<br><sub>White: 1.0 | Black: 0.0</sub>",
                            side="top",
                        ),
                        len=0.4,
                        y=0.8,
                    ) if show_scale_px else None,
                ),
                row=1,
                col=i + 1,
            )

            # Row 2: Saliency Map (|dOutput/dX|)
            fig.add_trace(
                go.Heatmap(
                    z=sal,
                    colorscale="Hot",
                    showscale=show_scale_sal,
                    colorbar=dict(
                        title=dict(
                            text="Sensitivity (|∂y/∂x|)<br><sub>White/Yellow: High influence<br>Red/Black: Low influence</sub>",
                            side="top",
                        ),
                        len=0.4,
                        y=0.2,
                    ) if show_scale_sal else None,
                ),
                row=2,
                col=i + 1,
            )

            fig.update_xaxes(showticklabels=False, row=1, col=i + 1)
            fig.update_yaxes(showticklabels=False, autorange="reversed", row=1, col=i + 1)
            fig.update_xaxes(showticklabels=False, row=2, col=i + 1)
            fig.update_yaxes(showticklabels=False, autorange="reversed", row=2, col=i + 1)

        _save(fig, ej3_dir / "ej3_saliency_maps.html", "Saliency Maps: Pixel Sensitivity Gradient Magnitude", height=580)
        written.append(ej3_dir / "ej3_saliency_maps.html")

    print(f"Generated Ej3 plots with English legends: {', '.join(p.name for p in written)}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise 3 Plot Generator")
    parser.add_argument("--ej2-dir", default="metrics/ej2")
    parser.add_argument("--ej3-dir", default="metrics/ej3")
    args = parser.parse_args()
    generate_ej3_plots(Path(args.ej2_dir), Path(args.ej3_dir))