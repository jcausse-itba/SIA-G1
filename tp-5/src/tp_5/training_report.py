from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def write_training_report(
    loss_history: Sequence[float],
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str | Path,
    is_classification: bool = False,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Learning curve: training loss by epoch", "Held-out predictions"),
        vertical_spacing=0.18,
    )
    figure.add_trace(
        go.Scatter(
            x=np.arange(len(loss_history)),
            y=loss_history,
            mode="lines+markers",
            name="Training MSE",
            line={"color": "#278276", "width": 2},
            marker={"size": 5},
            hovertemplate="Epoch %{x}<br>Training MSE %{y:.6g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if is_classification:
        sample_indices = np.linspace(0, len(actual) - 1, min(len(actual), 1500), dtype=int)
        figure.add_trace(
            go.Scatter(
                x=sample_indices,
                y=actual[sample_indices],
                mode="markers",
                name="Actual class",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=sample_indices,
                y=predicted[sample_indices],
                mode="markers",
                name="Predicted class",
            ),
            row=2,
            col=1,
        )
        figure.update_xaxes(title_text="Test sample", row=2, col=1)
        figure.update_yaxes(title_text="Class index", row=2, col=1)
    else:
        figure.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode="markers",
                opacity=0.55,
                marker_color="#278276",
                name="Held-out predictions",
            ),
            row=2,
            col=1,
        )
        lower = float(min(actual.min(), predicted.min()))
        upper = float(max(actual.max(), predicted.max()))
        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                name="Perfect prediction",
                line={"dash": "dash", "color": "#bd4b45"},
            ),
            row=2,
            col=1,
        )
        figure.update_xaxes(title_text="Actual value", row=2, col=1)
        figure.update_yaxes(title_text="Predicted value", row=2, col=1)

    figure.update_xaxes(title_text="Epoch", row=1, col=1)
    figure.update_yaxes(title_text="Mean squared error", row=1, col=1)
    figure.update_layout(
        title_text="MLP Training Results",
        template="plotly_white",
        height=850,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )