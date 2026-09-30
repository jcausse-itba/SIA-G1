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
        rows=1,
        cols=2,
        subplot_titles=("Training loss", "Held-out predictions"),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Scatter(
            x=np.arange(1, len(loss_history) + 1),
            y=loss_history,
            mode="lines",
            name="Training MSE",
            line={"color": "#278276", "width": 2},
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
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=sample_indices,
                y=predicted[sample_indices],
                mode="markers",
                name="Predicted class",
            ),
            row=1,
            col=2,
        )
        figure.update_xaxes(title_text="Test sample", row=1, col=2)
        figure.update_yaxes(title_text="Class index", row=1, col=2)
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
            row=1,
            col=2,
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
            row=1,
            col=2,
        )
        figure.update_xaxes(title_text="Actual value", row=1, col=2)
        figure.update_yaxes(title_text="Predicted value", row=1, col=2)

    figure.update_xaxes(title_text="Epoch", row=1, col=1)
    figure.update_yaxes(title_text="Mean squared error", row=1, col=1)
    figure.update_layout(
        title_text="MLP Training Results",
        template="plotly_white",
        height=520,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )