import contextlib
from html import escape
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tp_3.engine.activation_functions.identity import Identity
from tp_3.engine.activation_functions.standard import Tanh
from tp_3.engine.loss_functions.mse import MeanSquaredError
from tp_3.engine.models.mlp import MLP
from tp_3.engine.optimizers.sgd import SGD


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data and documentation" / "fraud_dataset.csv"
REPORT_PATH = ROOT / "fraud_dataset_ranges.html"
MODEL_PATH = ROOT / "data and documentation" / "fraud_probability_model.pkl"
TARGET = "big_model_fraud_probability"
HIDDEN_NEURONS = 16
EPOCHS = 1500
LEARNING_RATE = 0.01


def fit_model(inputs: np.ndarray, targets: np.ndarray) -> MLP:
    np.random.seed(42)
    model = MLP(
        [inputs.shape[1], HIDDEN_NEURONS, 1],
        Tanh(),
        Identity(),
        MeanSquaredError(),
        SGD,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(
            inputs,
            targets.reshape(-1, 1),
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            print_every=EPOCHS + 1,
        )
    return model


def display_value(value, column: str) -> str:
    if column == "timestamp":
        return pd.to_datetime(value, unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    if isinstance(value, (int, float)):
        return f"{value:,.4f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def main() -> None:
    frame = pd.read_csv(
        DATA_PATH,
        usecols=lambda column: column != "flagged_fraud",
    )
    feature_columns = [column for column in frame.columns if column != TARGET]
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * 0.8)
    train, test = ordered.iloc[:split_index], ordered.iloc[split_index:]
    train_features = train[feature_columns].to_numpy(dtype=np.float64)
    test_features = test[feature_columns].to_numpy(dtype=np.float64)
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    train_features = (train_features - feature_mean) / feature_scale
    test_features = (test_features - feature_mean) / feature_scale

    model = fit_model(train_features, train[TARGET].to_numpy())
    actual = test[TARGET].to_numpy()
    predicted = np.clip(model.forward(test_features).ravel(), 0.0, 1.0)
    residuals = predicted - actual
    metrics = {
        "MAE": float(np.mean(np.abs(residuals))),
        "RMSE": float(np.sqrt(np.mean(residuals**2))),
        "R2": float(1 - np.sum(residuals**2) / np.sum((actual - actual.mean()) ** 2)),
    }
    mean_baseline = float(np.sqrt(np.mean((train[TARGET].mean() - actual) ** 2)))
    correlations = pd.DataFrame(
        {
            "Pearson": frame[feature_columns].corrwith(frame[TARGET], method="pearson"),
            "Spearman": frame[feature_columns].corrwith(frame[TARGET], method="spearman"),
        }
    )
    correlations = correlations.reindex(
        correlations["Spearman"].abs().sort_values(ascending=True).index
    )

    full_features = frame[feature_columns].to_numpy(dtype=np.float64)
    full_feature_mean = full_features.mean(axis=0)
    full_feature_scale = full_features.std(axis=0)
    full_feature_scale[full_feature_scale == 0] = 1.0
    full_features = (full_features - full_feature_mean) / full_feature_scale
    model = fit_model(full_features, frame[TARGET].to_numpy())
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "feature_mean": full_feature_mean,
                "feature_scale": full_feature_scale,
            },
            model_file,
        )

    charts = []
    summaries = []

    for index, column in enumerate(frame.columns):
        values = frame[column]
        plot_values = values
        if column == "timestamp":
            plot_values = pd.to_datetime(values, unit="s", utc=True).dt.tz_localize(None)

        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Value distribution", "Range and outliers"),
            horizontal_spacing=0.12,
        )
        figure.add_trace(
            go.Histogram(x=plot_values, nbinsx=40, marker_color="#278276", showlegend=False),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Box(x=plot_values, boxpoints="outliers", marker_color="#bd4b45", showlegend=False),
            row=1,
            col=2,
        )
        figure.update_layout(
            title=column,
            template="plotly_white",
            height=330,
            margin={"l": 45, "r": 25, "t": 70, "b": 45},
            bargap=0.06,
        )
        figure.update_yaxes(title_text="Rows", row=1, col=1)
        figure.update_xaxes(title_text="Value", row=1, col=1)
        figure.update_xaxes(title_text="Value", row=1, col=2)
        charts.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=(index == 0),
                config={"responsive": True, "displaylogo": False},
            )
        )

        quantiles = values.quantile([0, 0.25, 0.5, 0.75, 1])
        labels = ["Min", "25%", "Median", "75%", "Max"]
        summary_cells = "".join(
            f"<td>{escape(display_value(value, column))}</td>"
            for value in quantiles.to_list()
        )
        summaries.append(
            f"<tr><th>{escape(column)}</th>{summary_cells}<td>{values.nunique():,}</td></tr>"
        )

    correlation_figure = go.Figure()
    correlation_figure.add_trace(
        go.Bar(
            x=correlations["Pearson"],
            y=correlations.index,
            orientation="h",
            name="Pearson (linear)",
            marker_color="#278276",
        )
    )
    correlation_figure.add_trace(
        go.Bar(
            x=correlations["Spearman"],
            y=correlations.index,
            orientation="h",
            name="Spearman (rank-based)",
            marker_color="#d28a3d",
        )
    )
    correlation_figure.update_layout(
        title="Correlation of each predictor with the target score",
        xaxis_title="Correlation coefficient",
        yaxis_title=None,
        barmode="group",
        template="plotly_white",
        height=450,
    )
    correlation_chart = correlation_figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )
    correlation_rows = "".join(
        f"<tr><th>{escape(column)}</th><td>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td></tr>"
        for column, row in correlations.iterrows()
    )

    prediction_figure = go.Figure(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            opacity=0.55,
            marker_color="#278276",
            name="Held-out transactions",
        )
    )
    prediction_figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect prediction",
            line={"dash": "dash", "color": "#bd4b45"},
        )
    )
    prediction_figure.update_layout(
        title="Held-out score predictions vs. actual values",
        xaxis_title="Actual probability",
        yaxis_title="Predicted probability",
        template="plotly_white",
        height=430,
    )
    prediction_chart = prediction_figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )
    metric_rows = "".join(
        f"<tr><th>{name}</th><td>{value:.4f}</td></tr>"
        for name, value in metrics.items()
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fraud Dataset Analysis</title>
  <style>
    :root {{ color-scheme: light; --ink: #202825; --muted: #5f6c67; --line: #d9e1dd; --accent: #176c5c; }}
    body {{ margin: 0; color: var(--ink); background: #f5f7f5; font: 16px/1.5 Georgia, serif; }}
    main {{ max-width: 1160px; margin: 0 auto; padding: 32px 22px 60px; }}
    h1, h2 {{ font-family: Segoe UI, sans-serif; line-height: 1.2; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    h2 {{ margin-top: 36px; font-size: 21px; }}
    p {{ color: var(--muted); }}
    .summary {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; background: white; font: 14px Segoe UI, sans-serif; }}
    th, td {{ text-align: right; padding: 9px 11px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--accent); }}
    .chart {{ margin: 18px 0 28px; background: white; }}
    @media (max-width: 640px) {{ main {{ padding: 22px 12px 40px; }} h1 {{ font-size: 25px; }} }}
  </style>
</head>
<body><main>
    <h1>Fraud Dataset Analysis</h1>
    <p>{len(frame):,} rows. Each available column has a histogram and box plot; the table summarizes minimum, quartiles, median, maximum, and unique values.</p>
    <h2>Probability prediction</h2>
    <p>Extra Trees regression predicts <code>{TARGET}</code>. The latest 20% of transactions by timestamp is held out for evaluation; the earliest 80% is used for training. Predictions use the other available columns.</p>
    <div class="summary"><table><thead><tr><th>Metric</th><th>Held-out score</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
    <p>Always predicting the training mean gives an RMSE of {mean_baseline:.4f}. The fitted model is saved at <code>{MODEL_PATH.relative_to(ROOT).as_posix()}</code>.</p>
  <div class="summary"><table>
    <thead><tr><th>Column</th>{''.join(f'<th>{label}</th>' for label in labels)}<th>Unique</th></tr></thead>
    <tbody>{''.join(summaries)}</tbody>
  </table></div>
  {''.join(f'<section class="chart">{chart}</section>' for chart in charts)}
        <h2>Correlations with target</h2>
        <p>Pearson measures linear association; Spearman measures rank-based monotonic association. Correlation alone does not establish causation or capture every nonlinear interaction.</p>
        <section class="chart">{correlation_chart}</section>
        <div class="summary"><table>
            <thead><tr><th>Predictor</th><th>Pearson</th><th>Spearman</th></tr></thead>
            <tbody>{correlation_rows}</tbody>
        </table></div>
    <h2>Held-out predictions</h2>
    <section class="chart">{prediction_chart}</section>
</main></body>
</html>"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Analyzed {len(frame.columns)} columns across {len(frame):,} rows.")
    print("Held-out metrics:", ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
    print(f"Mean baseline RMSE: {mean_baseline:.4f}")
    print("Correlations with target (Pearson, Spearman):")
    print(
        correlations.sort_values(
            "Spearman",
            key=lambda values: values.abs(),
            ascending=False,
        ).to_string(float_format=lambda value: f"{value:.4f}")
    )
    print(f"Report written to: {REPORT_PATH}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()