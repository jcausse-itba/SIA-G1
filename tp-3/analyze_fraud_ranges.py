from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "fraud_dataset.csv"
REPORT_PATH = ROOT / "fraud_dataset_ranges.html"
TARGET = "big_model_fraud_probability"
QUANTILE_LABELS = ["Min", "25%", "Median", "75%", "Max"]


def display_value(value, column: str) -> str:
    if column == "timestamp":
        return pd.to_datetime(value, unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    if isinstance(value, (int, float)):
        return f"{value:,.4f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def main() -> None:
    frame = pd.read_csv(DATA_PATH, usecols=lambda column: column != "flagged_fraud")
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {DATA_PATH}.")
    feature_columns = [column for column in frame.columns if column != TARGET]
    correlations = pd.DataFrame(
        {
            "Pearson": frame[feature_columns].corrwith(frame[TARGET], method="pearson"),
            "Spearman": frame[feature_columns].corrwith(frame[TARGET], method="spearman"),
        }
    )
    correlations = correlations.reindex(
        correlations["Spearman"].abs().sort_values(ascending=True).index
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
        f"<tr><th>{escape(str(column))}</th><td>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td></tr>"
        for column, row in correlations.iterrows()
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fraud Dataset Statistics and Correlations</title>
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
    <h1>Fraud Dataset Statistics and Correlations</h1>
    <p>{len(frame):,} rows. Each available column has a histogram and box plot; the table summarizes minimum, quartiles, median, maximum, and unique values.</p>
  <div class="summary"><table>
    <thead><tr><th>Column</th>{''.join(f'<th>{label}</th>' for label in QUANTILE_LABELS)}<th>Unique</th></tr></thead>
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
</main></body>
</html>"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Analyzed {len(frame.columns)} columns across {len(frame):,} rows.")
    print("Correlations with target (Pearson, Spearman):")
    print(
        correlations.sort_values(
            "Spearman",
            key=lambda values: values.abs(),
            ascending=False,
        ).to_string(float_format=lambda value: f"{value:.4f}")
    )
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()