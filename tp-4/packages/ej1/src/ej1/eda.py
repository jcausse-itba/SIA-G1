"""Punto 1: Exploratory Data Analysis y justificación de la selección de variables (sin log1p)."""
import json
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .cli import ID_COLUMNS
from .data import Dataset
from .plots import DIVERGING, out_dir, save_csv, save_fig

HIGH_CORR = 0.85


def _derived_checks(raw: pd.DataFrame) -> Dict[str, Any]:
    """Verifica numéricamente las relaciones que declara el enunciado."""
    eng = raw[["likes_per_1000", "comments_per_1000", "saves_per_1000", "shares_per_1000"]].sum(axis=1)
    est = raw["average_reach"] * raw["argentina_audience_pct"] / 100
    topic = raw.filter(like="topic_").sum(axis=1)
    content = raw.filter(like="content_").sum(axis=1)
    return {
        "price_ars / price_usd (min, max)": [float(raw.price_ars.div(raw.price_usd).min()),
                                            float(raw.price_ars.div(raw.price_usd).max())],
        "max |total_engagement - sum(likes,comments,saves,shares)|":
            float((eng - raw.total_engagement_per_1000).abs().max()),
        "organic + sponsored (min, max)": [float((raw.organic_posts_pct + raw.sponsored_posts_pct).min()),
                                           float((raw.organic_posts_pct + raw.sponsored_posts_pct).max())],
        "max relative error estimated_argentina_reach vs reach*pct/100":
            float(((est - raw.estimated_argentina_reach).abs() / raw.estimated_argentina_reach).max()),
        "sum topic_* (min, max)": [float(topic.min()), float(topic.max())],
        "sum content_* (min, max)": [float(content.min()), float(content.max())],
    }


def run_eda(ds: Dataset, cfg: Dict[str, Any]) -> None:
    out = out_dir(cfg, "eda")
    raw = ds.raw
    num = raw.drop(columns=list(ID_COLUMNS)).select_dtypes("number")

    # ---- Descriptivos
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc["cv"] = desc["std"] / desc["mean"]
    save_csv(desc.round(4), out, "descriptivos", index=True)

    # ---- Relaciones derivadas / composicionales
    checks = _derived_checks(raw)
    with open(f"{out}/derived_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    # ---- Matriz de Covarianza
    cov_matrix = num.cov()
    fig_cov = px.imshow(cov_matrix, title="Matriz de Covarianza", text_auto=True,
                        color_continuous_scale=['red', 'white', 'green'], color_continuous_midpoint=0)
    save_fig(fig_cov, out, "covarianza")

    # ---- Matriz de Correlación Estándar
    corr_matrix = num.corr()
    fig_corr = px.imshow(corr_matrix, title="Matriz de Correlación", text_auto=True,
                         color_continuous_scale=['red', 'white', 'green'], color_continuous_midpoint=0)
    save_fig(fig_corr, out, "correlacion")

    # ---- Matriz de Correlación Agrupada (Ordenada con Ward)
    order = leaves_list(linkage(corr_matrix, method='ward'))
    corr_matrix_clustered = corr_matrix.iloc[order, order]
    fig_corr_clust = px.imshow(corr_matrix_clustered, title="Matriz de Correlación Agrupada", text_auto=True,
                               color_continuous_scale=['red', 'white', 'green'], color_continuous_midpoint=0,
                               width=775, height=775)
    save_fig(fig_corr_clust, out, "correlacion_agrupada")

    # ---- Pares altamente correlacionados
    pairs = [
        (a, b, float(cast(Any, val)))
        for i, a in enumerate(corr_matrix.columns)
        for b in corr_matrix.columns[i + 1:]
        for val in [corr_matrix.loc[a, b]]
        if abs(float(cast(Any, val))) >= HIGH_CORR
    ]
    pairs_df = pd.DataFrame(pairs, columns=["var_a", "var_b", "r"])
    pairs_df = pairs_df.reindex(pairs_df["r"].abs().sort_values(ascending=False).index)
    save_csv(pairs_df.round(4), out, "pares_alta_correlacion")

    # ---- Matriz de correlación de las features seleccionadas
    # (Si deseas mantener el clustering por distancia personalizada en las features seleccionadas)
    dist = np.clip(1 - ds.transformed.corr().abs().to_numpy(), 0, None)
    np.fill_diagonal(dist, 0)
    fcorr_order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    fcorr = ds.transformed.corr().iloc[fcorr_order, fcorr_order]
    
    fig = px.imshow(fcorr, text_auto=cast(Any, ".2f"), color_continuous_scale=DIVERGING, zmin=-1, zmax=1,
                    title="Correlación de las features seleccionadas (estandarizadas)", width=950, height=950)
    fig.update_traces(textfont_size=8)
    save_fig(fig, out, "correlacion_features")

    # ---- Boxplots: escala original estandarizada
    z_raw = (num - num.mean()) / num.std()
    fig = px.box(z_raw.melt(var_name="variable", value_name="z"), x="variable", y="z",
                 points="outliers", title="Boxplots de todas las variables (estandarizadas)")
    fig.update_layout(height=600)
    save_fig(fig, out, "boxplots_original")

    # ---- Perfil por temática dominante
    topics = raw.filter(like="topic_")
    dom = topics.idxmax(axis=1).astype(str).str.replace("topic_", "").str.replace("_pct", "")
    prof = pd.concat([dom.rename("tematica_dominante"), num], axis=1).groupby("tematica_dominante").mean()
    prof.insert(0, "n", dom.value_counts())
    save_csv(prof.T.round(2), out, "perfil_por_tematica", index=True)

    print(f"[eda] {len(raw)} creadores, {num.shape[1]} variables numéricas -> "
          f"{len(ds.features)} features seleccionadas.")
    for k, v in checks.items():
        print(f"[eda]   {k}: {v}")
    print(f"[eda] Salidas en {out}/")