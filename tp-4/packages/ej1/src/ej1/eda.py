"""Punto 1: Exploratory Data Analysis y justificación de la selección de variables."""
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


def _clustered_corr(corr: pd.DataFrame) -> pd.DataFrame:
    # Distancia 1-|r| para que variables muy (anti)correlacionadas queden juntas.
    dist = np.clip(1 - corr.abs().to_numpy(), 0, None)
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    return cast(pd.DataFrame, corr.iloc[order, order])


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

    # ---- Correlación (todas las numéricas, agrupada) y pares altamente correlacionados
    corr = num.corr()
    cc = _clustered_corr(corr)
    fig = px.imshow(cc, text_auto=".2f", color_continuous_scale=DIVERGING, zmin=-1, zmax=1,  # type: ignore[arg-type]
                    title="Matriz de correlación agrupada (todas las variables)", width=1100, height=1100)
    fig.update_traces(textfont_size=7)
    save_fig(fig, out, "correlacion_agrupada")

    pairs = [
        (a, b, float(val))  # type: ignore[arg-type]
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1:]
        for val in [corr.loc[a, b]]
        if abs(float(val)) >= HIGH_CORR  # type: ignore[arg-type]
    ]
    pairs_df = pd.DataFrame(pairs, columns=["var_a", "var_b", "r"])
    pairs_df = pairs_df.reindex(pairs_df["r"].abs().sort_values(ascending=False).index)
    save_csv(pairs_df.round(4), out, "pares_alta_correlacion")

    # ---- Matriz de correlación de las features finales (después de log1p)
    fcorr = _clustered_corr(ds.transformed.corr())
    fig = px.imshow(fcorr, text_auto=".2f", color_continuous_scale=DIVERGING, zmin=-1, zmax=1,  # type: ignore[arg-type]
                    title="Correlación de las features seleccionadas (con log1p)", width=950, height=950)
    fig.update_traces(textfont_size=8)
    save_fig(fig, out, "correlacion_features")

    # ---- Asimetría antes / después de log1p
    logs = [c for c in cfg["log_features"] if c in num.columns]
    sk = pd.DataFrame({"original": num[logs].skew(),
                       "log1p": num[logs].apply(np.log1p).skew()}).reset_index(names="variable")
    fig = px.bar(sk.melt(id_vars="variable", var_name="escala", value_name="skew"),
                 x="variable", y="skew", color="escala", barmode="group",
                 title="Asimetría de conteos y precio: original vs log1p")
    save_fig(fig, out, "asimetria_log")

    # ---- Boxplots: escala original estandarizada vs features finales
    z_raw = (num - num.mean()) / num.std()
    fig = px.box(z_raw.melt(var_name="variable", value_name="z"), x="variable", y="z",
                 points="outliers", title="Boxplots de todas las variables (estandarizadas, sin log)")
    fig.update_layout(height=600)
    save_fig(fig, out, "boxplots_original")

    z_feat = pd.DataFrame(ds.X, columns=ds.features)
    fig = px.box(z_feat.melt(var_name="variable", value_name="z"), x="variable", y="z",
                 points="outliers", hover_data=None,
                 title="Boxplots de las features finales (log1p + estandarización)")
    fig.update_layout(height=600)
    save_fig(fig, out, "boxplots_features")

    # ---- Perfil por temática dominante (para conectar con los clientes)
    topics = raw.filter(like="topic_")
    dom = topics.idxmax(axis=1).astype(str).str.replace("topic_", "").str.replace("_pct", "")
    prof = pd.concat([dom.rename("tematica_dominante"), num], axis=1).groupby("tematica_dominante").mean()
    prof.insert(0, "n", dom.value_counts())
    save_csv(prof.T.round(2), out, "perfil_por_tematica", index=True)

    print(f"[eda] {len(raw)} creadores, {num.shape[1]} variables numéricas -> "
          f"{len(ds.features)} features seleccionadas ({len(ds.log_features)} con log1p).")
    for k, v in checks.items():
        print(f"[eda]   {k}: {v}")
    print(f"[eda] Excluidas: {sorted(set(num.columns) - set(ds.features))}")
    print(f"[eda] Salidas en {out}/")