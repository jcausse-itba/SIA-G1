"""Punto 4: Selección de creadores por campaña y objetivo vs estrategia por followers.

Los scores son promedios ponderados de z-scores de variables crudas del CSV (no de las features
del modelo): acá sí interesan `estimated_argentina_reach` y el costo, que fueron excluidas del
clustering por ser derivadas. Los pesos son supuestos explícitos (el enunciado lo permite) y están
centralizados en OBJECTIVES / CAMPAIGN_BONUS para poder discutirlos y modificarlos.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import Dataset
from .plots import out_dir, save_csv, save_fig

# Variables con cola derecha: se usa log antes del z-score (mismo criterio que en el modelo).
LOG_VARS = {"followers", "average_reach", "estimated_argentina_reach", "price_usd", "arg_reach_per_usd"}

# Objetivo -> {variable: peso}. Pesos negativos penalizan.
OBJECTIVES: Dict[str, Dict[str, float]] = {
    # Alcance masivo: muchas cuentas argentinas, y que sea eficiente en costo.
    "reach": {"estimated_argentina_reach": 0.5, "arg_reach_per_usd": 0.3, "argentina_audience_pct": 0.2},
    # Descubrimiento: llegar a gente que no sigue al creador y que esa gente actúe (visitas, shares).
    "discovery": {"non_follower_reach_pct": 0.3, "shares_per_1000": 0.2, "profile_visits_per_1000": 0.2,
                  "trend_posts_pct": 0.1, "short_form_posts_pct": 0.1, "argentina_audience_pct": 0.1},
    # Profundidad educativa: contenido que se guarda, se comenta y se mira entero.
    "depth": {"saves_per_1000": 0.25, "content_tutorial_pct": 0.2, "average_watch_time_seconds": 0.2,
              "comments_per_1000": 0.15, "average_caption_words": 0.1, "argentina_audience_pct": 0.1},
}
# Ajustes por cliente: el refugio busca consultas/adopciones (conversación + visitas al perfil);
# el resto busca ventas (reseñas/showcase de producto ayudan a convertir).
CAMPAIGN_BONUS: Dict[str, Dict[str, float]] = {
    "dogs": {"profile_visits_per_1000": 0.15, "comments_per_1000": 0.15},
    "football": {"content_review_pct": 0.1, "content_showcase_pct": 0.1},
    "makeup": {"content_review_pct": 0.1, "content_showcase_pct": 0.1},
    "art": {"content_review_pct": 0.1, "content_showcase_pct": 0.1},
}
# Penalización suave por saturación publicitaria (audiencia "cansada" de publis).
SPONSORED_PENALTY = 0.1


def _scoring_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["arg_reach_per_usd"] = df["estimated_argentina_reach"] / df["price_usd"]
    return df


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.Series(np.log1p(df[col]), index=df.index) if col in LOG_VARS else df[col]
    return (s - s.mean()) / s.std()


def objective_score(df: pd.DataFrame, campaign: str, objective: str) -> pd.Series:
    weights = dict(OBJECTIVES[objective])
    for k, v in CAMPAIGN_BONUS.get(campaign, {}).items():
        weights[k] = weights.get(k, 0) + v
    weights["sponsored_posts_pct"] = weights.get("sponsored_posts_pct", 0) - SPONSORED_PENALTY
    total = sum(abs(v) for v in weights.values())
    
    res = pd.Series(0.0, index=df.index)
    for c, w in weights.items():
        res = res + w * _z(df, c)
    return res / total


def selection_metrics(df: pd.DataFrame, idx: pd.Index, topic_col: str, followers_idx: pd.Index) -> Dict[str, float]:
    s = df.loc[idx]
    return {
        "n": len(s),
        "costo_total_usd": s["price_usd"].sum(),
        "alcance_arg_total": s["estimated_argentina_reach"].sum(),
        "alcance_arg_por_usd": s["estimated_argentina_reach"].sum() / s["price_usd"].sum(),
        "alcance_arg_en_tematica": (s["estimated_argentina_reach"] * s[topic_col] / 100).sum(),
        "alcance_tematico_por_usd": (s["estimated_argentina_reach"] * s[topic_col] / 100).sum() / s["price_usd"].sum(),
        "tematica_pct_media": s[topic_col].mean(),
        "followers_medio": s["followers"].mean(),
        "engagement_medio": s["total_engagement_per_1000"].mean(),
        "saves_medio": s["saves_per_1000"].mean(),
        "non_follower_reach_medio": s["non_follower_reach_pct"].mean(),
        "profile_visits_medio": s["profile_visits_per_1000"].mean(),
        "solapamiento_con_followers": len(set(idx) & set(followers_idx)),
    }


def run_campaign(ds: Dataset, cfg: Dict[str, Any], campaign: str,
                 assign: pd.DataFrame | None = None) -> pd.DataFrame:
    out = out_dir(cfg, f"campaign/{campaign}")
    df = _scoring_frame(ds.raw)
    topic_col = f"topic_{campaign}_pct"
    n = cfg["top_n"]
    objectives: List[str] = list(OBJECTIVES) if cfg["objective"] == "all" else [cfg["objective"]]

    eligible = df[topic_col] >= cfg["min_topic_pct"]
    if eligible.sum() < n:
        print(f"[campaign:{campaign}] Sólo {eligible.sum()} creadores con {topic_col} >= {cfg['min_topic_pct']}.")

    # Estrategia actual de la agencia: top-N por followers, sin mirar temática.
    followers_idx = df["followers"].nlargest(n).index
    selections = {"followers": followers_idx}

    scores = ds.ids.copy()
    scores[topic_col] = df[topic_col]
    scores["elegible"] = eligible
    for obj in objectives:
        sc = objective_score(df, campaign, obj)
        scores[f"score_{obj}"] = sc
        selections[obj] = sc[eligible].nlargest(n).index
    if assign is not None:
        scores["region"] = assign["region"].to_numpy()
        scores["neuron"] = assign["neuron"].to_numpy()
    save_csv(scores.round(4), out, "scores")

    # ---- Selecciones (quién) y métricas comparativas (qué gana / pierde cada estrategia)
    sel_rows = []
    for strat, idx in selections.items():
        for rank, i in enumerate(idx, start=1):
            row = {"estrategia": strat, "rank": rank, "name": df.at[i, "name"],
                   "followers": df.at[i, "followers"], topic_col: df.at[i, topic_col],
                   "estimated_argentina_reach": df.at[i, "estimated_argentina_reach"],
                   "price_usd": df.at[i, "price_usd"]}
            if assign is not None:
                row["region"] = assign.at[i, "region"]
            sel_rows.append(row)
    sel_df = pd.DataFrame(sel_rows)
    save_csv(sel_df.round(2), out, "selecciones")

    comp = pd.DataFrame({s: selection_metrics(df, idx, topic_col, followers_idx)
                         for s, idx in selections.items()})
    save_csv(comp.round(3), out, "comparacion", index=True)

    # ---- Gráfico: todos los creadores en (log followers, alcance temático), marcando selecciones
    plot = df[["name", "followers", topic_col, "estimated_argentina_reach", "price_usd"]].copy()
    plot["alcance_arg_tematico"] = plot["estimated_argentina_reach"] * plot[topic_col] / 100
    plot["estrategia"] = "—"
    for strat, idx in selections.items():
        plot.loc[idx, "estrategia"] = np.where(plot.loc[idx, "estrategia"] == "—", strat,
                                               plot.loc[idx, "estrategia"] + "+" + strat)
    # Mostrar el nombre sólo si pertenecen a alguna estrategia seleccionada
    plot["label_name"] = np.where(plot["estrategia"] != "—", plot["name"], "")

    fig = px.scatter(
        plot, 
        x="followers", 
        y="alcance_arg_tematico", 
        color="estrategia", 
        size="price_usd",
        hover_name="name", 
        text="label_name",    # <--- Solo muestra el nombre de los elegidos por estrategia
        log_x=True, 
        log_y=True, 
        size_max=28,
        title=f"Campaña {campaign}: followers vs alcance argentino en la temática "
              f"(tamaño = precio USD)"
    )
    
    fig.update_traces(textposition='top center', textfont=dict(size=10))
    save_fig(fig, out, "selecciones")

    norm = comp.drop(["n", "solapamiento_con_followers"]).T
    norm = norm / norm.loc["followers"]  # relativo a la estrategia por followers (=1)
    fig = go.Figure()
    for strat in norm.index:
        fig.add_bar(name=strat, x=norm.columns, y=norm.loc[strat])
    fig.add_hline(y=1, line_dash="dot")
    fig.update_layout(barmode="group", title=f"Campaña {campaign}: métricas relativas a la estrategia por followers",
                      yaxis_type="log", height=550)
    save_fig(fig, out, "comparacion")

    print(f"[campaign:{campaign}] {int(eligible.sum())} elegibles ({topic_col} >= {cfg['min_topic_pct']}), top {n}:")
    for strat, idx in selections.items():
        c = comp[strat]
        print(f"[campaign:{campaign}]   {strat:<9} {', '.join(df.loc[idx, 'name'])}")
        print(f"[campaign:{campaign}]   {'':<9} costo ${c.costo_total_usd:,.0f} | alcance temático/USD "
              f"{c.alcance_tematico_por_usd:.1f} | temática {c.tematica_pct_media:.0f}% | "
              f"solapa con followers {int(c.solapamiento_con_followers)}/{n}")
    print(f"[campaign:{campaign}] Salidas en {out}/")
    return comp
