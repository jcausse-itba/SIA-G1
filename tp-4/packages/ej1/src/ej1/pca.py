"""Punto 1: Análisis de Componentes Principales (implementación propia vía eigh)."""
from dataclasses import dataclass
from typing import Any, Dict, Hashable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import Dataset
from .plots import DIVERGING, out_dir, save_csv, save_fig, short


@dataclass
class PCAResult:
    eigenvalues: np.ndarray      # todos, en orden decreciente
    loadings: pd.DataFrame       # features x PCk (autovectores, signo normalizado)
    scores: pd.DataFrame         # creadores x PCk

    @property
    def explained(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()


def fit_pca(ds: Dataset, n_components: int | None = None) -> PCAResult:
    X = ds.X
    cov = np.cov(X, rowvar=False)            # == matriz de correlación (X estandarizada)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]

    # Convención de signo: el loading de mayor |valor| de cada componente es positivo.
    # Hace la interpretación estable y permite comparar directamente con Oja.
    signs = np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(vecs.shape[1])])
    vecs = vecs * signs

    q = n_components or len(vals)
    cols = [f"PC{i + 1}" for i in range(q)]
    loadings = pd.DataFrame(vecs[:, :q], index=ds.features, columns=cols)
    scores = pd.DataFrame(X @ vecs[:, :q], columns=cols)
    return PCAResult(eigenvalues=vals, loadings=loadings, scores=scores)


def biplot(pca: PCAResult, ds: Dataset, color=None, color_label: str = "", title: str = "Biplot PC1 vs PC2"):
    sc = pca.scores.copy()
    sc["name"] = ds.names
    if color is not None:
        sc[color_label] = color
    ev = pca.explained
    fig = px.scatter(sc, x="PC1", y="PC2", text="name", color=color_label if color is not None else None,
                     title=f"{title}  (PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%})", height=850)
    fig.update_traces(textposition="top center", textfont_size=9, marker_size=9)

    # Escala de flechas: usar |.| (el max sin abs falla si los loadings dominantes son negativos).
    L = pca.loadings[["PC1", "PC2"]]
    scale = 0.9 * sc[["PC1", "PC2"]].abs().to_numpy().max() / L.abs().to_numpy().max()
    for feat, (lx, ly) in L.iterrows():
        feat_str = str(feat) if not isinstance(feat, str) else feat
        fig.add_annotation(x=lx * scale, y=ly * scale, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor="#c0392b",
                           text="", opacity=0.8)
        fig.add_annotation(x=lx * scale * 1.08, y=ly * scale * 1.08, text=short(feat_str), showarrow=False,
                           font=dict(color="#c0392b", size=10))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def run_pca(ds: Dataset, cfg: Dict[str, Any]) -> PCAResult:
    out = out_dir(cfg, "pca")
    pca = fit_pca(ds, cfg["n_components"])
    full = fit_pca(ds, None)  # scree siempre con todas
    ev, cum = full.explained, np.cumsum(full.explained)

    # ---- Scree + acumulada + criterio de Kaiser
    idx = [f"PC{i + 1}" for i in range(len(ev))]
    var_df = pd.DataFrame({"componente": idx, "autovalor": full.eigenvalues,
                           "varianza_explicada": ev, "acumulada": cum})
    save_csv(var_df.round(5), out, "varianza_explicada")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=idx, y=ev, name="Varianza explicada")
    fig.add_scatter(x=idx, y=cum, name="Acumulada", mode="lines+markers", secondary_y=True)
    fig.add_hline(y=1 / len(ev), line_dash="dot", annotation_text="Kaiser (λ=1)")
    fig.update_layout(title="Scree plot")
    fig.update_yaxes(tickformat=".0%", secondary_y=False)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], secondary_y=True)
    save_fig(fig, out, "scree")

    # ---- Loadings (primeras componentes interpretables)
    k_show = min(6, pca.loadings.shape[1])
    L = pca.loadings.iloc[:, :k_show]
    save_csv(pca.loadings.round(4), out, "loadings", index=True)
    fig = px.imshow(L, text_auto=True, color_continuous_scale=DIVERGING, zmin=-0.6, zmax=0.6,
                    aspect="auto", title="Loadings de las primeras componentes", height=800)
    save_fig(fig, out, "loadings")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loadings PC1", "Loadings PC2"))
    for j, pc in enumerate(("PC1", "PC2"), start=1):
        s = pca.loadings[pc].sort_values()
        s_vals = s.to_numpy()
        fig.add_bar(x=s_vals, y=s.index, orientation="h", row=1, col=j, showlegend=False,
                    marker_color=np.where(s_vals >= 0, "#1e8449", "#c0392b"))
    fig.update_layout(height=750)
    save_fig(fig, out, "loadings_pc1_pc2")

    # ---- Biplot (coloreado por temática dominante) + boxplot + índice PC1
    dom = ds.raw.filter(like="topic_").idxmax(axis=1).astype(str).str.replace("topic_", "", regex=False).str.replace("_pct", "", regex=False)
    save_fig(biplot(full, ds, color=dom.to_numpy(), color_label="tematica"), out, "biplot")

    k_box = min(10, full.scores.shape[1])
    fig = px.box(full.scores.iloc[:, :k_box].melt(var_name="componente", value_name="score"),
                 x="componente", y="score", points="all", title="Boxplot de las componentes principales")
    save_fig(fig, out, "boxplot_componentes")

    idx_df = pd.DataFrame({"name": ds.names, "PC1": full.scores["PC1"]}).sort_values("PC1", ascending=False)
    fig = px.bar(idx_df, x="name", y="PC1", title="Índice PC1 (creadores ordenados)")
    fig.update_xaxes(tickangle=-60)
    save_fig(fig, out, "indice_pc1")

    scores = pd.concat([ds.ids, pca.scores], axis=1)
    save_csv(scores.round(5), out, "scores")

    n90 = int(np.searchsorted(cum, 0.90) + 1)
    kaiser = int((full.eigenvalues > 1).sum())
    print(f"[pca] {len(ev)} componentes. PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}, PC1+PC2 {cum[1]:.1%}.")
    print(f"[pca] Kaiser (λ>1): {kaiser} componentes; 90% de varianza: {n90} componentes.")
    near_zero = int((full.eigenvalues < 1e-8).sum())
    if near_zero:
        print(f"[pca] {near_zero} autovalores ~0 (restricciones lineales exactas entre features, "
              f"p.ej. grupos topic_*/content_* que suman 100).")
    for pc in ("PC1", "PC2"):
        s = full.loadings[pc]
        top = s.reindex(s.abs().sort_values(ascending=False).index)[:5]
        print(f"[pca] {pc}: " + ", ".join(f"{k}={v:+.2f}" for k, v in top.items()))
    print(f"[pca] Salidas en {out}/")
    return full