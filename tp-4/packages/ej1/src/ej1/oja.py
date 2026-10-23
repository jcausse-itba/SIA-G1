"""Punto 3: Regla de Oja para obtener la primera componente principal y un índice de creadores."""
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import Dataset
from .pca import PCAResult
from .plots import out_dir, save_csv, save_fig, short


@dataclass
class OjaResult:
    w: np.ndarray            # pesos finales (dirección aprendida)
    norm_hist: np.ndarray    # ||w|| al final de cada época
    cos_hist: np.ndarray     # |cos(w, PC1)| al final de cada época (nan si no hay PC1)
    diverged: bool


def train_oja(X: np.ndarray, lr: float, epochs: int, rng: np.random.Generator,
              pc1: np.ndarray | None = None) -> OjaResult:
    """Regla de Oja muestra a muestra:  y = w·x ;  w <- w + η · y · (x - y·w)."""
    n, d = X.shape
    w = rng.uniform(-1, 1, d)
    w /= np.linalg.norm(w) * 10                  # arranque chico (||w|| = 0.1)
    norms, coss = np.empty(epochs), np.full(epochs, np.nan)
    diverged = False
    for e in range(epochs):
        for i in rng.permutation(n):
            x = X[i]
            y = w @ x
            w = w + lr * y * (x - y * w)
        norms[e] = np.linalg.norm(w)
        if not np.isfinite(norms[e]) or norms[e] > 1e6:
            diverged = True
            norms[e:] = np.nan
            break
        if pc1 is not None:
            coss[e] = abs(w @ pc1) / norms[e]
    return OjaResult(w=w, norm_hist=norms, cos_hist=coss, diverged=diverged)


def run_oja(ds: Dataset, cfg: Dict[str, Any], pca: PCAResult) -> pd.DataFrame:
    out = out_dir(cfg, "oja")
    rng = np.random.default_rng(cfg["seed"])
    pc1 = pca.loadings["PC1"].to_numpy()
    res = train_oja(ds.X, cfg["oja_lr"], cfg["oja_epochs"], rng, pc1)
    if res.diverged:
        raise RuntimeError(f"Oja diverged with --oja-lr {cfg['oja_lr']}; try a smaller learning rate.")

    # Alinear el signo con PC1 (Oja converge a ±v1; el signo depende de la inicialización).
    w = res.w * np.sign(res.w @ pc1)
    index = ds.X @ w
    pc1_score = pca.scores["PC1"].to_numpy()
    corr = float(np.corrcoef(index, pc1_score)[0, 1])
    cos = float(abs(w @ pc1) / np.linalg.norm(w))

    # ---- Tabla de índice
    df = ds.ids.copy()
    df["oja_index"] = index
    df["pc1_score"] = pc1_score
    df["rank_oja"] = df["oja_index"].rank(ascending=False).astype(int)
    df["rank_pc1"] = df["pc1_score"].rank(ascending=False).astype(int)
    df["followers"] = ds.raw["followers"]
    df["rank_followers"] = df["followers"].rank(ascending=False).astype(int)
    df = df.sort_values("oja_index", ascending=False)
    save_csv(df.round(5), out, "indice_oja")

    w_df = pd.DataFrame({"feature": ds.features, "oja_w": w, "pc1_loading": pc1})
    save_csv(w_df.round(5), out, "pesos_vs_pc1")

    # ---- Convergencia: ||w|| -> 1 y |cos(w, PC1)| -> 1 (se agregan lr x10 y x0.1 como referencia)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("‖w‖ por época", "|cos(w, PC1)| por época"))
    palette = px.colors.qualitative.Plotly
    for i, lr in enumerate(sorted({cfg["oja_lr"] / 10, cfg["oja_lr"], cfg["oja_lr"] * 10})):
        r = res if lr == cfg["oja_lr"] else train_oja(ds.X, lr, cfg["oja_epochs"],
                                                     np.random.default_rng(cfg["seed"]), pc1)
        ep = np.arange(1, len(r.norm_hist) + 1)
        name = f"η={lr:g}" + (" (diverge)" if r.diverged else "")
        fig.add_scatter(x=ep, y=r.norm_hist, name=name, legendgroup=name, line_color=palette[i], row=1, col=1)
        fig.add_scatter(x=ep, y=r.cos_hist, name=name, legendgroup=name, showlegend=False,
                        line_color=palette[i], row=1, col=2)
    fig.update_xaxes(type="log", title="época")
    fig.update_layout(title="Convergencia de la regla de Oja", height=480, width=1200)
    save_fig(fig, out, "convergencia")

    # ---- Pesos de Oja vs loadings de PC1
    w_plot = w_df.assign(feature=w_df["feature"].map(short)).sort_values("pc1_loading")
    fig = go.Figure()
    fig.add_bar(y=w_plot["feature"], x=w_plot["pc1_loading"], orientation="h", name="PC1 (eigh)")
    fig.add_bar(y=w_plot["feature"], x=w_plot["oja_w"], orientation="h", name="Oja")
    fig.update_layout(barmode="group", title=f"Pesos de Oja vs loadings de PC1 (|cos| = {cos:.4f})",
                      height=800)
    save_fig(fig, out, "pesos_vs_pc1")

    # ---- Índice vs PC1 e índice vs followers
    fig = px.scatter(df, x="pc1_score", y="oja_index", text="name", hover_data=["rank_oja", "rank_pc1"],
                     title=f"Índice de Oja vs proyección sobre PC1 (r = {corr:.4f})")
    fig.update_traces(textposition="top center", textfont_size=8)
    save_fig(fig, out, "oja_vs_pc1")

    fig = px.bar(df, x="name", y="oja_index", color=np.log10(df["followers"]),
                 color_continuous_scale="Viridis", labels={"color": "log10(followers)"},
                 title="Índice de Oja (color = log10 followers)")
    fig.update_xaxes(tickangle=-60)
    save_fig(fig, out, "indice_oja")

    top_w = w_df.reindex(w_df["oja_w"].abs().sort_values(ascending=False).index).head(6)
    print(f"[oja] η={cfg['oja_lr']}, {cfg['oja_epochs']} épocas: ||w||={np.linalg.norm(w):.4f}, "
          f"|cos(w, PC1)|={cos:.5f}, corr(índice, PC1)={corr:.5f}")
    print("[oja] Pesos dominantes: " + ", ".join(f"{f}={v:+.2f}" for f, v in zip(top_w.feature, top_w.oja_w)))
    print(f"[oja] Top 5: {', '.join(df['name'].head(5))}  |  Bottom 5: {', '.join(df['name'].tail(5))}")
    print(f"[oja] Salidas en {out}/")
    return df
