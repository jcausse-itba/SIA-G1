"""Punto 2: Red de Kohonen (SOM) con vecindad escalón, métricas, Matriz U y barrido."""
import itertools
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import fcluster, linkage

from .data import Dataset
from .plots import SEQUENTIAL, DIVERGING, out_dir, save_csv, save_fig, short


# =========================================================================== modelo
class SOM:
    """Mapa autoorganizado rectangular.

    Regla de actualización (vecindad escalón):
        w_j <- w_j + η(t) · (x - w_j)   si ||pos_j - pos_bmu|| <= R(t)
    con η y R decreciendo linealmente entre sus valores inicial y final.
    """

    def __init__(self, rows: int, cols: int, dim: int, rng: np.random.Generator):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.rng = rng
        # Coordenadas (fila, col) de cada neurona; neurona k = fila*cols + col.
        self.pos = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=float)
        self._grid_dist = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        self.W = np.zeros((rows * cols, dim))

    @property
    def n_neurons(self) -> int:
        return self.rows * self.cols

    def default_radius(self) -> float:
        return max(self.rows, self.cols) / 2

    def init_weights(self, X: np.ndarray, method: str) -> None:
        if method == "sample":
            idx = self.rng.choice(len(X), self.n_neurons, replace=len(X) < self.n_neurons)
            self.W = X[idx].copy()
        else:  # random: uniforme dentro del rango de cada feature
            self.W = self.rng.uniform(X.min(axis=0), X.max(axis=0), size=(self.n_neurons, self.dim))

    def bmu(self, x: np.ndarray) -> int:
        return int(np.argmin(((self.W - x) ** 2).sum(axis=1)))

    def train(self, X: np.ndarray, iterations: int, lr0: float, lr1: float,
              r0: float | None, r1: float) -> None:
        r0 = max(self.default_radius() if r0 is None else r0, r1)
        samples = self.rng.integers(len(X), size=iterations)
        for t, i in enumerate(samples):
            frac = t / max(iterations - 1, 1)
            lr = lr0 + (lr1 - lr0) * frac
            radius = r0 + (r1 - r0) * frac
            x = X[i]
            mask = self._grid_dist[self.bmu(x)] <= radius
            self.W[mask] += lr * (x - self.W[mask])

    # ------------------------------------------------------------------ análisis
    def distances(self, X: np.ndarray) -> np.ndarray:
        """Distancias (n_muestras x n_neuronas) en el espacio de features."""
        return np.sqrt(((X[:, None, :] - self.W[None, :, :]) ** 2).sum(axis=-1))

    def bmus(self, X: np.ndarray) -> np.ndarray:
        return self.distances(X).argmin(axis=1)

    def quantization_error(self, X: np.ndarray) -> float:
        return float(self.distances(X).min(axis=1).mean())

    def topographic_error(self, X: np.ndarray) -> float:
        """Fracción de muestras cuyas 2 BMUs no son vecinas directas (4-vecindad) en la grilla."""
        if self.n_neurons < 2:
            return 0.0
        best2 = np.argsort(self.distances(X), axis=1)[:, :2]
        return float((self._grid_dist[best2[:, 0], best2[:, 1]] > 1.0 + 1e-9).mean())

    def hits(self, X: np.ndarray) -> np.ndarray:
        return np.bincount(self.bmus(X), minlength=self.n_neurons).reshape(self.rows, self.cols)

    def dead_ratio(self, X: np.ndarray) -> float:
        return float((self.hits(X) == 0).mean())

    def u_matrix(self) -> np.ndarray:
        """Distancia euclídea media de cada neurona a sus vecinas de 4-vecindad."""
        U = np.zeros((self.rows, self.cols))
        Wg = self.W.reshape(self.rows, self.cols, self.dim)
        for r in range(self.rows):
            for c in range(self.cols):
                nb = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                      if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]
                U[r, c] = np.mean([np.linalg.norm(Wg[r, c] - Wg[a, b]) for a, b in nb]) if nb else 0.0
        return U

    def metrics(self, X: np.ndarray) -> Dict[str, float]:
        return {"quantization_error": self.quantization_error(X),
                "topographic_error": self.topographic_error(X),
                "dead_neurons": self.dead_ratio(X)}

    def save(self, path: str, ds: Dataset) -> None:
        np.savez(path, W=self.W, rows=self.rows, cols=self.cols, features=np.array(ds.features),
                 log_features=np.array(ds.log_features), mean=ds.mean, std=ds.std)


# =========================================================================== entrenamiento
def train_som(X: np.ndarray, rows: int, cols: int, seed: int | None, *, iterations: int, init: str,
              lr0: float, lr1: float, r0: float | None, r1: float) -> SOM:
    som = SOM(rows, cols, X.shape[1], np.random.default_rng(seed))
    som.init_weights(X, init)
    som.train(X, iterations, lr0, lr1, r0, r1)
    return som


def _run_seeds(base_seed: int | None, n_runs: int) -> List[int | None]:
    if base_seed is None:
        return [None] * n_runs
    return [base_seed + i for i in range(n_runs)]


def fit_best_som(ds: Dataset, cfg: Dict[str, Any]) -> tuple[SOM, pd.DataFrame]:
    """Entrena --n-runs SOMs con la configuración elegida y devuelve la mejor.

    Criterio: entre las corridas con error topográfico y neuronas muertas <= sus medianas, la de
    menor error de cuantización. Elegir sólo por TE o QE favorece mapas "comprimidos" en pocas
    neuronas (muchas muertas); los filtros por mediana descartan esas corridas atípicas.
    Devuelve también las métricas de todas las corridas.
    """
    soms, rows = [], []
    for run, seed in enumerate(_run_seeds(cfg["seed"], cfg["n_runs"])):
        som = train_som(ds.X, cfg["grid_rows"], cfg["grid_cols"], seed,
                        iterations=cfg["iterations"], init=cfg["init_method"],
                        lr0=cfg["initial_lr"], lr1=cfg["final_lr"],
                        r0=cfg["initial_radius"], r1=cfg["final_radius"])
        soms.append(som)
        rows.append({"run": run, "seed": seed, **som.metrics(ds.X)})
    df = pd.DataFrame(rows)
    ok = df[(df["topographic_error"] <= df["topographic_error"].median())
            & (df["dead_neurons"] <= df["dead_neurons"].median())]
    if ok.empty:
        ok = df
    best_run = int(ok.sort_values(["quantization_error", "dead_neurons"]).iloc[0]["run"])
    df["selected"] = df["run"] == best_run
    return soms[best_run], df


def neuron_regions(som: SOM, n_regions: int) -> np.ndarray:
    """Agrupa las neuronas (Ward sobre los pesos) en regiones interpretables -> (rows, cols)."""
    if n_regions <= 1 or som.n_neurons <= 1:
        return np.zeros((som.rows, som.cols), dtype=int)
    labels = fcluster(linkage(som.W, method="ward"), t=n_regions, criterion="maxclust") - 1
    return labels.reshape(som.rows, som.cols)


# =========================================================================== gráficos
def _grid_heatmap(Z, title, text=None, hover=None, colorscale=SEQUENTIAL, zmid=None, fmt=".2f"):
    rows, cols = Z.shape
    fig = go.Figure(go.Heatmap(
        z=Z, x=list(range(cols)), y=list(range(rows)), colorscale=colorscale, zmid=zmid,
        text=text, texttemplate="%{text}" if text is not None else f"%{{z:{fmt}}}",
        customdata=hover, hovertemplate=("%{customdata}<extra></extra>" if hover is not None else None),
        xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed", title="fila", dtick=1)
    fig.update_xaxes(title="columna", dtick=1)
    fig.update_layout(title=title, width=120 + 90 * cols, height=160 + 90 * rows)
    return fig


def _names_per_neuron(som: SOM, bmus: np.ndarray, names: np.ndarray) -> np.ndarray:
    lab = np.full((som.rows, som.cols), "", dtype=object)
    for k in range(som.n_neurons):
        members = names[bmus == k]
        lab[k // som.cols, k % som.cols] = "<br>".join(members) if len(members) else "(vacía)"
    return lab


def plot_som(som: SOM, ds: Dataset, pca_scores: pd.DataFrame | None, regions: np.ndarray, out: str) -> None:
    X, names = ds.X, ds.names
    bmus = som.bmus(X)
    hits = som.hits(X)
    who = _names_per_neuron(som, bmus, names)

    # Hit map con los nombres en el hover
    save_fig(_grid_heatmap(hits, "Cantidad de creadores por neurona", text=hits.astype(str),
                           hover=who), out, "hits")

    # Matriz U
    U = som.u_matrix()
    save_fig(_grid_heatmap(U, "Matriz U (distancia media a vecinas)", hover=who, colorscale="Greys"),
             out, "u_matrix")

    # Regiones (con la U de fondo para ver fronteras)
    reg_txt = np.array([[f"R{regions[r, c]}<br>({hits[r, c]})" for c in range(som.cols)]
                        for r in range(som.rows)])
    k = int(regions.max()) + 1
    colors = px.colors.qualitative.Set2
    discrete = []  # escala escalonada: un color plano por región
    for i in range(k):
        c = colors[i % len(colors)]
        discrete += [[i / k, c], [(i + 1) / k, c]]
    fig = _grid_heatmap(regions, "Regiones de la grilla (Ward sobre pesos) y hits", text=reg_txt,
                        hover=who, colorscale=discrete)
    fig.update_traces(zmin=-0.5, zmax=k - 0.5, showscale=False)
    save_fig(fig, out, "regiones")

    # Planos de componentes
    n = len(ds.features)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[short(f) for f in ds.features],
                        horizontal_spacing=0.03, vertical_spacing=0.05)
    Wg = som.W.reshape(som.rows, som.cols, -1)
    for j, f in enumerate(ds.features):
        fig.add_trace(go.Heatmap(z=Wg[:, :, j], colorscale=DIVERGING, zmid=0, zmin=-2.5, zmax=2.5,
                                 showscale=(j == 0), customdata=who,
                                 hovertemplate=f"{f}: %{{z:.2f}}<br>%{{customdata}}<extra></extra>"),
                      row=j // ncols + 1, col=j % ncols + 1)
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title="Planos de componentes (pesos estandarizados por feature)",
                      height=230 * nrows, width=1150)
    save_fig(fig, out, "planos_componentes")

    # PC1 / PC2 promedio por neurona (para comparar con el biplot)
    if pca_scores is not None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("PC1 media por neurona", "PC2 media por neurona"))
        for j, pc in enumerate(("PC1", "PC2")):
            Z = np.full(som.n_neurons, np.nan)
            for k in range(som.n_neurons):
                if (bmus == k).any():
                    Z[k] = pca_scores[pc].to_numpy()[bmus == k].mean()
            fig.add_trace(go.Heatmap(z=Z.reshape(som.rows, som.cols), colorscale=DIVERGING, zmid=0,
                                     texttemplate="%{z:.1f}", customdata=who, xgap=2, ygap=2,
                                     colorbar=dict(x=0.45 if j == 0 else 1.0),
                                     hovertemplate="%{customdata}<extra></extra>"), row=1, col=j + 1)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=160 + 90 * som.rows, width=2 * (160 + 90 * som.cols))
        save_fig(fig, out, "pc_por_neurona")


# =========================================================================== tareas
def assignment_table(som: SOM, ds: Dataset, regions: np.ndarray) -> pd.DataFrame:
    b = som.bmus(ds.X)
    df = ds.ids.copy()
    df["neuron"] = b
    df["row"], df["col"] = b // som.cols, b % som.cols
    df["region"] = regions.ravel()[b]
    df["dist_bmu"] = som.distances(ds.X)[np.arange(len(b)), b]
    return df


def region_profiles(assign: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Medias de las variables crudas por región + temática dominante (para interpretar)."""
    raw = ds.raw.drop(columns=["creator_id", "name"])
    prof = raw.groupby(assign["region"].to_numpy()).mean()
    prof.insert(0, "n_creadores", assign["region"].value_counts().sort_index())
    topics = prof.filter(like="topic_")
    prof.insert(1, "tematica_dominante", topics.idxmax(axis=1).str.replace("topic_", "").str.replace("_pct", ""))
    prof.insert(2, "miembros", assign.groupby("region")["name"].apply(lambda s: " ".join(s)))
    prof.index.name = "region"
    return prof


def run_kohonen(ds: Dataset, cfg: Dict[str, Any], pca_scores: pd.DataFrame | None = None) -> tuple[SOM, pd.DataFrame]:
    out = out_dir(cfg, "kohonen")
    t0 = time.time()
    som, runs = fit_best_som(ds, cfg)
    save_csv(runs.round(5), out, "metricas_corridas")
    regions = neuron_regions(som, cfg["n_regions"])
    assign = assignment_table(som, ds, regions)
    save_csv(assign.round(4), out, "asignacion")
    save_csv(region_profiles(assign, ds).round(2).T, out, "perfil_regiones", index=True)
    plot_som(som, ds, pca_scores, regions, out)
    som.save(os.path.join(out, "som_model.npz"), ds)

    m = runs[["quantization_error", "topographic_error", "dead_neurons"]]
    best = som.metrics(ds.X)
    print(f"[kohonen] Grilla {som.rows}x{som.cols}, {cfg['iterations']} it, lr {cfg['initial_lr']}->{cfg['final_lr']}, "
          f"R {cfg['initial_radius'] or som.default_radius()}->{cfg['final_radius']}, init={cfg['init_method']}, "
          f"{cfg['n_runs']} corridas ({time.time() - t0:.1f}s).")
    print(f"[kohonen]   QE {m.quantization_error.mean():.3f}±{m.quantization_error.std():.3f}  "
          f"TE {m.topographic_error.mean():.3f}±{m.topographic_error.std():.3f}  "
          f"muertas {m.dead_neurons.mean():.1%}±{m.dead_neurons.std():.1%}")
    print(f"[kohonen]   Mejor corrida: QE {best['quantization_error']:.3f}, TE {best['topographic_error']:.3f}, "
          f"muertas {best['dead_neurons']:.1%}")
    for r, g in assign.groupby("region"):
        print(f"[kohonen]   R{r} ({len(g)}): {', '.join(g['name'])}")
    print(f"[kohonen] Salidas en {out}/")
    return som, assign


def run_sweep(ds: Dataset, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = out_dir(cfg, "sweep")
    iters_list = cfg["sweep_iterations"] or [cfg["iterations"]]
    combos = list(itertools.product(cfg["sweep_grids"], cfg["sweep_lrs"], cfg["sweep_radii"], iters_list))
    print(f"[sweep] {len(combos)} configuraciones x {cfg['n_runs']} corridas...")
    rows = []
    t0 = time.time()
    for (r, c), lr0, r0, iters in combos:
        for run, seed in enumerate(_run_seeds(cfg["seed"], cfg["n_runs"])):
            som = train_som(ds.X, r, c, seed, iterations=iters, init=cfg["init_method"],
                            lr0=lr0, lr1=cfg["final_lr"], r0=r0, r1=cfg["final_radius"])
            rows.append({"grid": f"{r}x{c}", "neurons": r * c, "initial_lr": lr0,
                         "initial_radius": r0 if r0 is not None else som.default_radius(),
                         "radius_cfg": "auto" if r0 is None else r0,
                         "iterations": iters, "run": run, **som.metrics(ds.X)})
    df = pd.DataFrame(rows)
    save_csv(df.round(5), out, "sweep_corridas")

    keys = ["grid", "neurons", "initial_lr", "radius_cfg", "iterations"]
    agg = (df.groupby(keys, sort=False)[["quantization_error", "topographic_error", "dead_neurons"]]
             .agg(["mean", "std"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().sort_values(["neurons", "initial_lr"])
    save_csv(agg.round(5), out, "sweep_resumen")

    # Un panel por métrica; una línea por (lr, radio, iteraciones); eje x = cantidad de neuronas.
    metrics = [("quantization_error", "Error de cuantización"), ("topographic_error", "Error topográfico"),
               ("dead_neurons", "Neuronas muertas")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t for _, t in metrics])
    palette = px.colors.qualitative.Dark24
    for i, ((lr0, rc, it), g) in enumerate(agg.groupby(["initial_lr", "radius_cfg", "iterations"], sort=False)):
        label = f"lr0={lr0}, R0={rc}, it={it}"
        for j, (m, _) in enumerate(metrics):
            fig.add_scatter(x=g["grid"], y=g[f"{m}_mean"], error_y=dict(type="data", array=g[f"{m}_std"]),
                            mode="lines+markers", name=label, legendgroup=label, showlegend=(j == 0),
                            line_color=palette[i % len(palette)], row=1, col=j + 1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=3)
    fig.update_layout(title=f"Barrido de hiperparámetros ({cfg['n_runs']} corridas por punto, media ± desvío)",
                      height=520, width=1400)
    save_fig(fig, out, "sweep")

    print(f"[sweep] Listo en {time.time() - t0:.1f}s. Resumen (media):")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(agg[keys + ["quantization_error_mean", "topographic_error_mean", "dead_neurons_mean"]]
              .round(3).to_string(index=False))
    print(f"[sweep] Salidas en {out}/")
    return agg
