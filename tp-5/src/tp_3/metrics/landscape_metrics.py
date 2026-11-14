"""Charts and flatness / critical-point analysis for the loss landscape (reads landscape.npz / .json).

  landscape_surface.html         3D surface (alpha, beta, loss) with theta*, minima and saddles marked
  landscape_contour.html         same, as contour map
  landscape_slices.html          loss along alpha (beta = 0) and along beta (alpha = 0)
  landscape_radial.html          loss vs distance to theta* (min / mean / max over all directions)
  landscape_metrics.html         flatness numbers
  landscape_critical_points.html minima (with valley depth), saddles and maxima
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]  # circular order


def _save(fig: go.Figure, path: Path, title: str, height: int = 600, **layout) -> None:
    fig.update_layout(title_text=title, template="plotly_white", height=height, **layout)
    fig.write_html(path, include_plotlyjs="directory", config={"responsive": True, "displaylogo": False})


# ----------------------------------------------------------------------- analysis
def discrete_critical_points(z: np.ndarray, eps: float) -> dict[str, list[tuple[int, int]]]:
    """Interior grid points classified with the 8-neighbour sign pattern (differences below eps count as equal).
    min: nothing lower and something higher; max: the opposite; saddle: >= 4 sign changes around the point."""
    found: dict[str, list[tuple[int, int]]] = {"min": [], "max": [], "saddle": []}
    for j in range(1, z.shape[0] - 1):
        for i in range(1, z.shape[1] - 1):
            diff = np.array([z[j + dj, i + di] - z[j, i] for dj, di in NEIGHBORS])
            s = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
            if (s >= 0).all() and (s > 0).any():
                found["min"].append((j, i))
            elif (s <= 0).all() and (s < 0).any():
                found["max"].append((j, i))
            else:
                nz = s[s != 0]
                if len(nz) >= 4 and int(np.sum(nz != np.roll(nz, 1))) >= 4:
                    found["saddle"].append((j, i))
    return found


def minima_persistence(z: np.ndarray, eps: float) -> list[dict[str, Any]]:
    """Valleys of the grid via merge-tree persistence: raise the 'water level' point by point; when two basins
    meet, the one with the shallower bottom dies. depth = pass height - basin bottom."""
    n_j, n_i = z.shape
    order = np.argsort(z, axis=None, kind="stable")
    parent = np.full(z.size, -1)
    find = lambda a: a if parent[a] == a else find_compress(a)

    def find_compress(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    bottom: dict[int, int] = {}
    valleys: list[dict[str, Any]] = []
    for flat in order:
        j, i = divmod(int(flat), n_i)
        parent[flat] = flat
        bottom[flat] = flat
        for dj, di in NEIGHBORS:
            nj, ni = j + dj, i + di
            if 0 <= nj < n_j and 0 <= ni < n_i and parent[nj * n_i + ni] != -1:
                a, b = find_compress(flat), find_compress(nj * n_i + ni)
                if a == b:
                    continue
                if z.flat[bottom[a]] > z.flat[bottom[b]] or (z.flat[bottom[a]] == z.flat[bottom[b]] and a > b):
                    a, b = b, a                       # b is the shallower basin: it dies here
                dead = bottom.pop(b)
                depth = float(z[j, i] - z.flat[dead])
                if depth > eps:
                    valleys.append({"index": divmod(int(dead), n_i), "loss": float(z.flat[dead]), "depth": depth,
                                    "pass_loss": float(z[j, i]), "pass_index": (j, i)})
                parent[b] = a
    survivor = next(iter(bottom.values()))
    valleys.append({"index": divmod(int(survivor), n_i), "loss": float(z.flat[survivor]), "depth": None,
                    "pass_loss": None, "pass_index": None})
    for v in valleys:   # a bottom on the window edge may keep descending outside the window: not a confirmed minimum
        j, i = v["index"]
        v["on_border"] = j in (0, n_j - 1) or i in (0, n_i - 1)
    return sorted(valleys, key=lambda v: v["loss"])


def analyze(alphas, betas, z, meta) -> dict[str, Any]:
    finite = np.isfinite(z)
    z = np.where(finite, z, np.nanmax(np.where(finite, z, np.nan)))
    center, tol = meta["center_loss"], meta["tol"]
    a_mesh, b_mesh = np.meshgrid(alphas, betas)
    radius = np.hypot(a_mesh, b_mesh)
    span = float(z.max() - z.min())
    eps = max(1e-3 * span, 1e-12)
    crit = discrete_critical_points(z, eps)
    valleys = minima_persistence(z, eps)
    saddles = {pt: "montura" for pt in crit["saddle"]}
    for v in valleys:
        if v["pass_index"] is not None and not v["on_border"]:
            saddles.setdefault(tuple(v["pass_index"]), "paso entre valles")

    ax = meta["axis"]
    h = meta["h"]
    curv = {"alpha": (ax["alpha_plus"] + ax["alpha_minus"] - 2 * center) / h ** 2,
            "beta": (ax["beta_plus"] + ax["beta_minus"] - 2 * center) / h ** 2}
    threshold = center + tol * (z.max() - center)
    flat = z <= threshold
    flat_radius = float(radius[~flat].min()) if (~flat).any() else float(radius.max())
    sharp = {}
    for frac in (0.1, 0.25, 0.5):
        inside = radius <= frac * meta["range"]
        sharp[frac] = float(z[inside].max() - center) if inside.any() else float("nan")
    jm, im = np.unravel_index(np.argmin(z), z.shape)
    return {"z": z, "radius": radius, "eps": eps, "crit": crit, "valleys": valleys, "saddles": saddles, "curv": curv,
            "flat_fraction": float(flat.mean()), "flat_radius": flat_radius, "sharp": sharp,
            "grid_min": float(z.min()), "grid_min_at": (float(alphas[im]), float(betas[jm])), "grid_max": float(z.max()),
            "n_nonfinite": int((~finite).sum())}


# -------------------------------------------------------------------------- charts
def write_landscape_plots(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    meta = json.loads((out / "landscape.json").read_text())
    data = np.load(out / "landscape.npz")
    alphas, betas = data["alphas"], data["betas"]
    A = analyze(alphas, betas, data["z"], meta)
    z, log = A["z"], meta.get("zscale") == "log"
    shown = np.log10(np.maximum(z, 1e-300)) if log else z
    zlabel = "log10(loss)" if log else "Loss"
    to_show = (lambda v: np.log10(max(v, 1e-300))) if log else (lambda v: v)

    # markers: center, global/local minima, saddles
    markers: list[dict[str, Any]] = [{"name": "θ* (modelo entrenado)", "a": 0.0, "b": 0.0, "z": to_show(meta["center_loss"]),
                                      "color": "#000000", "symbol": "diamond"}]
    for k, v in enumerate(A["valleys"]):
        j, i = v["index"]
        name = ("Mínimo en el borde (puede seguir bajando)" if v["on_border"]
                else "Mínimo global (grilla)" if k == 0 else "Mínimo local")
        markers.append({"name": name, "a": float(alphas[i]), "b": float(betas[j]), "z": shown[j, i], "symbol": "circle",
                        "color": "#999999" if v["on_border"] else "#2ca02c" if k == 0 else "#ff7f0e"})
    for (j, i) in A["saddles"]:
        markers.append({"name": "Punto de montura", "a": float(alphas[i]), "b": float(betas[j]), "z": shown[j, i],
                        "color": "#d62728", "symbol": "x"})

    def marker_traces(three_d: bool) -> list:
        traces, seen = [], set()
        for m in markers:
            kw = dict(x=[m["a"]], y=[m["b"]], mode="markers", name=m["name"], legendgroup=m["name"],
                      showlegend=m["name"] not in seen, marker={"size": 7 if three_d else 11, "color": m["color"],
                                                                "symbol": m["symbol"], "line": {"width": 1, "color": "white"}})
            traces.append(go.Scatter3d(z=[m["z"]], **kw) if three_d else go.Scatter(**kw))
            seen.add(m["name"])
        return traces

    sub = (f"{meta['model_type']} ({meta['activation']}), N={meta['n_params']:,}, datos={meta['data']} "
           f"({meta['n_samples']:,}), grilla {meta['grid']}×{meta['grid']} en [-{meta['range']:g}, {meta['range']:g}]")
    fig = go.Figure(go.Surface(x=alphas, y=betas, z=shown, colorscale="Viridis", colorbar={"title": zlabel},
                               contours={"z": {"show": True, "usecolormap": True, "project": {"z": True}}}))
    for t in marker_traces(True):
        fig.add_trace(t)
    fig.update_layout(scene={"xaxis_title": "α (dirección 1)", "yaxis_title": "β (dirección 2)", "zaxis_title": zlabel})
    _save(fig, out / "landscape_surface.html", f"Superficie de loss — {sub}", height=720)

    fig = go.Figure(go.Contour(x=alphas, y=betas, z=shown, colorscale="Viridis", ncontours=25,
                               colorbar={"title": zlabel}, line={"width": 0.5}))
    for t in marker_traces(False):
        fig.add_trace(t)
    fig.update_xaxes(title_text="α", scaleanchor="y")
    fig.update_yaxes(title_text="β")
    _save(fig, out / "landscape_contour.html", f"Curvas de nivel — {sub}", height=650, hovermode="closest")

    mid = len(alphas) // 2
    fig = go.Figure([go.Scatter(x=alphas, y=shown[mid, :], mode="lines+markers", name="a lo largo de α (β=0)"),
                     go.Scatter(x=betas, y=shown[:, mid], mode="lines+markers", name="a lo largo de β (α=0)")])
    fig.add_vline(x=0, line_dash="dash", line_color="#888")
    fig.update_xaxes(title_text="Desplazamiento")
    fig.update_yaxes(title_text=zlabel)
    _save(fig, out / "landscape_slices.html", "Cortes 1D por θ*", height=500, hovermode="x unified")

    r, flat_z = A["radius"].ravel(), shown.ravel()
    edges = np.linspace(0, float(r.max()), 13)
    which = np.digitize(r, edges[1:-1])
    centers = [(edges[k] + edges[k + 1]) / 2 for k in range(12) if (which == k).any()]
    stat = lambda f: [float(f(flat_z[which == k])) for k in range(12) if (which == k).any()]
    fig = go.Figure([go.Scatter(x=centers, y=stat(np.max), mode="lines", name="máximo", line={"color": "#d62728"}),
                     go.Scatter(x=centers, y=stat(np.mean), mode="lines+markers", name="media", line={"color": "#1f77b4"}),
                     go.Scatter(x=centers, y=stat(np.min), mode="lines", name="mínimo", line={"color": "#2ca02c"})])
    fig.add_hline(y=to_show(meta["center_loss"]), line_dash="dash", line_color="#888", annotation_text="loss en θ*")
    fig.update_xaxes(title_text="Distancia a θ*  √(α²+β²)")
    fig.update_yaxes(title_text=zlabel)
    _save(fig, out / "landscape_radial.html", "Loss vs distancia a θ* (todas las direcciones del plano)", height=500,
          hovermode="x unified")

    sh = A["sharp"]
    rows = [("Loss en θ*", f"{meta['center_loss']:.6g}"),
            ("Mínimo en la grilla", f"{A['grid_min']:.6g} en (α={A['grid_min_at'][0]:.2f}, β={A['grid_min_at'][1]:.2f})"),
            ("Máximo en la grilla", f"{A['grid_max']:.6g}"),
            ("Curvatura en θ* (α, β)", f"{A['curv']['alpha']:.4g}, {A['curv']['beta']:.4g}"),
            (f"Fracción de la grilla con loss ≤ L*+{meta['tol']:.0%} del rango", f"{A['flat_fraction']:.1%}"),
            ("Radio plano (distancia al primer punto fuera de esa región)", f"{A['flat_radius']:.3f}  (de {meta['range']:g})"),
            ("Aumento máx. de loss a 10% / 25% / 50% del rango", f"{sh[0.1]:.4g} / {sh[0.25]:.4g} / {sh[0.5]:.4g}"),
            ("Mínimos interiores / en el borde / monturas / máximos (tolerancia %.2g)" % A["eps"],
             f"{sum(not v['on_border'] for v in A['valleys'])} / {sum(v['on_border'] for v in A['valleys'])} / "
             f"{len(A['saddles'])} / {len(A['crit']['max'])}"),
            ("N parámetros (pesos + sesgos)", f"{meta['n_params']:,} ({meta['n_weights']:,} + {meta['n_biases']:,})"),
            ("cos(d1, d2)", f"{meta['cosine_d1_d2']:.2e}"), ("Dirección de sesgos", meta["bias_direction"])]
    fig = go.Figure(go.Table(header={"values": ["Medida", "Valor"], "fill_color": "#eaeaea", "align": "left"},
                             cells={"values": [[a for a, _ in rows], [b for _, b in rows]], "align": "left"}))
    _save(fig, out / "landscape_metrics.html", "Planitud del mínimo", height=480)

    table = []
    for k, v in enumerate(A["valleys"]):
        j, i = v["index"]
        label = ("Mínimo en el borde" if v["on_border"] else "Mínimo global (grilla)" if k == 0 else "Mínimo local")
        detail = "—" if v["depth"] is None else f"profundidad {v['depth']:.4g}; paso a loss {v['pass_loss']:.4g}"
        table.append((label, alphas[i], betas[j], v["loss"],
                      detail + ("; la loss puede seguir bajando fuera de la grilla" if v["on_border"] else "")))
    for (j, i), kind in A["saddles"].items():
        table.append(("Punto de montura", alphas[i], betas[j], float(z[j, i]), kind))
    for j, i in A["crit"]["max"]:
        table.append(("Máximo local", alphas[i], betas[j], float(z[j, i]), ""))
    fig = go.Figure(go.Table(
        header={"values": ["Tipo", "α", "β", "Loss", "Detalle"], "fill_color": "#eaeaea", "align": "left"},
        cells={"values": [[t[0] for t in table], [f"{t[1]:.3f}" for t in table], [f"{t[2]:.3f}" for t in table],
                          [f"{t[3]:.6g}" for t in table], [t[4] for t in table]], "align": "left"}))
    _save(fig, out / "landscape_critical_points.html", "Mínimos, monturas y máximos en el plano", height=300 + 28 * len(table))

    files = [out / f"landscape_{n}.html" for n in ("surface", "contour", "slices", "radial", "metrics", "critical_points")]
    print("Charts written:", ", ".join(f.name for f in files))
    for label, value in rows:
        print(f"  {label}: {value}")
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild the loss landscape charts")
    ap.add_argument("--out-dir", default="metrics/landscape")
    write_landscape_plots(ap.parse_args().out_dir)