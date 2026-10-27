"""Análisis genérico de redes de Hopfield (ortogonalidad, capacidad, ruido, energía y espurios).

Todas las funciones trabajan con cualquier conjunto de patrones bipolares {-1, +1} cuya
grilla visual sea `shape = (filas, columnas)` (letras 5x5, códigos QR 37x37, etc.).
Ninguna depende de un caso de uso concreto: el módulo de cada tarea (`letters.py`, `qr.py`)
solo aporta sus patrones y llama a `run_full_analysis`.
"""

__all__ = [
    "HOPFIELD_CAPACITY_RATIO",
    "TARGET_IDX",
    "AnalysisParams",
    "OrthogonalityResult",
    "AnalysisResults",
    "theoretical_capacity",
    "add_pattern_heatmap",
    "plot_stored_patterns",
    "analyze_orthogonality",
    "find_globally_optimal_sequence",
    "build_orthogonal_sequence",
    "analyze_capacity",
    "analyze_random_capacity",
    "evaluate_noise_sweep",
    "critical_noise_level",
    "analyze_spurious_states",
    "print_summary",
    "run_full_analysis",
]

from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .hopfield import HopfieldNetwork, HopfieldResult, apply_noise, hamming_distance
from .plots import BINARY_SCALE, DIVERGING, format_ascii_grid, save_csv, save_fig

# Capacidad teórica de Hopfield con patrones aleatorios: P_max ≈ 0.138 · N
HOPFIELD_CAPACITY_RATIO = 0.138
TARGET_IDX = 1  

_EPS = 1e-12
_PALETTE = ["#1e8449", "#2980b9", "#c0392b", "#f39c12", "#8e44ad", "#16a085", "#d35400", "#2c3e50"]
_DIFF_SCALE = [[0.0, "#f5f6fa"], [0.5, "#e74c3c"], [1.0, "#2c3e50"]]
_OUTCOME_META = [
    ("recovery_rate", "Recuperado exacto (ξ^μ)", "#1e8449"),
    ("other_stored_rate", "Otro patrón almacenado (ξ^ν)", "#2980b9"),
    ("spurious_rate", "Estado espurio (mezcla)", "#e67e22"),
    ("inverted_rate", "Patrón invertido (-ξ)", "#8e44ad"),
    ("cycle_rate", "Ciclo / no converge", "#c0392b"),
]
_OUTCOME_KEYS = ("recovered", "other_stored", "inverted", "spurious", "cycle")


# ──────────────────────────────────────────────────────────────────────────────
# Parámetros y resultados
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AnalysisParams:
    seed: int | None
    update_mode: str
    max_iterations: int
    noise_levels: Tuple[float, ...]
    trials: int
    noise_type: str
    spurious_noise: float
    capacity_noise: float = 0.12
    capacity_trials: int = 15
    max_capacity_p: int = 10
    spurious_trials: int | None = None
    rule: str = "hebb"

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "AnalysisParams":
        seed = cfg.get("seed")
        spurious_trials = cfg.get("spurious_trials")
        return cls(
            seed=None if seed is None else int(seed),
            update_mode=str(cfg["update_mode"]),
            max_iterations=int(cfg["max_iterations"]),
            noise_levels=tuple(float(x) for x in cfg["noise_levels"]),
            trials=int(cfg["trials"]),
            noise_type=str(cfg.get("type", "white")),
            spurious_noise=float(cfg["spurious_noise"]),
            capacity_noise=float(cfg.get("capacity_noise", 0.12)),
            capacity_trials=int(cfg.get("capacity_trials", 15)),
            max_capacity_p=int(cfg.get("max_capacity_p", 10)),
            spurious_trials=None if spurious_trials is None else int(spurious_trials),
            rule=str(cfg.get("rule", "hebb")),
        )


@dataclass
class OrthogonalityResult:
    """Resultado del análisis de ortogonalidad entre patrones almacenados."""

    matrix: pd.DataFrame
    pairs: pd.DataFrame
    mean_abs_overlap: float
    max_abs_overlap: float
    random_expected_abs_overlap: float  # E|m| esperado entre patrones aleatorios: sqrt(2 / (pi N))


@dataclass
class AnalysisResults:
    """Todas las tablas generadas por `run_full_analysis`."""

    stored: pd.DataFrame
    orthogonality: OrthogonalityResult
    capacity: pd.DataFrame
    random_capacity: pd.DataFrame | None
    noise_summary: pd.DataFrame
    noise_by_pattern: pd.DataFrame
    spurious: pd.DataFrame


@dataclass
class _Example:
    level: float
    query: np.ndarray
    final_state: np.ndarray
    label: str
    iterations: int
    energy: float


@dataclass
class _SpuriousRecord:
    state_type: str
    matched: str | None
    source: str
    noise_level: float
    query: np.ndarray
    final_state: np.ndarray
    energy: float
    iterations: int
    converged: bool
    is_fixed_point: bool
    overlaps: np.ndarray
    count: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
def theoretical_capacity(n_neurons: int) -> float:
    """Capacidad teórica aproximada 0.138·N (patrones aleatorios, recuperación con pocos errores)."""
    return HOPFIELD_CAPACITY_RATIO * n_neurons


def _grid_dims(n_items: int, max_cols: int = 8) -> Tuple[int, int]:
    """Devuelve (filas, columnas) para disponer `n_items` subplots en una grilla de ancho acotado."""
    n_cols = max(1, min(n_items, max_cols))
    n_rows = -(-n_items // n_cols)
    return n_rows, n_cols


def _check_shape(net: HopfieldNetwork, shape: Tuple[int, int]) -> None:
    if shape[0] * shape[1] != net.n_neurons:
        raise ValueError(
            f"shape={shape} no coincide con la cantidad de neuronas de la red ({net.n_neurons})."
        )


def add_pattern_heatmap(
    fig: go.Figure,
    vec: np.ndarray,
    shape: Tuple[int, int],
    row: int,
    col: int,
    n_cols: int,
    diff_ref: np.ndarray | None = None,
    show_values: bool | None = None,
) -> None:
    """Agrega un patrón (vector bipolar) como grilla `shape` al subplot (row, col) de `fig`."""
    mat = np.asarray(vec).reshape(shape)
    n_cells = shape[0] * shape[1]
    if show_values is None:
        show_values = n_cells <= 100

    if diff_ref is not None:
        ref = np.asarray(diff_ref).reshape(shape)
        z = np.where(mat != ref, 0.0, mat.astype(np.float64))
        colorscale: Any = _DIFF_SCALE
    else:
        z = mat.astype(np.float64)
        colorscale = BINARY_SCALE

    kwargs: Dict[str, Any] = {
        "z": z,
        "zmin": -1,
        "zmax": 1,
        "colorscale": colorscale,
        "showscale": False,
        "xgap": 2 if show_values else 0,
        "ygap": 2 if show_values else 0,
    }
    if show_values:
        kwargs["text"] = np.where(mat > 0, "+1", "-1").tolist()
        kwargs["texttemplate"] = "%{text}"
        kwargs["textfont"] = {"size": 10}
    fig.add_trace(go.Heatmap(**kwargs), row=row, col=col)

    axis_idx = (row - 1) * n_cols + col
    anchor = "x" if axis_idx == 1 else f"x{axis_idx}"
    fig.update_xaxes(
        showticklabels=False, showgrid=False, zeroline=False, constrain="domain", row=row, col=col
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        autorange="reversed",
        scaleanchor=anchor,
        scaleratio=1,
        constrain="domain",
        row=row,
        col=col,
    )


def _add_capacity_vline(fig: go.Figure, x: float, n_cols: int) -> None:
    for col in range(1, n_cols + 1):
        loc: Dict[str, Any] = {"row": 1, "col": col}
        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color="#555555",
            annotation_text=f"C ≈ 0.138·N = {x:.2f}",
            annotation_position="top right",
            **loc,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Patrones almacenados y estabilidad
# ──────────────────────────────────────────────────────────────────────────────
def plot_stored_patterns(
    net: HopfieldNetwork,
    out: str,
    shape: Tuple[int, int],
    item_label: str = "patrón",
) -> pd.DataFrame:
    """Grafica los patrones almacenados y guarda tabla de energía/estabilidad."""
    _check_shape(net, shape)
    p = net.n_patterns
    rows_info: List[Dict[str, Any]] = []
    titles: List[str] = []
    for idx, name in enumerate(net.names):
        pat = net.patterns[idx]
        e = net.energy(pat)
        stable = net.is_fixed_point(pat)
        aligned = pat.astype(np.float64) * net.local_field(pat)
        rows_info.append({
            "pattern": name,
            "energy": round(e, 4),
            "is_fixed_point": stable,
            "unstable_pixels": int(np.count_nonzero(aligned < -_EPS)),
            "min_stability_margin": round(float(aligned.min()), 4),
            "pixels_pos": int(np.count_nonzero(pat == 1)),
            "pixels_neg": int(np.count_nonzero(pat == -1)),
        })
        titles.append(f"{name}<br>E = {e:.2f} ({'estable' if stable else 'inestable'})")

    df_stored = pd.DataFrame(rows_info)
    save_csv(df_stored, out, "patrones_almacenados")

    n_rows, n_cols = _grid_dims(p)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.18 / n_rows,
    )
    for idx in range(p):
        add_pattern_heatmap(fig, net.patterns[idx], shape, idx // n_cols + 1, idx % n_cols + 1, n_cols)
    fig.update_layout(
        title=f"Patrones ({item_label}s) {shape[0]}×{shape[1]} almacenados en la red de Hopfield",
        height=300 * n_rows + 60,
        width=max(230 * n_cols, 700),
    )
    save_fig(fig, out, "patrones_almacenados")
    return df_stored


# ──────────────────────────────────────────────────────────────────────────────
# 2. Ortogonalidad
# ──────────────────────────────────────────────────────────────────────────────
def analyze_orthogonality(
    net: HopfieldNetwork,
    out: str,
    item_label: str = "patrón",
) -> OrthogonalityResult:
    """Matriz de solapamientos m_{μν} = ξ^μ·ξ^ν / N, tabla de pares y comparación con patrones aleatorios."""
    overlap = net.orthogonality_matrix()
    dots = net.dot_product_matrix()
    p = net.n_patterns
    n = net.n_neurons

    df_mat = pd.DataFrame(overlap.round(4), index=net.names, columns=net.names)
    save_csv(df_mat, out, "ortogonalidad_matriz", index=True)

    pair_rows: List[Dict[str, Any]] = []
    for i in range(p):
        for j in range(i + 1, p):
            ov_ij = float(overlap[i, j])
            pair_rows.append({
                "pair": f"{net.names[i]}-{net.names[j]}",
                "pattern_1": net.names[i],
                "pattern_2": net.names[j],
                "dot_product": int(dots[i, j]),
                "normalized_overlap": round(ov_ij, 4),
                "abs_overlap": round(abs(ov_ij), 4),
                "hamming_distance": hamming_distance(net.patterns[i], net.patterns[j]),
                "angle_deg": round(float(np.degrees(np.arccos(np.clip(ov_ij, -1.0, 1.0)))), 2),
            })
    pair_columns = [
        "pair", "pattern_1", "pattern_2", "dot_product",
        "normalized_overlap", "abs_overlap", "hamming_distance", "angle_deg",
    ]
    df_pairs = (
        pd.DataFrame(pair_rows, columns=pair_columns)
        .sort_values("abs_overlap", ascending=False)
        .reset_index(drop=True)
    )
    save_csv(df_pairs, out, "ortogonalidad_pares")

    off_diag = np.abs(overlap[~np.eye(p, dtype=bool)])
    mean_abs = float(off_diag.mean()) if off_diag.size else 0.0
    max_abs = float(off_diag.max()) if off_diag.size else 0.0
    random_expected = float(np.sqrt(2.0 / (np.pi * n)))

    show_text = p <= 12
    text_annot = (
        [[f"m={overlap[i, j]:+.2f}<br>⟨ξ,ξ⟩={int(dots[i, j])}" for j in range(p)] for i in range(p)]
        if show_text
        else None
    )
    heat_kwargs: Dict[str, Any] = {
        "z": overlap,
        "x": net.names,
        "y": net.names,
        "zmin": -1.0,
        "zmax": 1.0,
        "colorscale": DIVERGING,
        "colorbar": {"title": "Solapamiento m"},
    }
    if text_annot is not None:
        heat_kwargs["text"] = text_annot
        heat_kwargs["texttemplate"] = "%{text}"
    fig = go.Figure(data=go.Heatmap(**heat_kwargs))
    side = int(np.clip(110 * p + 160, 480, 1000))
    fig.update_layout(
        title=(
            f"Matriz de ortogonalidad entre {item_label}s "
            f"(|m| medio = {mean_abs:.3f}, |m| máx = {max_abs:.3f}; "
            f"aleatorios: E|m| ≈ {random_expected:.3f})"
        ),
        xaxis_title="Patrón",
        yaxis_title="Patrón",
        yaxis_autorange="reversed",
        height=side,
        width=side + 80,
    )
    save_fig(fig, out, "ortogonalidad")

    return OrthogonalityResult(
        matrix=df_mat,
        pairs=df_pairs,
        mean_abs_overlap=mean_abs,
        max_abs_overlap=max_abs,
        random_expected_abs_overlap=random_expected,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Capacidad de almacenamiento
# ──────────────────────────────────────────────────────────────────────────────
def find_globally_optimal_sequence(
    pool: Mapping[str, np.ndarray],
    target_p: int,
    rule: str = "hebb",
) -> List[str]:
    """Evalúa todas las combinaciones posibles de tamaño `target_p` en el pool y devuelve
    aquella que maximiza la cantidad de patrones estables (y minimiza el solapamiento máximo).
    """
    pool_keys = list(pool.keys())
    if target_p > len(pool_keys):
        return pool_keys

    best_combo: List[str] = []
    max_stable = -1
    min_max_ov = float("inf")

    for combo in itertools.combinations(pool_keys, target_p):
        sub_pats = np.stack([pool[n] for n in combo], axis=0)
        sub_net = HopfieldNetwork(sub_pats, list(combo), rule=rule)

        n_stable = sum(1 for pat in sub_net.patterns if sub_net.is_fixed_point(pat))
        ov = sub_net.orthogonality_matrix()
        off_diag = np.abs(ov[~np.eye(target_p, dtype=bool)])
        max_ov = float(off_diag.max()) if off_diag.size else 0.0

        if n_stable > max_stable or (n_stable == max_stable and max_ov < min_max_ov):
            max_stable = n_stable
            min_max_ov = max_ov
            best_combo = list(combo)

    return best_combo


def build_orthogonal_sequence(
    seed_names: Sequence[str],
    pool: Mapping[str, np.ndarray],
    max_p: int = 10,
) -> List[str]:
    """Mantenida por compatibilidad: selecciona iterativamente con heurística greedy."""
    chosen = [str(n) for n in seed_names]
    unknown = [n for n in chosen if n not in pool]
    if unknown:
        raise ValueError(f"Patrones fuera del pool: {unknown}")
    n_neurons = len(next(iter(pool.values())))
    remaining = [k for k in pool if k not in chosen]

    def _max_overlap(cand: str) -> float:
        return max(
            abs(int(pool[cand].astype(np.int32) @ pool[k].astype(np.int32))) / n_neurons
            for k in chosen
        )

    while len(chosen) < max_p and remaining:
        best = min(remaining, key=_max_overlap)
        chosen.append(best)
        remaining.remove(best)
    return chosen[:max_p]

def analyze_capacity(
    params: AnalysisParams,
    out: str,
    pool: Mapping[str, np.ndarray],
    sequences: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Estudia la capacidad evaluando la combinación globalmente óptima que maximiza
    los patrones estables para cada P, además de las secuencias pasadas explícitamente.
    """
    rng = np.random.default_rng(params.seed)
    n_neurons = len(next(iter(pool.values())))
    cap = theoretical_capacity(n_neurons)
    n_flip = int(round(params.capacity_noise * n_neurons))

    eval_sequences: Dict[str, List[List[str]]] = {}

    # 1. Búsqueda combinatoria de la secuencia óptima para cada P
    max_p = min(params.max_capacity_p, len(pool))
    optimal_by_p = [
        find_globally_optimal_sequence(pool, target_p=p, rule=params.rule)
        for p in range(1, max_p + 1)
    ]
    eval_sequences["Secuencia óptima (máx. estabilidad)"] = optimal_by_p

    # 2. Agregar secuencias adicionales enviadas desde la llamada si difieren
    for seq_name, seq_list in sequences.items():
        eval_sequences[seq_name] = [list(seq_list[:p]) for p in range(1, min(max_p, len(seq_list)) + 1)]

    rows: List[Dict[str, Any]] = []
    max_p_plotted = 1

    for group_label, seq_p_list in eval_sequences.items():
        for p_idx, sub_names in enumerate(seq_p_list, start=1):
            p = len(sub_names)
            sub_pats = np.stack([pool[n] for n in sub_names], axis=0)
            sub_net = HopfieldNetwork(sub_pats, sub_names, rule=params.rule)
            max_p_plotted = max(max_p_plotted, p)

            ov = sub_net.orthogonality_matrix()
            off = np.abs(ov[~np.eye(p, dtype=bool)])
            mean_abs_ov = float(off.mean()) if off.size else 0.0

            n_stable = sum(1 for pat in sub_net.patterns if sub_net.is_fixed_point(pat))

            hits = 0
            for idx, pat in enumerate(sub_net.patterns):
                for _ in range(params.capacity_trials):
                    q = apply_noise(pat, params.capacity_noise, rng, noise_type="white")
                    res = sub_net.predict(
                        q,
                        update_mode=params.update_mode,
                        max_iterations=params.max_iterations,
                        rng=rng,
                    )
                    outcome, _, _ = sub_net.classify_state(res.final_state, res.converged, target_idx=idx)
                    if outcome == "recovered":
                        hits += 1

            rows.append({
                "sequence_type": group_label,
                "n_patterns": p,
                "load_alpha": round(p / n_neurons, 4),
                "patterns": ",".join(sub_names),
                "mean_abs_overlap": round(mean_abs_ov, 4),
                "stable_patterns": n_stable,
                "stable_ratio": round(n_stable / p, 4),
                "recovery_rate_noise": round(hits / (p * params.capacity_trials), 4),
                "noise_level": params.capacity_noise,
            })

    df_cap = pd.DataFrame(rows)
    save_csv(df_cap, out, "capacidad")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Fracción de patrones estables (sin ruido)",
            f"Tasa de recuperación con ruido ({params.capacity_noise:.0%} = {n_flip} píxeles)",
        ),
    )
    for g_idx, group_label in enumerate(eval_sequences):
        grp = df_cap[df_cap["sequence_type"] == group_label]
        color = _PALETTE[g_idx % len(_PALETTE)]
        for col, y_col, show_legend in (
            (1, "stable_ratio", True),
            (2, "recovery_rate_noise", False),
        ):
            fig.add_scatter(
                x=grp["n_patterns"],
                y=grp[y_col],
                mode="lines+markers",
                name=group_label,
                legendgroup=group_label,
                showlegend=show_legend,
                line={"color": color, "width": 2.5},
                hovertext=grp["patterns"],
                row=1,
                col=col,
            )
    if cap <= max_p_plotted + 1:
        _add_capacity_vline(fig, cap, 2)
    for col in (1, 2):
        fig.update_xaxes(title_text="Cantidad de patrones almacenados (P)", dtick=1, row=1, col=col)
        fig.update_yaxes(range=[-0.05, 1.05], title_text="Proporción", row=1, col=col)
    fig.update_layout(
        title=(
            f"Capacidad de almacenamiento (N = {n_neurons}, "
            f"capacidad teórica 0.138·N ≈ {cap:.1f} patrones)"
        ),
        height=460,
        width=1100,
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
    )
    save_fig(fig, out, "capacidad")
    return df_cap


def analyze_random_capacity(
    params: AnalysisParams,
    out: str,
    n_neurons: int,
    alphas: Sequence[float] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30),
    n_repeats: int = 3,
    n_probe: int = 5,
) -> pd.DataFrame:
    """Capacidad con patrones aleatorios para N neuronas, barriendo la carga α = P/N."""
    rng = np.random.default_rng(None if params.seed is None else params.seed + 31)
    bits = np.array([-1, 1], dtype=np.int8)

    rows: List[Dict[str, Any]] = []
    seen_p: set[int] = set()
    for alpha in alphas:
        p = max(1, int(round(alpha * n_neurons)))
        if p in seen_p:
            continue
        seen_p.add(p)
        stable_ratios: List[float] = []
        hits = 0
        total = 0
        for _ in range(n_repeats):
            pats = rng.choice(bits, size=(p, n_neurons))
            net = HopfieldNetwork(pats)
            stable_ratios.append(sum(1 for x in net.patterns if net.is_fixed_point(x)) / p)
            probe = rng.choice(p, size=min(n_probe, p), replace=False)
            for i in probe:
                q = apply_noise(net.patterns[int(i)], params.capacity_noise, rng, noise_type="white")
                res = net.predict(
                    q,
                    update_mode=params.update_mode,
                    max_iterations=params.max_iterations,
                    rng=rng,
                )
                outcome, _, _ = net.classify_state(res.final_state, res.converged, target_idx=int(i))
                hits += int(outcome == "recovered")
                total += 1
        rows.append({
            "n_neurons": n_neurons,
            "n_patterns": p,
            "load_alpha": round(p / n_neurons, 4),
            "stable_ratio": round(float(np.mean(stable_ratios)), 4),
            "recovery_rate_noise": round(hits / total, 4),
            "noise_level": params.capacity_noise,
        })

    df = pd.DataFrame(rows)
    save_csv(df, out, "capacidad_aleatoria")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Fracción de patrones estables (sin ruido)",
            f"Recuperación con ruido ({params.capacity_noise:.0%})",
        ),
    )
    for col, y_col in ((1, "stable_ratio"), (2, "recovery_rate_noise")):
        fig.add_scatter(
            x=df["load_alpha"],
            y=df[y_col],
            mode="lines+markers",
            name="Patrones aleatorios",
            showlegend=(col == 1),
            line={"color": _PALETTE[1], "width": 2.5},
            hovertext=[f"P={int(v)}" for v in df["n_patterns"]],
            row=1,
            col=col,
        )
        loc: Dict[str, Any] = {"row": 1, "col": col}
        fig.add_vline(
            x=HOPFIELD_CAPACITY_RATIO,
            line_dash="dash",
            line_color="#555555",
            annotation_text="α_c ≈ 0.138",
            annotation_position="top right",
            **loc,
        )
        fig.update_xaxes(title_text="Carga α = P / N", row=1, col=col)
        fig.update_yaxes(range=[-0.05, 1.05], title_text="Proporción", row=1, col=col)
    fig.update_layout(
        title=f"Capacidad con patrones aleatorios (N = {n_neurons})",
        height=460,
        width=1100,
    )
    save_fig(fig, out, "capacidad_aleatoria")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. Recuperación vs ruido, iteraciones y energía
# ──────────────────────────────────────────────────────────────────────────────
def critical_noise_level(df_summary: pd.DataFrame, threshold: float = 0.5) -> float | None:
    """Menor nivel de ruido evaluado en el que la tasa de recuperación cae por debajo de `threshold`."""
    below = df_summary[df_summary["recovery_rate"] < threshold]
    if below.empty:
        return None
    return float(below["noise_level"].min())


def evaluate_noise_sweep(
    net: HopfieldNetwork,
    params: AnalysisParams,
    out: str,
    shape: Tuple[int, int],
    max_examples: int = 8,
    max_energy_panels: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recuperación para cada nivel de ruido y patrón; grafica métricas, ejemplos y energía por paso."""
    _check_shape(net, shape)
    rng = np.random.default_rng(params.seed)
    n = net.n_neurons
    levels = sorted(set(params.noise_levels))

    trial_records: List[Dict[str, Any]] = []
    energy_records: List[Dict[str, Any]] = []
    showcase: Dict[str, List[_Example]] = {name: [] for name in net.names}

    for level in levels:
        for pat_idx, name in enumerate(net.names):
            target = net.patterns[pat_idx]
            for t in range(params.trials):
                query = apply_noise(target, level, rng, noise_type=params.noise_type, shape=shape)
                res = net.predict(
                    query,
                    update_mode=params.update_mode,
                    max_iterations=params.max_iterations,
                    rng=rng,
                )
                outcome, matched_name, _ = net.classify_state(
                    res.final_state, res.converged, target_idx=pat_idx
                )
                ham_out = hamming_distance(res.final_state, target)
                ov_target = float(target.astype(np.float64) @ res.final_state.astype(np.float64)) / n

                trial_records.append({
                    "noise_level": level,
                    "flipped_pixels": hamming_distance(query, target),
                    "pattern": name,
                    "trial": t,
                    "outcome": outcome,
                    "matched_pattern": matched_name or outcome,
                    "converged": res.converged,
                    "cycle_detected": res.cycle_detected,
                    "iterations": res.iterations,
                    "hamming_out": ham_out,
                    "pixel_accuracy": 1.0 - ham_out / n,
                    "target_overlap": ov_target,
                    "initial_energy": float(res.energy_history[0]),
                    "final_energy": float(res.energy_history[-1]),
                })

                if t == 0:
                    for step_idx, e_val in enumerate(res.energy_history):
                        energy_records.append({
                            "noise_level": level,
                            "pattern": name,
                            "step": step_idx,
                            "energy": float(e_val),
                            "outcome": outcome,
                        })
                    showcase[name].append(_Example(
                        level=level,
                        query=query,
                        final_state=res.final_state,
                        label=matched_name if matched_name else outcome,
                        iterations=res.iterations,
                        energy=float(res.energy_history[-1]),
                    ))

    df_trials = pd.DataFrame(trial_records)
    for key in _OUTCOME_KEYS:
        df_trials[f"is_{key}"] = df_trials["outcome"] == key

    df_summary = df_trials.groupby("noise_level", as_index=False).agg(
        recovery_rate=("is_recovered", "mean"),
        other_stored_rate=("is_other_stored", "mean"),
        inverted_rate=("is_inverted", "mean"),
        spurious_rate=("is_spurious", "mean"),
        cycle_rate=("is_cycle", "mean"),
        mean_pixel_accuracy=("pixel_accuracy", "mean"),
        mean_iterations=("iterations", "mean"),
        std_iterations=("iterations", lambda s: float(np.std(s.to_numpy(dtype=np.float64)))),
        mean_final_energy=("final_energy", "mean"),
    )
    df_summary.insert(1, "flipped_pixels", (df_summary["noise_level"] * n).round().astype(int))
    df_summary = df_summary.round(4)
    save_csv(df_summary, out, "recuperacion_por_ruido")

    df_by_pattern = df_trials.groupby(["pattern", "noise_level"], as_index=False).agg(
        recovery_rate=("is_recovered", "mean"),
        spurious_rate=("is_spurious", "mean"),
        inverted_rate=("is_inverted", "mean"),
        cycle_rate=("is_cycle", "mean"),
        mean_pixel_accuracy=("pixel_accuracy", "mean"),
        mean_iterations=("iterations", "mean"),
    ).round(4)
    save_csv(df_by_pattern, out, "recuperacion_por_patron")

    # Gráfico 1: estados finales e iteraciones vs ruido
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Distribución de estados alcanzados según nivel de ruido",
            "Iteraciones (barridos) promedio hasta convergencia",
        ),
    )
    for col_name, label, color in _OUTCOME_META:
        fig.add_scatter(
            x=df_summary["noise_level"],
            y=df_summary[col_name],
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 2.5},
            row=1,
            col=1,
        )
    fig.add_scatter(
        x=df_summary["noise_level"],
        y=df_summary["mean_iterations"],
        error_y={"type": "data", "array": df_summary["std_iterations"], "visible": True},
        mode="lines+markers",
        name="Iteraciones ± σ",
        line={"color": "#2c3e50", "width": 2.5},
        row=1,
        col=2,
    )
    for col in (1, 2):
        fig.update_xaxes(title_text="Nivel de ruido (fracción de píxeles invertidos)", row=1, col=col)
    fig.update_yaxes(title_text="Proporción de consultas", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Cantidad de barridos", row=1, col=2)
    fig.update_layout(
        title=(
            f"Desempeño de recuperación asociativa ({params.update_mode}, ruido={params.noise_type}, "
            f"P={net.n_patterns}, N={n})"
        ),
        height=460,
        width=1120,
    )
    save_fig(fig, out, "recuperacion_por_ruido")

    # Gráfico 2: energía por paso
    df_energy = pd.DataFrame(energy_records)
    save_csv(df_energy, out, "energia_por_paso")

    panel_names = net.names[:max_energy_panels]
    fig_e = make_subplots(
        rows=1,
        cols=len(panel_names),
        subplot_titles=[f"Consultas de {name}" for name in panel_names],
        shared_yaxes=True,
    )
    for pat_idx, name in enumerate(panel_names):
        sub_e = df_energy[df_energy["pattern"] == name]
        for lvl_idx, lvl in enumerate(levels):
            grp = sub_e[sub_e["noise_level"] == lvl]
            fig_e.add_scatter(
                x=grp["step"],
                y=grp["energy"],
                mode="lines+markers",
                name=f"ruido={lvl:g}",
                legendgroup=f"ruido={lvl:g}",
                showlegend=(pat_idx == 0),
                line={"color": _PALETTE[lvl_idx % len(_PALETTE)], "width": 2},
                row=1,
                col=pat_idx + 1,
            )
        e_pat = net.energy(net.patterns[pat_idx])
        loc: Dict[str, Any] = {"row": 1, "col": pat_idx + 1}
        fig_e.add_hline(
            y=e_pat,
            line_dash="dot",
            line_color="#333333",
            annotation_text=f"E({name})={e_pat:.2f}",
            annotation_position="bottom right",
            **loc,
        )
        fig_e.update_xaxes(title_text="Paso (barrido)", dtick=1, row=1, col=pat_idx + 1)
    fig_e.update_yaxes(title_text="Energía E(s) = -½ sᵀWs", row=1, col=1)
    fig_e.update_layout(
        title=f"Evolución de la energía por paso ({params.update_mode})",
        height=440,
        width=max(290 * len(panel_names), 900),
    )
    save_fig(fig_e, out, "energia_por_paso")

    # Gráfico 3: consulta ruidosa -> estado final por nivel de ruido, para cada patrón objetivo
    for target_idx, target_name in enumerate(net.names):
        examples = showcase[target_name]
        if not examples:
            continue
        if len(examples) > max_examples:
            keep = np.unique(np.linspace(0, len(examples) - 1, max_examples).round().astype(int))
            examples = [examples[int(i)] for i in keep]

        n_cols = len(examples)
        target_pattern = net.patterns[target_idx]
        sub_titles: List[str] = []
        for ex in examples:
            flips = hamming_distance(ex.query, target_pattern)
            sub_titles.append(f"Ruido {ex.level:.0%} ({flips} px)<br>Entrada")
        for ex in examples:
            sub_titles.append(f"Final: {ex.label} ({ex.iterations} it)<br>E = {ex.energy:.2f}")

        fig_ex = make_subplots(
            rows=2,
            cols=n_cols,
            subplot_titles=sub_titles,
            vertical_spacing=0.14,
            horizontal_spacing=0.04,
        )
        for col_idx, ex in enumerate(examples, start=1):
            add_pattern_heatmap(fig_ex, ex.query, shape, 1, col_idx, n_cols, diff_ref=target_pattern)
            add_pattern_heatmap(fig_ex, ex.final_state, shape, 2, col_idx, n_cols, diff_ref=target_pattern)
        fig_ex.update_layout(
            title=(
                f"Recuperación visual de '{target_name}' según nivel de ruido "
                "(rojo = píxel distinto del patrón original)"
            ),
            height=540,
            width=max(210 * n_cols, 750),
        )
        save_fig(fig_ex, out, f"ejemplos_recuperacion_{target_idx}")

    return df_summary, df_by_pattern


# ──────────────────────────────────────────────────────────────────────────────
# 5. Estados espurios
# ──────────────────────────────────────────────────────────────────────────────
def _register_spurious(
    found: Dict[bytes, _SpuriousRecord],
    net: HopfieldNetwork,
    query: np.ndarray,
    res: HopfieldResult,
    outcome: str,
    matched: str | None,
    source: str,
    noise_level: float,
) -> None:
    key = res.final_state.tobytes()
    existing = found.get(key)
    if existing is not None:
        existing.count += 1
        return
    found[key] = _SpuriousRecord(
        state_type=outcome,
        matched=matched,
        source=source,
        noise_level=noise_level,
        query=query.copy(),
        final_state=res.final_state.copy(),
        energy=float(res.energy_history[-1]),
        iterations=res.iterations,
        converged=res.converged,
        is_fixed_point=net.is_fixed_point(res.final_state),
        overlaps=net.overlaps(res.final_state),
    )


def analyze_spurious_states(
    net: HopfieldNetwork,
    params: AnalysisParams,
    out: str,
    shape: Tuple[int, int],
) -> pd.DataFrame:
    """Ingresa consultas muy ruidosas y cataloga los atractores que no son un patrón almacenado."""
    _check_shape(net, shape)
    rng = np.random.default_rng(None if params.seed is None else params.seed + 997)
    n = net.n_neurons
    n_search = params.spurious_trials if params.spurious_trials is not None else max(params.trials * 4, 80)
    found: Dict[bytes, _SpuriousRecord] = {}

    for idx, name in enumerate(net.names):
        pat = net.patterns[idx]
        for _ in range(n_search):
            q = apply_noise(pat, params.spurious_noise, rng, noise_type=params.noise_type, shape=shape)
            res = net.predict(
                q, update_mode=params.update_mode, max_iterations=params.max_iterations, rng=rng
            )
            outcome, matched, _ = net.classify_state(res.final_state, res.converged, target_idx=idx)
            if outcome in ("spurious", "inverted", "cycle"):
                _register_spurious(found, net, q, res, outcome, matched, name, params.spurious_noise)

    if net.n_patterns >= 3 and not any(r.state_type == "spurious" for r in found.values()):
        mix_raw = net.patterns[:3].astype(np.int32).sum(axis=0)
        q_mix = np.where(mix_raw >= 0, 1, -1).astype(np.int8)
        res = net.predict(
            q_mix, update_mode=params.update_mode, max_iterations=params.max_iterations, rng=rng
        )
        outcome, matched, _ = net.classify_state(res.final_state, res.converged)
        if outcome in ("spurious", "cycle"):
            _register_spurious(
                found, net, q_mix, res, outcome, matched,
                f"mix({','.join(net.names[:3])})",
                round(hamming_distance(q_mix, net.patterns[0]) / n, 2),
            )

    if not any(r.state_type == "inverted" for r in found.values()):
        q_inv = apply_noise(net.patterns[0], 0.80, rng, noise_type="white")
        res = net.predict(
            q_inv, update_mode=params.update_mode, max_iterations=params.max_iterations, rng=rng
        )
        outcome, matched, _ = net.classify_state(res.final_state, res.converged)
        if outcome == "inverted":
            _register_spurious(found, net, q_inv, res, outcome, matched, net.names[0], 0.80)

    priority = {"spurious": 0, "inverted": 1, "cycle": 2}
    items = sorted(
        found.values(),
        key=lambda r: (priority.get(r.state_type, 3), -r.count, r.energy),
    )

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        label = item.matched or ("ciclo" if item.state_type == "cycle" else "mezcla_espuria")
        ov_cols = {
            f"overlap_{nm}": round(float(ov), 4) for nm, ov in zip(net.names, item.overlaps)
        }
        rows.append({
            "spurious_id": f"S{idx}",
            "state_type": item.state_type,
            "matched_label": label,
            "source_pattern": item.source,
            "query_noise_level": item.noise_level,
            "occurrences": item.count,
            "energy": round(item.energy, 4),
            "is_fixed_point": item.is_fixed_point,
            "iterations": item.iterations,
            "min_hamming_to_stored": min(hamming_distance(item.final_state, p) for p in net.patterns),
            **ov_cols,
        })
    df_spurious = pd.DataFrame(rows)
    save_csv(df_spurious, out, "estados_espurios")

    showcase = items[:4]
    if showcase:
        n_cols = len(showcase)
        titles: List[str] = [
            f"Consulta #{i} (origen: {it.source}, ruido={it.noise_level:.0%})"
            for i, it in enumerate(showcase, start=1)
        ]
        for it in showcase:
            if it.state_type == "inverted":
                tipo = f"Invertido ({it.matched})"
            elif it.state_type == "spurious":
                tipo = "Mezcla espuria"
            else:
                tipo = "Ciclo"
            if net.n_patterns <= 6:
                ov_str = " ".join(f"{nm}:{ov:+.2f}" for nm, ov in zip(net.names, it.overlaps))
                titles.append(f"{tipo} | E={it.energy:.2f}<br>m=[{ov_str}]")
            else:
                best = int(np.argmax(np.abs(it.overlaps)))
                titles.append(
                    f"{tipo} | E={it.energy:.2f}<br>|m| máx = {abs(it.overlaps[best]):.2f} ({net.names[best]})"
                )

        fig = make_subplots(
            rows=2,
            cols=n_cols,
            subplot_titles=titles,
            vertical_spacing=0.16,
            horizontal_spacing=0.05,
        )
        for col_idx, it in enumerate(showcase, start=1):
            add_pattern_heatmap(fig, it.query, shape, 1, col_idx, n_cols)
            add_pattern_heatmap(fig, it.final_state, shape, 2, col_idx, n_cols)
        fig.update_layout(
            title="Identificación de estados espurios a partir de patrones muy ruidosos",
            height=560,
            width=max(260 * n_cols, 760),
        )
        save_fig(fig, out, "estados_espurios")

    return df_spurious


# ──────────────────────────────────────────────────────────────────────────────
# Orquestación y resumen por consola
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(
    net: HopfieldNetwork,
    results: AnalysisResults,
    params: AnalysisParams,
    shape: Tuple[int, int],
    out: str,
    tag: str = "hopfield",
    item_label: str = "patrón",
    ascii_preview: bool | None = None,
) -> None:
    """Imprime un resumen del análisis (ortogonalidad, capacidad, ruido, espurios)."""
    n = net.n_neurons
    if ascii_preview is None:
        ascii_preview = n <= 100

    print(
        f"\n[{tag}] {item_label.capitalize()}s almacenados ({net.n_patterns}): "
        f"{', '.join(net.names)} (N = {n} neuronas, grilla {shape[0]}×{shape[1]})"
    )
    for idx, name in enumerate(net.names):
        e = float(results.stored["energy"].iloc[idx])
        st = "estable" if bool(results.stored["is_fixed_point"].iloc[idx]) else "INESTABLE"
        print(f"  {name}: E = {e:.2f}, {st}")
        if ascii_preview:
            for line in format_ascii_grid(net.patterns[idx], shape).splitlines():
                print(f"    {line}")

    orth = results.orthogonality
    msg = (
        f"\n[{tag}] Ortogonalidad: |m| medio = {orth.mean_abs_overlap:.3f} "
        f"(aleatorios: ≈ {orth.random_expected_abs_overlap:.3f}) | |m| máx = {orth.max_abs_overlap:.3f}"
    )
    if not orth.pairs.empty:
        top = orth.pairs.iloc[0]
        msg += (
            f" | par más correlacionado: {top['pair']} "
            f"(m = {float(top['normalized_overlap']):+.2f}, Hamming = {int(top['hamming_distance'])})"
        )
    print(msg)

    cap = theoretical_capacity(n)
    n_stable = int(results.stored["is_fixed_point"].sum())
    print(
        f"[{tag}] Capacidad teórica (0.138·N): {cap:.2f} patrones "
        f"(con P={net.n_patterns}: {n_stable}/{net.n_patterns} puntos fijos estables)"
    )
    if results.random_capacity is not None and not results.random_capacity.empty:
        rc = results.random_capacity
        fail = rc[rc["stable_ratio"] < 0.99]
        if fail.empty:
            print(f"[{tag}] Capacidad aleatoria: estable en todo el rango (α ≤ {rc['load_alpha'].max():.2f}).")
        else:
            print(
                f"[{tag}] Capacidad aleatoria: primeros patrones inestables en α ≈ "
                f"{float(fail['load_alpha'].min()):.3f} (P = {int(fail['n_patterns'].min())})."
            )

    print(f"\n[{tag}] Recuperación por nivel de ruido ({params.noise_type}, {params.update_mode}):")
    cols = [
        "noise_level", "flipped_pixels", "recovery_rate", "spurious_rate",
        "inverted_rate", "cycle_rate", "mean_iterations",
    ]
    print(results.noise_summary[cols].to_string(index=False))
    crit = critical_noise_level(results.noise_summary, 0.5)
    if crit is None:
        print(f"[{tag}] La recuperación se mantiene ≥ 50 % en todos los niveles evaluados.")
    else:
        print(f"[{tag}] Ruido crítico (recuperación < 50 %): {crit:.0%} de los píxeles.")

    sp = results.spurious
    n_mix = int((sp["state_type"] == "spurious").sum()) if "state_type" in sp.columns else 0
    n_inv = int((sp["state_type"] == "inverted").sum()) if "state_type" in sp.columns else 0
    print(
        f"\n[{tag}] Estados espurios identificados (ruido={params.spurious_noise:.0%}): "
        f"{len(sp)} distintos ({n_mix} mezclas espurias, {n_inv} patrones invertidos)"
    )
    print(f"[{tag}] Gráficos HTML y tablas CSV guardados en: {out}/")


def run_full_analysis(
    net: HopfieldNetwork,
    params: AnalysisParams,
    out: str,
    shape: Tuple[int, int],
    pool: Mapping[str, np.ndarray],
    capacity_sequences: Mapping[str, Sequence[str]],
    item_label: str = "patrón",
    tag: str = "hopfield",
    include_random_capacity: bool = False,
    ascii_preview: bool | None = None,
) -> AnalysisResults:
    """Ejecuta todo el análisis sobre `net` y devuelve las tablas."""
    _check_shape(net, shape)
    df_stored = plot_stored_patterns(net, out, shape, item_label)
    orth = analyze_orthogonality(net, out, item_label)
    df_cap = analyze_capacity(params, out, pool, capacity_sequences)
    df_rand = analyze_random_capacity(params, out, net.n_neurons) if include_random_capacity else None
    df_noise, df_by_pattern = evaluate_noise_sweep(net, params, out, shape)
    df_spurious = analyze_spurious_states(net, params, out, shape)

    results = AnalysisResults(
        stored=df_stored,
        orthogonality=orth,
        capacity=df_cap,
        random_capacity=df_rand,
        noise_summary=df_noise,
        noise_by_pattern=df_by_pattern,
        spurious=df_spurious,
    )
    print_summary(net, results, params, shape, out, tag, item_label, ascii_preview)
    return results